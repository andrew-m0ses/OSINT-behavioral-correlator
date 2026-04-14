"""
osint/pipeline.py
Data management pipeline: collect → extract → pair → train.

Ground truth strategy:
  - Mine self-disclosures from GitHub (twitter_username field in API)
  - Mine bio links from HN (github.com/xxx in about field)
  - Use union-find to group accounts by identity
  - Build positive pairs from groups, hard negatives from similarity
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from .collectors import collect_account, mine_identity_links, HNCollector, GitHubCollector
from .features import extract as extract_features, cosine_similarity


RAW_DIR    = Path("data/raw")
FEAT_DIR   = Path("data/features")
PAIRS_DIR  = Path("data/pairs")
MODELS_DIR = Path("data/models")
LINKS_FILE = Path("data/identity_links.jsonl")

MIN_POSTS  = 15   # Skip accounts with fewer posts
random.seed(42)


# ─── Storage helpers ──────────────────────────────────────────────────────────

def save_raw(data: dict):
    p = RAW_DIR / data["platform"] / f"{data['username']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


def load_raw(platform: str, username: str) -> Optional[dict]:
    p = RAW_DIR / platform / f"{username}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_features(feat: dict):
    p = FEAT_DIR / feat["platform"] / f"{feat['username']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(feat, f)


def load_features(platform: str, username: str) -> Optional[dict]:
    p = FEAT_DIR / platform / f"{username}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_all_features() -> dict:
    """Returns {"platform:username": feature_dict}"""
    all_f = {}
    for fp in FEAT_DIR.rglob("*.json"):
        try:
            d = json.loads(fp.read_text())
            all_f[f"{d['platform']}:{d['username']}"] = d
        except Exception:
            pass
    return all_f


def append_link(link: dict):
    LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LINKS_FILE, "a") as f:
        f.write(json.dumps(link) + "\n")


def load_links() -> list[dict]:
    if not LINKS_FILE.exists():
        return []
    links = []
    for line in LINKS_FILE.read_text().strip().splitlines():
        if line:
            try:
                links.append(json.loads(line))
            except Exception:
                pass
    return links


# ─── Collection ───────────────────────────────────────────────────────────────

def collect_and_save(platform: str, username: str, max_posts: int = 300,
                     force: bool = False) -> Optional[dict]:
    """
    Collect account data, cache to disk, extract features.
    Returns feature dict or None.
    """
    # Check cache
    if not force:
        feat = load_features(platform, username)
        if feat:
            return feat

    # Collect raw data
    data = collect_account(platform, username, max_posts=max_posts)
    if data is None or data["post_count"] < MIN_POSTS:
        return None

    save_raw(data)

    # Extract features
    feat = extract_features(data, use_embeddings=True)
    save_features(feat)

    # Mine identity links from profile
    links = mine_identity_links(platform, username)
    for link in links:
        append_link(link)

    return feat


def discover_and_collect(platform: str, n: int = 100, max_posts: int = 200):
    """Auto-discover users from a platform and collect all of them."""
    if platform == "hn":
        usernames = HNCollector().discover_active_users(n)
    elif platform == "github":
        usernames = GitHubCollector().discover_active_users(n)
    else:
        print(f"Discovery not implemented for {platform}")
        return

    print(f"Discovered {len(usernames)} {platform} users — collecting...")
    collected = 0
    for username in usernames:
        feat = collect_and_save(platform, username, max_posts=max_posts)
        if feat:
            collected += 1
    print(f"  Collected {collected}/{len(usernames)} accounts")


# ─── Identity Grouping (Union-Find) ──────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self._p = {}

    def find(self, x):
        self._p.setdefault(x, x)
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, x, y):
        self._p[self.find(x)] = self.find(y)

    def groups(self) -> dict:
        g = {}
        for x in self._p:
            r = self.find(x)
            g.setdefault(r, set()).add(x)
        return {r: list(v) for r, v in g.items() if len(v) > 1}


def build_identity_groups(all_features: dict) -> list[list[str]]:
    """
    Use known identity links + union-find to group accounts by person.
    """
    uf = UnionFind()
    links = load_links()
    keys  = set(all_features.keys())

    for link in links:
        a = link.get("from") or (link.get("account_a",{}).get("platform","") + ":" + link.get("account_a",{}).get("username",""))
        b = link.get("to")   or (link.get("account_b",{}).get("platform","") + ":" + link.get("account_b",{}).get("username",""))
        if a in keys and b in keys:
            uf.union(a, b)

    return list(uf.groups().values())


# ─── Pair Construction ────────────────────────────────────────────────────────

def build_positive_pairs(groups: list[list[str]]) -> list[tuple]:
    pairs = []
    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                pairs.append((g[i], g[j], 1))
    return pairs


def build_random_negatives(groups: list[list[str]], all_keys: list[str],
                           n: int = 10000) -> list[tuple]:
    same = set()
    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                same.add(tuple(sorted([g[i], g[j]])))

    # Group keys by platform for balanced sampling
    by_platform = {}
    for k in all_keys:
        plat = k.split(":")[0]
        by_platform.setdefault(plat, []).append(k)
    platforms = list(by_platform.keys())

    negs = []
    seen = set()
    attempts = 0
    # Build 50% cross-platform, 50% same-platform negatives
    while len(negs) < n and attempts < n * 20:
        if random.random() < 0.5 and len(platforms) >= 2:
            # Cross-platform negative
            p1, p2 = random.sample(platforms, 2)
            a = random.choice(by_platform[p1])
            b = random.choice(by_platform[p2])
        else:
            # Same-platform negative
            plat = random.choice(platforms)
            keys = by_platform[plat]
            if len(keys) < 2:
                attempts += 1; continue
            a, b = random.sample(keys, 2)
        if a == b:
            attempts += 1; continue
        pair = tuple(sorted([a, b]))
        if pair in same or pair in seen:
            attempts += 1; continue
        negs.append((a, b, 0))
        same.add(pair)
        attempts += 1
    return negs


def build_hard_negatives(all_features: dict, groups: list[list[str]],
                         n: int = 5000) -> list[tuple]:
    """
    Find negative pairs with HIGH similarity — the model's hardest cases.
    Computed using batch cosine similarity over all feature vectors.
    """
    keys = list(all_features.keys())
    vecs = np.array([all_features[k]["vector"] for k in keys], dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.where(norms == 0, 1, norms)

    same = set()
    for g in groups:
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                same.add(tuple(sorted([g[i], g[j]])))

    hard = []
    bs = 256
    for start in range(0, len(keys), bs):
        end = min(start + bs, len(keys))
        batch = vecs_n[start:end]         # (bs, D)
        sims  = batch @ vecs_n.T          # (bs, N)
        for i_local in range(len(batch)):
            i_global = start + i_local
            row = sims[i_local].copy()
            row[i_global] = -2            # Self
            top = np.argsort(row)[::-1][:20]
            for j in top:
                if j == i_global: continue
                pair = tuple(sorted([keys[i_global], keys[j]]))
                if pair in same: continue
                if row[j] < 0.3: break   # Not similar enough to be a hard negative
                hard.append((keys[i_global], keys[j], 0))
                same.add(pair)
                if len(hard) >= n: break
        if len(hard) >= n: break

    return hard[:n]


def identity_split(groups: list[list[str]], pos_pairs: list, neg_pairs: list,
                   val_r: float = 0.15, test_r: float = 0.10) -> dict:
    """
    Split by identity — same person never appears in both train and val/test.
    Critical to prevent data leakage.
    """
    shuffled = groups.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_r))
    n_val  = max(1, int(n * val_r))

    test_ids = set()
    for g in shuffled[:n_test]: test_ids.update(g)
    val_ids  = set()
    for g in shuffled[n_test:n_test+n_val]: val_ids.update(g)

    def route(pairs):
        tr, vl, te = [], [], []
        for p in pairs:
            if p[0] in test_ids or p[1] in test_ids: te.append(p)
            elif p[0] in val_ids or p[1] in val_ids:  vl.append(p)
            else:                                        tr.append(p)
        return tr, vl, te

    ptr, pvl, pte = route(pos_pairs)
    ntr, nvl, nte = route(neg_pairs)
    return {
        "train": ptr + ntr,
        "val":   pvl + nvl,
        "test":  pte + nte,
    }


def save_pairs(split: dict, all_features: dict):
    for name, pairs in split.items():
        if not pairs: continue
        random.shuffle(pairs)
        try:
            X_a = np.array([all_features[p[0]]["vector"] for p in pairs], dtype=np.float32)
            X_b = np.array([all_features[p[1]]["vector"] for p in pairs], dtype=np.float32)
            y   = np.array([p[2] for p in pairs], dtype=np.float32)
        except KeyError as e:
            print(f"Warning: missing features for {e}, skipping {name}")
            continue

        d = PAIRS_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "X_a.npy", X_a)
        np.save(d / "X_b.npy", X_b)
        np.save(d / "y.npy",   y)
        meta = [{"a": p[0], "b": p[1], "label": p[2]} for p in pairs]
        (d / "meta.json").write_text(json.dumps(meta))
        n_pos = int(y.sum())
        print(f"  {name}: {len(pairs):,} pairs ({n_pos} pos, {len(pairs)-n_pos} neg)")


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    accounts: list[tuple[str, str]],   # [(platform, username), ...]
    max_posts: int = 300,
    neg_ratio: int = 10,
    hard_neg_ratio: int = 5,
    force_collect: bool = False,
) -> bool:
    """
    End-to-end: collect → extract → pair. Returns True if pairs were built.
    """
    print("\n━━ STEP 1: Collect & Extract Features ━━")
    for platform, username in accounts:
        collect_and_save(platform, username, max_posts=max_posts, force=force_collect)

    print("\n━━ STEP 2: Load All Features ━━")
    all_features = load_all_features()
    print(f"  {len(all_features)} accounts with features")
    if len(all_features) < 2:
        print("Need at least 2 accounts. Add more targets.")
        return False

    print("\n━━ STEP 3: Identity Groups ━━")
    groups = build_identity_groups(all_features)
    print(f"  {len(groups)} identity groups from known links")

    if not groups:
        print("  No identity links found — pairs will be built without ground truth positives.")
        print("  Mine links by running: osint mine-links")
        # Still build negatives for evaluation purposes
        all_keys = list(all_features.keys())
        neg_pairs = build_random_negatives([], all_keys, n=min(500, len(all_keys)*10))
        split = {"train": neg_pairs[:int(len(neg_pairs)*0.7)],
                 "val":   neg_pairs[int(len(neg_pairs)*0.7):int(len(neg_pairs)*0.85)],
                 "test":  neg_pairs[int(len(neg_pairs)*0.85):]}
        save_pairs(split, all_features)
        return True

    print("\n━━ STEP 4: Build Pairs ━━")
    all_keys   = list(all_features.keys())
    pos_pairs  = build_positive_pairs(groups)
    rand_negs  = build_random_negatives(groups, all_keys, n=len(pos_pairs)*neg_ratio)
    hard_negs  = build_hard_negatives(all_features, groups, n=len(pos_pairs)*hard_neg_ratio)
    all_negs   = rand_negs + hard_negs

    print(f"  {len(pos_pairs)} positives | {len(all_negs)} negatives")

    print("\n━━ STEP 5: Split by Identity ━━")
    split = identity_split(groups, pos_pairs, all_negs)
    save_pairs(split, all_features)
    return True
