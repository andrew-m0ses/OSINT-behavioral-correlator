"""
osint/features.py
Behavioral feature extraction — converts raw post history into a fixed
feature vector for each account.

Feature groups:
  Temporal    (31-dim)  posting hour + day histograms
  Stylometric (55-dim)  sentence structure, punctuation, vocab, function words
  Behavioral  ( 8-dim)  activity patterns
  Semantic    (384-dim) mean-pooled sentence embeddings
  Platform    ( 6-dim)  one-hot (replaced by learned embedding at training)
  ─────────────────────
  Total       484-dim
"""

import re
import string
import warnings
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

PLATFORMS = ["hn", "github", "habr", "v2ex", "reddit", "youtube", "twitter", "telegram"]

# Function words — personal tics, highly discriminating across writers
FUNCTION_WORDS = [
    "actually","basically","honestly","literally","obviously","clearly",
    "definitely","probably","perhaps","maybe","just","really","very",
    "quite","rather","somewhat","fairly","pretty","tbh","imo","imho",
    "fwiw","afaik","iirc","btw","ngl","idk","iiuc","however","therefore",
    "furthermore","moreover","nevertheless","nonetheless","consequently",
    "although","whereas","thus","hence","indeed","admittedly",
]

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            _embed_model = False
    return _embed_model if _embed_model is not False else None


# ─── Text Utilities ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # Strip HTML
    text = re.sub(r"https?://\S+", " URL ", text)  # Replace URLs with token
    text = re.sub(r"```[\s\S]*?```", " CODE ", text)  # Code blocks
    text = re.sub(r"`[^`]+`", " CODE ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 4]


def words(text: str) -> list[str]:
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in t.split() if w and w not in ("url", "code")]


# ─── Temporal Features (31-dim) ───────────────────────────────────────────────

def temporal_features(posts: list[dict]) -> np.ndarray:
    """
    24-dim hour histogram + 7-dim weekday histogram.
    Captures when a person posts — a surprisingly stable personal signal.
    """
    hours = np.zeros(24)
    days  = np.zeros(7)
    for p in posts:
        ts = p.get("timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            hours[dt.hour]     += 1
            days[dt.weekday()] += 1
        except Exception:
            pass
    if hours.sum() > 0: hours /= hours.sum()
    if days.sum()  > 0: days  /= days.sum()
    return np.concatenate([hours, days])  # 31-dim


# ─── Stylometric Features (55-dim) ───────────────────────────────────────────

def stylometric_features(posts: list[dict]) -> np.ndarray:
    """
    Writing style fingerprint. These are highly personal and stable
    across platforms for the same author.
    """
    all_text  = " ".join(clean_text(p.get("text", "")) for p in posts)
    raw_text  = " ".join(p.get("text", "") for p in posts)
    if not all_text.strip():
        return np.zeros(55)

    sents = sentences(all_text)
    wds   = words(all_text)
    if not wds:
        return np.zeros(55)

    total_w = max(len(wds), 1)
    total_c = max(len(raw_text), 1)

    # Sentence length stats
    sent_lens = [len(words(s)) for s in sents if s]
    avg_sl = np.mean(sent_lens) if sent_lens else 0
    std_sl = np.std(sent_lens)  if sent_lens else 0

    # Vocabulary richness
    wc = Counter(wds)
    vocab  = len(wc)
    ttr    = vocab / np.sqrt(total_w)                          # Type-token ratio
    hapax  = sum(1 for v in wc.values() if v == 1) / max(vocab, 1)

    # Punctuation signatures
    punct_density    = sum(c in string.punctuation for c in raw_text) / total_c
    ellipsis_freq    = raw_text.count("...") / total_w
    exclamation_freq = raw_text.count("!")   / total_w
    question_freq    = raw_text.count("?")   / total_w
    caps_ratio       = sum(1 for w in raw_text.split() if len(w)>1 and w.isupper()) / total_w
    emdash_freq      = raw_text.count("—")   / total_w
    hyphen_freq      = raw_text.count("-")   / total_w
    comma_rate       = raw_text.count(",")   / max(len(sents), 1)
    paren_freq       = (raw_text.count("(") + raw_text.count(")")) / total_w

    # Developer signals
    code_blocks  = len(re.findall(r"```", raw_text)) / 2
    code_ratio   = code_blocks / max(len(posts), 1)
    link_count   = raw_text.count("URL")  # We tokenised URLs above
    link_rate    = link_count / max(len(posts), 1)
    quote_lines  = len(re.findall(r"(?m)^>", raw_text))
    quote_rate   = quote_lines / max(len(sents), 1)

    # Post length stats
    post_lens = [len(words(clean_text(p.get("text","")))) for p in posts]
    avg_pl = np.mean(post_lens) if post_lens else 0
    std_pl = np.std(post_lens)  if post_lens else 0

    scalar = np.array([
        np.tanh(avg_sl / 25),
        np.tanh(std_sl / 15),
        np.tanh(ttr    / 8),
        hapax,
        np.tanh(punct_density  * 40),
        np.tanh(ellipsis_freq  * 80),
        np.tanh(exclamation_freq * 80),
        np.tanh(question_freq  * 80),
        caps_ratio,
        np.tanh(emdash_freq    * 150),
        np.tanh(hyphen_freq    * 40),
        np.tanh(comma_rate     / 4),
        np.tanh(paren_freq     * 80),
        np.tanh(code_ratio     * 5),
        np.tanh(link_rate      * 3),
        np.tanh(quote_rate     * 10),
        np.tanh(avg_pl         / 80),
        np.tanh(std_pl         / 60),
    ])  # 18-dim

    # Function word frequencies (37-dim)
    fw = np.array([wc.get(w, 0) / total_w * 1000 for w in FUNCTION_WORDS[:37]])
    fw = np.tanh(fw / 5)  # Normalize

    return np.concatenate([scalar, fw])  # 55-dim


# ─── Behavioral Features (8-dim) ──────────────────────────────────────────────

def behavioral_features(posts: list[dict]) -> np.ndarray:
    """
    High-level activity patterns that reveal personality / work habits.
    """
    if not posts:
        return np.zeros(8)

    # Burst score: fraction of posts within 2-hour windows
    timestamps = []
    for p in posts:
        ts = p.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp())
            except Exception:
                pass
    timestamps.sort()
    burst_posts = 0
    i = 0
    while i < len(timestamps):
        j = i + 1
        while j < len(timestamps) and timestamps[j] - timestamps[i] < 7200:
            j += 1
        if j - i > 2:
            burst_posts += j - i
        i = j
    burst_score = burst_posts / max(len(timestamps), 1)

    # Weekend ratio
    weekend = sum(1 for p in posts if _is_weekend(p.get("timestamp","")))
    weekend_ratio = weekend / max(len(posts), 1)

    # Reply vs original post ratio
    replies = sum(1 for p in posts if p.get("metadata",{}).get("parent_id") or p.get("type") == "comment")
    reply_ratio = replies / max(len(posts), 1)

    # Post length consistency (low std = consistent writer)
    lens = [len(words(clean_text(p.get("text","")))) for p in posts]
    len_consistency = 1 - (np.std(lens) / max(np.mean(lens), 1)) if lens else 0

    # Question posts ratio
    q_posts = sum(1 for p in posts if "?" in (p.get("text","") or ""))
    question_ratio = q_posts / max(len(posts), 1)

    # Link posts ratio
    link_posts = sum(1 for p in posts if "http" in (p.get("text","") or ""))
    link_ratio = link_posts / max(len(posts), 1)

    # Activity spread (avg days between posts)
    if len(timestamps) >= 2:
        span = timestamps[-1] - timestamps[0]
        avg_gap_days = span / max(len(timestamps)-1, 1) / 86400
        spread = np.tanh(avg_gap_days / 3)
    else:
        spread = 0.5

    # Normalized post volume
    vol = np.tanh(len(posts) / 200)

    return np.array([
        burst_score,
        weekend_ratio,
        reply_ratio,
        np.tanh(max(0, len_consistency)),
        question_ratio,
        link_ratio,
        spread,
        vol,
    ])


def _is_weekend(ts_str: str) -> bool:
    if not ts_str:
        return False
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z","+00:00"))
        return dt.weekday() >= 5
    except Exception:
        return False


# ─── Semantic Embedding (384-dim) ─────────────────────────────────────────────

def semantic_features(posts: list[dict], max_posts: int = 100) -> np.ndarray:
    """
    Mean-pool sentence transformer embeddings over all posts.
    Captures WHAT and HOW they write — topic + phrasing fingerprint.
    Falls back to zeros if sentence-transformers not installed.
    """
    model = _get_embed_model()
    if model is None:
        return np.zeros(384)

    texts = [clean_text(p.get("text","")) for p in posts[:max_posts]]
    texts = [t for t in texts if len(t) > 20]
    if not texts:
        return np.zeros(384)

    embs = model.encode(texts, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True)
    return embs.mean(axis=0)  # 384-dim


# ─── Platform One-Hot (6-dim) ──────────────────────────────────────────────────

def platform_onehot(platform: str) -> np.ndarray:
    vec = np.zeros(len(PLATFORMS))
    if platform in PLATFORMS:
        vec[PLATFORMS.index(platform)] = 1.0
    return vec


# ─── Full Feature Vector ──────────────────────────────────────────────────────

def extract(account_data: dict, use_embeddings: bool = True) -> dict:
    """
    Main entry point. Returns a feature dict with the full 484-dim vector.
    """
    posts    = account_data.get("posts", [])
    platform = account_data.get("platform", "unknown")
    username = account_data.get("username", "unknown")

    temp  = temporal_features(posts)              # 31
    style = stylometric_features(posts)           # 55
    behav = behavioral_features(posts)            #  8
    sem   = semantic_features(posts) if use_embeddings else np.zeros(384)  # 384
    plat  = platform_onehot(platform)             #  6

    vec = np.concatenate([temp, style, behav, sem, plat])  # 484

    return {
        "platform": platform,
        "username": username,
        "post_count": len(posts),
        "vector": vec.astype(np.float32).tolist(),
        "dim": len(vec),
        "groups": {
            "temporal":    [31, temp.tolist()],
            "stylometric": [55, style.tolist()],
            "behavioral":  [8,  behav.tolist()],
            "semantic":    [384, None],   # Don't store 384 floats raw
            "platform":    [6,  plat.tolist()],
        },
        # Human-readable summary
        "summary": {
            "peak_hour":     int(temp[:24].argmax()),
            "weekend_ratio": round(float(behav[1]), 2),
            "burst_score":   round(float(behav[0]), 2),
            "avg_sent_len":  round(float(np.arctanh(max(-0.999, min(0.999, style[0]))) * 25), 1),
            "vocab_richness":round(float(np.arctanh(max(-0.999, min(0.999, style[2]))) * 8), 2),
            "ellipsis_freq": round(float(np.arctanh(max(-0.999, min(0.999, style[5]))) / 80), 4),
            "tech_affinity": round(float(behav[5]), 2),
            "reply_ratio":   round(float(behav[2]), 2),
        }
    }


# ─── Similarity Computation ───────────────────────────────────────────────────

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def feature_breakdown(feat_a: dict, feat_b: dict) -> list[dict]:
    """
    Per-feature similarity breakdown for explainability.
    Returns list of { feature, score, signal } sorted by score desc.
    """
    sa = feat_a.get("summary", {})
    sb = feat_b.get("summary", {})

    checks = [
        ("Active hours alignment",  1 - abs(sa.get("peak_hour",0) - sb.get("peak_hour",0)) / 12),
        ("Weekend posting pattern", 1 - abs(sa.get("weekend_ratio",0) - sb.get("weekend_ratio",0))),
        ("Burst posting style",     1 - abs(sa.get("burst_score",0) - sb.get("burst_score",0))),
        ("Avg sentence length",     1 - abs(sa.get("avg_sent_len",0) - sb.get("avg_sent_len",0)) / 30),
        ("Vocabulary richness",     1 - abs(sa.get("vocab_richness",0) - sb.get("vocab_richness",0)) / 10),
        ("Ellipsis usage",          1 - min(1, abs(sa.get("ellipsis_freq",0) - sb.get("ellipsis_freq",0)) * 500)),
        ("Reply vs post ratio",     1 - abs(sa.get("reply_ratio",0) - sb.get("reply_ratio",0))),
        ("Topic/semantic overlap",  (cosine_similarity(feat_a["vector"], feat_b["vector"]) + 1) / 2),
    ]

    # Stylometric subvector (indices 31-86 in full vector)
    va = np.array(feat_a["vector"])[31:44]
    vb = np.array(feat_b["vector"])[31:44]
    style_sim = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))
    checks.append(("Writing style fingerprint", (style_sim + 1) / 2))

    results = []
    for label, raw_score in checks:
        score = float(max(0.0, min(1.0, raw_score)))
        results.append({
            "feature": label,
            "score": round(score, 3),
            "signal": "strong match" if score >= 0.82 else "partial match" if score >= 0.60 else "mismatch",
        })

    return sorted(results, key=lambda x: -x["score"])
