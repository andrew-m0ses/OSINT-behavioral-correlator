"""
find_pairs_so.py
Mines Stack Overflow for users who link GitHub in their profile.
SO has a DEDICATED GitHub field — highest confidence pairs available.
No credentials needed. Free: 10k req/day.

Run: python3 find_pairs_so.py

Then feed output into osint collect + pipeline + train.
"""

import requests, re, time, json, os

BASE = "https://api.stackexchange.com/2.3"
HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)"}
KEY = os.getenv("STACKOVERFLOW_KEY", "")  # optional, raises limit 10k->100k
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

IGNORE_GH = {"sponsors","orgs","apps","marketplace","features","topics",
             "collections","trending","explore","login","signup","blog",
             "about","contact","pricing","enterprise","readme","wiki"}

def so_get(path, params=None):
    p = {"site": "stackoverflow", "pagesize": 100}
    if KEY: p["key"] = KEY
    if params: p.update(params)
    try:
        r = requests.get(f"{BASE}{path}", params=p, timeout=12, headers=HEADERS)
        data = r.json()
        if data.get("backoff"): time.sleep(data["backoff"] + 1)
        return data
    except Exception as e:
        print(f"  SO error: {e}")
        return {}

def github_events(gh_user):
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN: h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(f"https://api.github.com/users/{gh_user}/events/public",
                        headers=h, params={"per_page": 5}, timeout=10)
        return len(r.json()) if r.status_code == 200 else 0
    except: return 0

def extract_github(profile):
    """Extract GitHub username from SO profile fields."""
    results = []

    # 1. Dedicated GitHub field (highest confidence)
    gh = profile.get("github_login", "") or ""
    if gh.strip():
        results.append(("github_field", gh.strip().lstrip("@")))

    # 2. Website URL
    website = profile.get("website_url", "") or ""
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", website):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in IGNORE_GH:
            results.append(("website", h))

    # 3. About me (bio) — strip HTML first
    about = re.sub(r"<[^>]+>", " ", profile.get("about_me", "") or "")
    about = about.replace("&#x2F;","/").replace("&amp;","&")
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", about):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in IGNORE_GH:
            results.append(("bio", h))

    # Deduplicate
    seen = set()
    out = []
    for src, h in results:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append((src, h))
    return out

def get_top_so_users(n_pages=10):
    """Get top SO users by reputation."""
    users = []
    for page in range(1, n_pages + 1):
        data = so_get("/users", {
            "page": page, "sort": "reputation", "order": "desc",
            "filter": "!*236Ig7v1z2MLe(H",  # includes github_login, about_me
        })
        items = data.get("items", [])
        if not items: break
        users.extend(items)
        print(f"  Page {page}: {len(users)} users so far")
        if not data.get("has_more"): break
        time.sleep(0.5)
    return users

def get_so_users_by_tag(tag, n_pages=5):
    """Get active SO users in a specific tag (good for language-specific coverage)."""
    # Get top answers in tag, collect answerers
    user_ids = set()
    for page in range(1, n_pages + 1):
        data = so_get("/answers", {
            "page": page, "sort": "votes", "order": "desc",
            "tagged": tag,
            "filter": "default",
        })
        for item in data.get("items", []):
            if "owner" in item and "user_id" in item["owner"]:
                user_ids.add(item["owner"]["user_id"])
        if not data.get("has_more"): break
        time.sleep(0.3)

    if not user_ids: return []

    # Fetch full profiles with GitHub field
    users = []
    uid_list = list(user_ids)
    for i in range(0, len(uid_list), 100):
        batch = ";".join(str(x) for x in uid_list[i:i+100])
        data = so_get(f"/users/{batch}", {"filter": "!*236Ig7v1z2MLe(H"})
        users.extend(data.get("items", []))
        time.sleep(0.5)
    return users

def main():
    print(f"GitHub token: {'set' if GITHUB_TOKEN else 'NOT SET'}")
    print(f"SO key: {'set' if KEY else 'not set (10k req/day limit)'}")

    all_users = []

    # 1. Top users by reputation (global)
    print("\n=== Fetching top SO users by reputation ===")
    all_users.extend(get_top_so_users(n_pages=10))  # ~1000 users

    # 2. Tag-based: Python, JavaScript, Rust, Go (active open-source devs)
    for tag in ["python", "javascript", "rust", "go", "typescript"]:
        print(f"\n=== Fetching active users in [{tag}] ===")
        tag_users = get_so_users_by_tag(tag, n_pages=3)
        all_users.extend(tag_users)
        print(f"  Got {len(tag_users)} users")

    # Deduplicate by user_id
    seen_ids = set()
    unique_users = []
    for u in all_users:
        uid = u.get("user_id")
        if uid and uid not in seen_ids:
            seen_ids.add(uid)
            unique_users.append(u)

    print(f"\nTotal unique SO users to check: {len(unique_users)}")
    print("Scanning for GitHub links...\n")

    pairs = []
    for i, user in enumerate(unique_users):
        display_name = user.get("display_name", "?")
        reputation = user.get("reputation", 0)
        user_id = user.get("user_id")

        gh_handles = extract_github(user)

        for source, gh_handle in gh_handles:
            n_events = github_events(gh_handle)
            confidence = "high" if source == "github_field" else "medium"
            status = f"{n_events} events" if n_events > 0 else "inactive/404"
            print(f"  so:{display_name:25s} -> github:{gh_handle:25s} rep={reputation:7d} {status} [{source}]")

            if n_events > 0:
                pairs.append({
                    "so_user": display_name,
                    "so_user_id": user_id,
                    "github_user": gh_handle,
                    "so_reputation": reputation,
                    "github_events": n_events,
                    "source": source,
                    "confidence": confidence,
                })
            time.sleep(0.3)

        if i % 100 == 0 and i > 0:
            print(f"\n  [{i}/{len(unique_users)} checked, {len(pairs)} pairs]\n")

        time.sleep(0.1)

        if len(pairs) >= 200:
            print("Reached 200 pairs — stopping.")
            break

    pairs.sort(key=lambda x: -x["so_reputation"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(pairs)} SO↔GitHub pairs")
    print(f"{'='*60}")
    for p in pairs[:50]:
        print(f"  so:{p['so_user']:25s} <-> github:{p['github_user']:25s} rep={p['so_reputation']:7d} [{p['source']}]")

    with open("discovered_pairs_so.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved to discovered_pairs_so.json")

    if not pairs:
        print("No pairs found. Try with a STACKOVERFLOW_KEY for higher rate limits.")
        return

    # Print collect commands
    top = pairs[:60]
    so_args = " ".join(f"stackoverflow:{p['so_user']}" for p in top)
    gh_args  = " ".join(f"github:{p['github_user']}" for p in top)

    print("\n=== COMMANDS TO RUN ===\n")
    print("# 1. Collect SO accounts:")
    print(f"python3 -m osint collect {so_args} --max-posts 500\n")
    print("# 2. Collect GitHub accounts:")
    print(f"python3 -m osint collect {gh_args} --max-posts 500\n")
    print("# 3. Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link")
    print("import json")
    print("with open('discovered_pairs_so.json') as f: pairs = json.load(f)")
    print("for p in pairs[:60]:")
    print("    a = 'stackoverflow:' + p['so_user']")
    print("    b = 'github:' + p['github_user']")
    print("    append_link({'from':a,'to':b,'source':p['source'],'confidence':p['confidence']})")
    print("    print('Added',a,'->',b)")
    print('"')

if __name__ == "__main__":
    main()
