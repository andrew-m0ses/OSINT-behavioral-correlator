"""
find_pairs_v2ex.py
Mines V2EX (Chinese developer forum) for users who link GitHub.
V2EX has a dedicated github field on every profile.
No credentials needed. Completely public API.

Run: python3 find_pairs_v2ex.py
"""

import requests, re, time, json, os

BASE_V1 = "https://www.v2ex.com/api"
HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)"}
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
IGNORE_GH = {"sponsors","orgs","apps","marketplace","features","topics",
             "collections","trending","explore","login","signup","blog",
             "about","contact","pricing","enterprise","readme","wiki"}

def github_events(gh_user):
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN: h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(f"https://api.github.com/users/{gh_user}/events/public",
                        headers=h, params={"per_page": 5}, timeout=10)
        return len(r.json()) if r.status_code == 200 else 0
    except: return 0

def get_v2ex_user(username):
    try:
        r = requests.get(f"{BASE_V1}/members/show.json",
                        params={"username": username},
                        headers=HEADERS, timeout=10)
        return r.json() if r.ok else {}
    except: return {}

def extract_github(user_data):
    results = []
    # Dedicated github field
    gh = (user_data.get("github") or "").strip()
    if gh:
        gh_user = gh.strip("/").split("/")[-1].split("?")[0].strip()
        if gh_user and len(gh_user) >= 2 and gh_user.lower() not in IGNORE_GH:
            results.append(("github_field", gh_user))

    # Website field
    website = user_data.get("website") or ""
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", website):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in IGNORE_GH: results.append(("website", h))

    # Bio/tagline
    for field in ["bio", "tagline"]:
        text = user_data.get(field) or ""
        for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text):
            h = m.group(1).rstrip(".,;:)")
            if h.lower() not in IGNORE_GH: results.append(("bio", h))

    seen = set(); out = []
    for src, h in results:
        if h.lower() not in seen:
            seen.add(h.lower()); out.append((src, h))
    return out

def get_hot_topic_members():
    """Get usernames from hot/recent V2EX topics and replies."""
    usernames = set()

    # Hot topics
    for endpoint in ["hot", "latest"]:
        try:
            r = requests.get(f"{BASE_V1}/topics/{endpoint}.json",
                           headers=HEADERS, timeout=10)
            if r.ok:
                for topic in r.json():
                    member = topic.get("member") or {}
                    u = member.get("username") or ""
                    if u: usernames.add(u)
            time.sleep(0.5)
        except: pass

    # Topics from popular nodes (tech nodes)
    tech_nodes = ["python", "go", "java", "javascript", "rust", "linux",
                  "programmer", "career", "share", "qna", "github"]
    for node in tech_nodes:
        try:
            r = requests.get(f"{BASE_V1}/nodes/show.json",
                           params={"name": node}, headers=HEADERS, timeout=8)
            if r.ok:
                # Get topics from this node
                r2 = requests.get(f"{BASE_V1}/topics/show.json",
                                params={"node_name": node, "page": 1},
                                headers=HEADERS, timeout=10)
                if r2.ok:
                    for t in r2.json():
                        m = (t.get("member") or {}).get("username", "")
                        if m: usernames.add(m)
            time.sleep(0.4)
        except: pass

    return usernames

def main():
    print(f"GitHub token: {'set' if GITHUB_TOKEN else 'NOT SET'}")

    print("\n=== Fetching V2EX active users ===")
    usernames = get_hot_topic_members()
    print(f"Got {len(usernames)} unique users from topics")

    # Also check top members by number (V2EX assigns sequential IDs)
    # Sample IDs from known active range
    print("\n=== Sampling V2EX members by ID range ===")
    import random
    # V2EX has ~600k+ members, active ones tend to be early registrants
    sample_ids = list(range(1, 5000, 10)) + list(range(5000, 50000, 100))
    random.shuffle(sample_ids)
    for mid in sample_ids[:200]:
        try:
            r = requests.get(f"{BASE_V1}/members/show.json",
                           params={"id": mid}, headers=HEADERS, timeout=8)
            if r.ok:
                u = r.json().get("username", "")
                if u: usernames.add(u)
            time.sleep(0.15)
        except: pass
    print(f"Total unique users: {len(usernames)}")

    print(f"\nScanning {len(usernames)} profiles for GitHub links...\n")

    pairs = []
    checked = 0
    for username in sorted(usernames):
        try:
            user = get_v2ex_user(username)
            if not user: time.sleep(0.2); continue

            gh_handles = extract_github(user)
            for source, gh_handle in gh_handles:
                n = github_events(gh_handle)
                status = f"{n} events" if n > 0 else "inactive"
                print(f"  v2ex:{username:25s} -> github:{gh_handle:25s} {status} [{source}]")
                if n > 0:
                    pairs.append({
                        "v2ex_user": username,
                        "github_user": gh_handle,
                        "github_events": n,
                        "source": source,
                    })
                time.sleep(0.3)

            checked += 1
            if checked % 50 == 0:
                print(f"\n  [{checked}/{len(usernames)} checked, {len(pairs)} pairs]\n")
            time.sleep(0.3)
        except: time.sleep(0.3)
        if len(pairs) >= 100: break

    print(f"\n{'='*60}\nRESULTS: {len(pairs)} V2EX↔GitHub pairs\n{'='*60}")
    for p in pairs:
        print(f"  v2ex:{p['v2ex_user']:25s} <-> github:{p['github_user']:25s} [{p['source']}]")

    with open("discovered_pairs_v2ex.json", "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to discovered_pairs_v2ex.json")

    if not pairs: return

    top = pairs[:40]
    v2ex_args = " ".join(f"v2ex:{p['v2ex_user']}" for p in top)
    gh_args = " ".join(f"github:{p['github_user']}" for p in top)
    print(f"\n# Collect V2EX:")
    print(f"python3 -m osint collect {v2ex_args} --max-posts 500\n")
    print(f"# Collect GitHub:")
    print(f"python3 -m osint collect {gh_args} --max-posts 500\n")
    print("# Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link; import json")
    print("pairs = json.load(open('discovered_pairs_v2ex.json'))")
    print("for p in pairs[:40]:")
    print("    append_link({'from':'v2ex:'+p['v2ex_user'],'to':'github:'+p['github_user'],'source':p['source'],'confidence':'high'})")
    print("    print('Added v2ex:'+p['v2ex_user'],'-> github:'+p['github_user'])")
    print('"')

if __name__ == "__main__":
    main()
