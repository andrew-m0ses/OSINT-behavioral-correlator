"""
find_pairs_habr.py
Mines Habr.com (Russian developer platform) for users who link GitHub.
Habr has a dedicated contacts/github field — highest confidence pairs.
No credentials needed. Free public API.

Run: python3 find_pairs_habr.py
"""

import requests, re, time, json, os

BASE = "https://habr.com/api/v2"
HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)", "Accept": "application/json"}
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

def get_habr_user(username):
    try:
        r = requests.get(f"{BASE}/users/{username}", headers=HEADERS, timeout=10)
        return r.json() if r.ok else {}
    except: return {}

def extract_github(user_data):
    results = []
    # Check contacts field (dedicated GitHub field)
    contacts = user_data.get("contacts") or {}
    if isinstance(contacts, list):
        for c in contacts:
            if isinstance(c, dict):
                title = (c.get("title") or "").lower()
                url = c.get("url") or c.get("value") or ""
                if "github" in title or "github.com" in url:
                    m = re.search(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", url)
                    if m:
                        results.append(("contacts_field", m.group(1).rstrip(".,;:)")))
    elif isinstance(contacts, dict):
        gh = contacts.get("github") or contacts.get("Github") or ""
        if gh:
            gh_user = gh.strip("/").split("/")[-1]
            if gh_user: results.append(("contacts_field", gh_user))

    # Check socialLinks
    social = user_data.get("socialLinks") or []
    for link in social:
        url = link.get("url") or link.get("href") or ""
        if "github.com" in url:
            m = re.search(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", url)
            if m: results.append(("social_link", m.group(1).rstrip(".,;:)")))

    # Check bio
    bio = re.sub(r"<[^>]+>", " ", user_data.get("aboutHtml") or user_data.get("about") or "")
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", bio):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in IGNORE_GH: results.append(("bio", h))

    # Deduplicate
    seen = set(); out = []
    for src, h in results:
        if h.lower() not in seen and h.lower() not in IGNORE_GH:
            seen.add(h.lower()); out.append((src, h))
    return out

def get_top_habr_users(n_pages=20):
    """Get top Habr users by rating."""
    users = []
    for page in range(1, n_pages + 1):
        try:
            r = requests.get(f"{BASE}/users", headers=HEADERS, timeout=12,
                           params={"page": page, "perPage": 50, "sort": "rating", "order": "desc"})
            if not r.ok: break
            data = r.json()
            items = data.get("userRefs") or data.get("users") or []
            if not items: break
            users.extend(items)
            print(f"  Page {page}: {len(users)} users")
            if not data.get("pagesCount") or page >= data.get("pagesCount", 1): break
            time.sleep(0.4)
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
    return users

def get_habr_users_from_articles(n_pages=10):
    """Get active Habr authors from recent articles."""
    users = set()
    for page in range(1, n_pages + 1):
        try:
            r = requests.get(f"{BASE}/articles", headers=HEADERS, timeout=12,
                           params={"page": page, "perPage": 20, "sort": "date"})
            if not r.ok: break
            data = r.json()
            items = data.get("articleRefs") or []
            for item in items:
                author = item.get("author") or {}
                alias = author.get("alias") or author.get("login") or ""
                if alias: users.add(alias)
            time.sleep(0.3)
        except: break
    return users

def main():
    print(f"GitHub token: {'set' if GITHUB_TOKEN else 'NOT SET'}")

    # Get candidates from multiple sources
    all_usernames = set()

    print("\n=== Fetching top Habr users by rating ===")
    top_users = get_top_habr_users(n_pages=20)
    for u in top_users:
        alias = u.get("alias") or u.get("login") or ""
        if alias: all_usernames.add(alias)
    print(f"Got {len(all_usernames)} users from ratings")

    print("\n=== Fetching active article authors ===")
    article_authors = get_habr_users_from_articles(n_pages=15)
    all_usernames.update(article_authors)
    print(f"Total unique users: {len(all_usernames)}")

    print(f"\nScanning {len(all_usernames)} profiles for GitHub links...\n")

    pairs = []
    checked = 0
    for username in sorted(all_usernames):
        try:
            user = get_habr_user(username)
            if not user: time.sleep(0.2); continue

            gh_handles = extract_github(user)
            score = user.get("score", 0) or user.get("rating", 0) or 0

            for source, gh_handle in gh_handles:
                n = github_events(gh_handle)
                status = f"{n} events" if n > 0 else "inactive"
                print(f"  habr:{username:25s} -> github:{gh_handle:25s} score={score:6} {status} [{source}]")
                if n > 0:
                    pairs.append({
                        "habr_user": username,
                        "github_user": gh_handle,
                        "habr_score": score,
                        "github_events": n,
                        "source": source,
                    })
                time.sleep(0.3)

            checked += 1
            if checked % 50 == 0:
                print(f"\n  [{checked}/{len(all_usernames)} checked, {len(pairs)} pairs]\n")
            time.sleep(0.2)
        except Exception as e:
            time.sleep(0.3)
        if len(pairs) >= 150: break

    pairs.sort(key=lambda x: -x.get("habr_score", 0))
    print(f"\n{'='*60}\nRESULTS: {len(pairs)} Habr↔GitHub pairs\n{'='*60}")
    for p in pairs[:30]:
        print(f"  habr:{p['habr_user']:25s} <-> github:{p['github_user']:25s} [{p['source']}]")

    with open("discovered_pairs_habr.json", "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to discovered_pairs_habr.json")

    if not pairs: return

    top = pairs[:50]
    habr_args = " ".join(f"habr:{p['habr_user']}" for p in top)
    gh_args = " ".join(f"github:{p['github_user']}" for p in top)
    print(f"\n# Collect Habr:")
    print(f"python3 -m osint collect {habr_args} --max-posts 500\n")
    print(f"# Collect GitHub:")
    print(f"python3 -m osint collect {gh_args} --max-posts 500\n")
    print("# Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link; import json")
    print("pairs = json.load(open('discovered_pairs_habr.json'))")
    print("for p in pairs[:50]:")
    print("    append_link({'from':'habr:'+p['habr_user'],'to':'github:'+p['github_user'],'source':p['source'],'confidence':'high'})")
    print("    print('Added habr:'+p['habr_user'],'-> github:'+p['github_user'])")
    print('"')

if __name__ == "__main__":
    main()
