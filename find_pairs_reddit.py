"""
find_pairs_reddit.py
Mines Reddit user profiles for GitHub links in their bio.
No credentials needed — uses Reddit's public JSON API.

Run: python3 find_pairs_reddit.py
"""

import requests, re, time, json, os

HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic research)"}
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
TIMEOUT = 10

IGNORE_GH = {"sponsors","orgs","apps","marketplace","features","topics",
             "collections","trending","explore","login","signup","blog",
             "about","contact","pricing","enterprise","readme","wiki"}

def decode_html(text):
    return (text
        .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", '"').replace("&#x27;", "'").replace("&#x2F;", "/")
    )

def github_events(gh_user):
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN: h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        r = requests.get(f"https://api.github.com/users/{gh_user}/events/public",
                        headers=h, params={"per_page": 5}, timeout=TIMEOUT)
        return len(r.json()) if r.status_code == 200 else 0
    except: return 0

def get_reddit_user(username):
    try:
        r = requests.get(f"https://www.reddit.com/user/{username}/about.json",
                        headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("data", {})
        return {}
    except: return {}

def extract_github_from_bio(bio_text):
    if not bio_text: return []
    text = decode_html(re.sub(r"<[^>]+>", " ", bio_text))
    found = []
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in IGNORE_GH and len(h) >= 2:
            found.append(h)
    return list(dict.fromkeys(found))

def get_subreddit_users(subreddit, n_pages=3):
    """Get active commenters from a subreddit."""
    users = set()
    after = None
    for _ in range(n_pages):
        params = {"limit": 100, "raw_json": 1}
        if after: params["after"] = after
        try:
            r = requests.get(f"https://www.reddit.com/r/{subreddit}/comments.json",
                           headers=HEADERS, params=params, timeout=TIMEOUT)
            if r.status_code != 200: break
            data = r.json().get("data", {})
            for child in data.get("children", []):
                author = child.get("data", {}).get("author", "")
                if author and author not in ("[deleted]", "AutoModerator"):
                    users.add(author)
            after = data.get("after")
            if not after: break
            time.sleep(1.0)
        except Exception as e:
            print(f"  Error: {e}")
            break
    return users

def main():
    print(f"GitHub token: {'set' if GITHUB_TOKEN else 'NOT SET'}")

    # Collect usernames from tech/programming subreddits
    # These have high overlap with GitHub-active developers
    subreddits = [
        # English tech
        "programming", "python", "rust", "golang", "javascript",
        "webdev", "devops", "linux", "opensource", "hackernews",
        # Russian
        "russia", "de_programming", "learnprogramming",
        # General high-activity
        "cscareerquestions", "softwaredevelopment",
    ]

    all_users = set()
    for sub in subreddits:
        print(f"  Fetching r/{sub}...", end=" ")
        users = get_subreddit_users(sub, n_pages=5)
        all_users.update(users)
        print(f"{len(users)} users (total: {len(all_users)})")
        time.sleep(1.0)

    print(f"\nTotal unique Reddit users: {len(all_users)}")
    print("Scanning bios for GitHub links...\n")

    pairs = []
    checked = 0

    for username in sorted(all_users):
        try:
            user = get_reddit_user(username)
            if not user:
                time.sleep(0.3)
                continue

            # Reddit bio is in subreddit.public_description or subreddit.description
            bio = ""
            sr = user.get("subreddit", {})
            if sr:
                bio = sr.get("public_description", "") or sr.get("description", "") or ""

            if not bio:
                time.sleep(0.2)
                checked += 1
                continue

            gh_handles = extract_github_from_bio(bio)

            for gh in gh_handles:
                n = github_events(gh)
                mtype = "exact" if gh.lower() == username.lower() else "bio-link"
                status = f"{n} events" if n > 0 else "inactive/404"
                print(f"  reddit:{username:22s} -> github:{gh:22s} {status} [{mtype}]")

                if n > 0:
                    pairs.append({
                        "reddit_user": username,
                        "github_user": gh,
                        "github_events": n,
                        "match_type": mtype,
                    })
                time.sleep(0.4)

            checked += 1
            if checked % 100 == 0:
                print(f"  [{checked}/{len(all_users)} checked, {len(pairs)} pairs]")
            time.sleep(0.5)  # Reddit rate limit

        except Exception as e:
            time.sleep(0.5)

        if len(pairs) >= 150: break

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(pairs)} Reddit↔GitHub pairs")
    print(f"{'='*60}")
    for p in pairs:
        print(f"  reddit:{p['reddit_user']:22s} <-> github:{p['github_user']:22s} [{p['match_type']}]")

    with open("discovered_pairs_reddit.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved to discovered_pairs_reddit.json")

    if not pairs:
        print("No pairs found.")
        return

    top = pairs[:50]
    reddit_args = " ".join(f"reddit:{p['reddit_user']}" for p in top)
    gh_args = " ".join(f"github:{p['github_user']}" for p in top)

    print("\n=== COMMANDS TO RUN ===\n")
    print("# 1. Collect Reddit accounts:")
    print(f"python3 -m osint collect {reddit_args} --max-posts 500\n")
    print("# 2. Collect GitHub accounts (skip already collected):")
    print(f"python3 -m osint collect {gh_args} --max-posts 500\n")
    print("# 3. Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link")
    print("import json")
    print("with open('discovered_pairs_reddit.json') as f: pairs = json.load(f)")
    print("for p in pairs[:50]:")
    print("    a = 'reddit:' + p['reddit_user']")
    print("    b = 'github:' + p['github_user']")
    print("    append_link({'from':a,'to':b,'source':'reddit_bio','confidence':'high'})")
    print("    print('Added',a,'->',b)")
    print('"')

if __name__ == "__main__":
    main()
