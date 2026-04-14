"""
find_pairs.py  v2
Crawls the HN leaderboard (top users by karma) and checks each profile
for GitHub links. Much higher yield than recent-stories approach.

Run: python3 find_pairs.py
"""
import requests, re, time, json, os, sys

HEADERS = {"User-Agent": "research-bot/1.0"}
TIMEOUT = 10

def hn_user(username):
    try:
        r = requests.get(f"https://hacker-news.firebaseio.com/v0/user/{username}.json",
                         timeout=TIMEOUT, headers=HEADERS)
        return r.json()
    except Exception:
        return None

def github_has_events(gh_username, token=None):
    h = {"Accept": "application/vnd.github+json"}
    if token: h["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"https://api.github.com/users/{gh_username}/events/public",
                         headers=h, params={"per_page": 10}, timeout=TIMEOUT)
        if r.status_code == 200: return len(r.json())
        return 0
    except Exception:
        return 0

def scrape_hn_leaderboard():
    print("Fetching HN leaderboard (top users by karma)...")
    try:
        r = requests.get("https://news.ycombinator.com/leaders",
                         timeout=TIMEOUT, headers=HEADERS)
        usernames = re.findall(r'user\?id=([A-Za-z0-9_-]+)', r.text)
        seen = set(); result = []
        for u in usernames:
            if u not in seen: seen.add(u); result.append(u)
        print(f"  Found {len(result)} top users")
        return result
    except Exception as e:
        print(f"  Failed: {e}"); return []

def get_users_from_recent_activity():
    print("Fetching users from recent HN items...")
    usernames = set()
    for endpoint in ["topstories","newstories","beststories","askstories","showstories"]:
        try:
            r = requests.get(f"https://hacker-news.firebaseio.com/v0/{endpoint}.json",
                             timeout=TIMEOUT, headers=HEADERS)
            for item_id in r.json()[:150]:
                try:
                    item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
                                        timeout=5, headers=HEADERS).json()
                    if item and item.get("by"): usernames.add(item["by"])
                    for kid in (item.get("kids") or [])[:20]:
                        try:
                            c = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{kid}.json",
                                             timeout=4, headers=HEADERS).json()
                            if c and c.get("by"): usernames.add(c["by"])
                        except: pass
                        time.sleep(0.02)
                    time.sleep(0.05)
                except: pass
            print(f"  After {endpoint}: {len(usernames)} users")
        except Exception as e:
            print(f"  {endpoint} failed: {e}")
    print(f"  Found {len(usernames)} users total")
    return list(usernames)

def extract_github_from_bio(about_html):
    text = re.sub(r"<[^>]+>", " ", about_html or "")
    text = text.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
    IGNORE = {"sponsors","orgs","apps","marketplace","features","topics",
              "collections","trending","explore","login","signup","settings",
              "notifications","pulls","issues","codespaces","copilot","blog",
              "about","contact","pricing","enterprise","readme","wiki"}
    found = []
    for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{0,38})", text):
        handle = m.group(1).rstrip(".,;:)")
        if handle.lower() not in IGNORE and len(handle) >= 2:
            found.append(handle)
    return list(dict.fromkeys(found))

def main():
    token = os.getenv("GITHUB_TOKEN")
    print(f"GitHub token: {'set' if token else 'NOT SET — set GITHUB_TOKEN for best results'}")

    usernames = scrape_hn_leaderboard()
    if len(usernames) < 50:
        usernames += get_users_from_recent_activity()

    seen = set(); unique_users = []
    for u in usernames:
        if u not in seen: seen.add(u); unique_users.append(u)
    print(f"\nTotal candidates: {len(unique_users)}")
    print("\n=== Checking profiles for GitHub links ===")

    verified_pairs = []
    checked = 0

    for username in unique_users:
        try:
            user = hn_user(username)
            if not user: time.sleep(0.1); continue
            about = user.get("about","") or ""
            karma = user.get("karma",0)
            gh_handles = extract_github_from_bio(about)

            if gh_handles:
                for gh_handle in gh_handles:
                    n_events = github_has_events(gh_handle, token)
                    mtype = "exact" if gh_handle.lower()==username.lower() else "bio-link"
                    status = f"{n_events} events" if n_events > 0 else "inactive/404"
                    print(f"  hn:{username:20s} -> github:{gh_handle:20s}  karma={karma:6d}  {status}  [{mtype}]")
                    if n_events > 0:
                        verified_pairs.append({"hn_user":username,"github_user":gh_handle,
                                               "hn_karma":karma,"github_events":n_events,"match_type":mtype})
                    time.sleep(0.4)

            checked += 1
            if checked % 25 == 0:
                print(f"\n  --- {checked}/{len(unique_users)} checked, {len(verified_pairs)} pairs ---\n")
            time.sleep(0.15)
        except Exception:
            time.sleep(0.2)
        if len(verified_pairs) >= 200: break

    verified_pairs.sort(key=lambda x: -x["hn_karma"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(verified_pairs)} verified pairs")
    print(f"{'='*60}")
    for p in verified_pairs:
        print(f"  hn:{p['hn_user']:22s} <-> github:{p['github_user']:22s}  karma={p['hn_karma']:6d}  events={p['github_events']:3d}  [{p['match_type']}]")

    with open("discovered_pairs.json","w") as f:
        json.dump(verified_pairs, f, indent=2)
    print(f"\nSaved to discovered_pairs.json")

    if not verified_pairs:
        print("\n0 pairs found. Debug steps:")
        print("  python3 -c \"import requests; r=requests.get('https://news.ycombinator.com/leaders'); print(r.status_code, r.text[:500])\"")
        return

    top = verified_pairs[:40]
    hn_args  = " ".join("hn:"+p["hn_user"]    for p in top)
    gh_args  = " ".join("github:"+p["github_user"] for p in top)

    print("\n=== COMMANDS TO RUN ===\n")
    print("# 1. Collect HN:")
    print(f"python3 -m osint collect {hn_args} --max-posts 500\n")
    print("# 2. Collect GitHub:")
    print(f"python3 -m osint collect {gh_args} --max-posts 500\n")
    print("# 3. Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link")
    print("pairs = [")
    for p in top:
        a = "hn:"+p["hn_user"]; b = "github:"+p["github_user"]
        print(f"    ('{a}', '{b}'),")
    print("]")
    print("for a,b in pairs:")
    print("    append_link({'from':a,'to':b,'source':'hn_bio_verified','confidence':'high'})")
    print("    print('Added',a,'->',b)")
    print('"')

if __name__ == "__main__":
    main()
