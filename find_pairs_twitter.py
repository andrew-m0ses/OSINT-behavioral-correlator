"""
find_pairs_twitter.py
Mines existing collected accounts for Twitter/X links,
then discovers additional pairs from HN bios.

We already have Twitter usernames from GitHub profiles in identity_links.jsonl.
This script also scans HN bios for Twitter links.

Run: python3 find_pairs_twitter.py
"""

import requests, re, time, json, os
from pathlib import Path

HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)"}
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def decode_html(text):
    return (text
        .replace("&#x2F;", "/").replace("&#x27;", "'")
        .replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"')
    )

def extract_twitter_from_text(text):
    """Extract Twitter/X usernames from text."""
    text = decode_html(re.sub(r"<[^>]+>", " ", text or ""))
    handles = []
    # twitter.com/username or x.com/username
    for m in re.finditer(r"(?:twitter\.com|x\.com)/([A-Za-z0-9_]{1,50})", text):
        h = m.group(1).rstrip(".,;:)")
        if h.lower() not in {"home","search","explore","notifications","messages",
                              "settings","i","intent","share","hashtag"}:
            handles.append(h)
    # @username patterns
    for m in re.finditer(r"(?<![A-Za-z0-9_])@([A-Za-z0-9_]{4,50})(?![A-Za-z0-9_])", text):
        handles.append(m.group(1))
    return list(dict.fromkeys(handles))

def main():
    pairs = []
    seen = set()

    print("=== Phase 1: Load existing identity links ===")
    links_file = Path("data/identity_links.jsonl")
    if links_file.exists():
        with open(links_file) as f:
            for line in f:
                if not line.strip(): continue
                try:
                    link = json.loads(line)
                    src = link.get("from", "")
                    dst = link.get("to", "")
                    if "twitter:" in dst:
                        tw_user = dst.replace("twitter:", "")
                        src_platform, src_user = src.split(":", 1)
                        key = f"{src_user}:{tw_user}"
                        if key not in seen:
                            seen.add(key)
                            pairs.append({
                                "source_platform": src_platform,
                                "source_user": src_user,
                                "twitter_user": tw_user,
                                "source": link.get("source", "existing_link"),
                                "confidence": link.get("confidence", "high"),
                            })
                            print(f"  {src} -> twitter:{tw_user}")
                except: pass
    print(f"Found {len(pairs)} existing Twitter links")

    print("\n=== Phase 2: Scan HN bios for Twitter links ===")
    # Get leaderboard users
    r = requests.get("https://news.ycombinator.com/leaders", headers=HEADERS, timeout=10)
    hn_users = list(dict.fromkeys(re.findall(r"user\?id=([A-Za-z0-9_-]+)", r.text)))
    print(f"Checking {len(hn_users)} HN leaderboard users...")

    for username in hn_users:
        try:
            u = requests.get(f"https://hacker-news.firebaseio.com/v0/user/{username}.json",
                           timeout=8, headers=HEADERS).json()
            if not u: continue
            about = u.get("about", "") or ""
            tw_handles = extract_twitter_from_text(about)
            for tw in tw_handles:
                key = f"{username}:{tw}"
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "source_platform": "hn",
                        "source_user": username,
                        "twitter_user": tw,
                        "source": "hn_bio",
                        "confidence": "high",
                    })
                    print(f"  hn:{username} -> twitter:{tw}")
            time.sleep(0.15)
        except: time.sleep(0.2)

    print("\n=== Phase 3: Scan GitHub bios for Twitter links ===")
    # Check all collected GitHub accounts
    gh_features = Path("data/features/github")
    if gh_features.exists():
        gh_users = [f.stem for f in gh_features.glob("*.json")]
        print(f"Checking {len(gh_users)} collected GitHub accounts...")
        h = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN: h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        for gh_user in gh_users:
            try:
                r = requests.get(f"https://api.github.com/users/{gh_user}",
                               headers=h, timeout=8)
                if r.ok:
                    d = r.json()
                    tw = d.get("twitter_username", "") or ""
                    if tw:
                        key = f"{gh_user}:{tw}"
                        if key not in seen:
                            seen.add(key)
                            pairs.append({
                                "source_platform": "github",
                                "source_user": gh_user,
                                "twitter_user": tw,
                                "source": "github_profile_api",
                                "confidence": "high",
                            })
                            print(f"  github:{gh_user} -> twitter:{tw}")
                    # Also check bio
                    bio = (d.get("bio") or "") + " " + (d.get("blog") or "")
                    for tw2 in extract_twitter_from_text(bio):
                        key2 = f"{gh_user}:{tw2}"
                        if key2 not in seen:
                            seen.add(key2)
                            pairs.append({
                                "source_platform": "github",
                                "source_user": gh_user,
                                "twitter_user": tw2,
                                "source": "github_bio",
                                "confidence": "medium",
                            })
                            print(f"  github:{gh_user} -> twitter:{tw2} (bio)")
                time.sleep(0.2)
            except: time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(pairs)} Twitter pairs found")
    print(f"{'='*60}\n")

    # Group by twitter user to show cross-platform triplets
    by_twitter = {}
    for p in pairs:
        tw = p["twitter_user"]
        by_twitter.setdefault(tw, []).append(p)

    triplets = {tw: ps for tw, ps in by_twitter.items() if len(ps) >= 2}
    print(f"Cross-platform triplets (twitter user linked from 2+ platforms): {len(triplets)}")
    for tw, ps in sorted(triplets.items()):
        srcs = ", ".join(f"{p['source_platform']}:{p['source_user']}" for p in ps)
        print(f"  twitter:{tw} <- {srcs}")

    with open("discovered_pairs_twitter.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved to discovered_pairs_twitter.json")

    # Print collect commands
    tw_users = list(dict.fromkeys(p["twitter_user"] for p in pairs))
    print(f"\n# Collect {len(tw_users)} Twitter accounts:")
    print(f"export APIFY_TOKEN=your_token_here")
    tw_args = " ".join(f"twitter:{u}" for u in tw_users[:50])
    print(f"python3 -m osint collect {tw_args} --max-posts 500\n")

    print("# Register links:")
    print('python3 -c "')
    print("from osint.pipeline import append_link; import json")
    print("pairs = json.load(open('discovered_pairs_twitter.json'))")
    print("for p in pairs:")
    print("    a = p['source_platform']+':'+p['source_user']")
    print("    b = 'twitter:'+p['twitter_user']")
    print("    append_link({'from':a,'to':b,'source':p['source'],'confidence':p['confidence']})")
    print("    print('Added',a,'->',b)")
    print('"')

if __name__ == "__main__":
    main()
