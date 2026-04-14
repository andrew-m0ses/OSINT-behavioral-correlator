"""
find_pairs_new.py
Finds NEW HN<->GitHub pairs, skipping ones already in identity_links.jsonl
"""
import requests, re, time, os, json

HEADERS = {'User-Agent': 'research-bot/1.0'}
token = os.getenv('GITHUB_TOKEN')

def decode_html(text):
    return (text
        .replace('&#x2F;', '/')
        .replace('&#x27;', "'")
        .replace('&amp;', '&')
        .replace('&lt;', '<')
        .replace('&gt;', '>')
        .replace('&quot;', '"')
        .replace('&#x3D;', '=')
    )

def check_github(gh_user):
    h = {'Accept': 'application/vnd.github+json'}
    if token: h['Authorization'] = f'Bearer {token}'
    try:
        r = requests.get(f'https://api.github.com/users/{gh_user}/events/public',
                        headers=h, params={'per_page': 5}, timeout=8)
        return len(r.json()) if r.status_code == 200 else 0
    except:
        return 0

# Load already-found pairs to skip them
existing = set()
try:
    with open('data/identity_links.jsonl') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                existing.add(d.get('from','') + '|' + d.get('to',''))
    print(f'Skipping {len(existing)} existing links')
except:
    print('No existing links file found')

# Collect usernames
usernames = set()
r = requests.get('https://news.ycombinator.com/leaders', headers=HEADERS)
for u in re.findall(r'user\?id=([A-Za-z0-9_-]+)', r.text):
    usernames.add(u)
print(f'Leaderboard: {len(usernames)}')

for endpoint in ['topstories','newstories','beststories','askstories','showstories']:
    try:
        r = requests.get(f'https://hacker-news.firebaseio.com/v0/{endpoint}.json',
                        timeout=10, headers=HEADERS)
        for sid in r.json()[:200]:
            try:
                s = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{sid}.json',
                               timeout=5, headers=HEADERS).json()
                if s and s.get('by'): usernames.add(s['by'])
                for kid in (s.get('kids') or [])[:20]:
                    try:
                        c = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{kid}.json',
                                       timeout=4, headers=HEADERS).json()
                        if c and c.get('by'): usernames.add(c['by'])
                    except: pass
                    time.sleep(0.02)
                time.sleep(0.04)
            except: pass
        print(f'After {endpoint}: {len(usernames)}')
    except: pass

IGNORE = {'sponsors','orgs','apps','marketplace','features','topics',
         'collections','trending','explore','login','signup','blog',
         'about','contact','pricing','enterprise','readme','wiki'}

print(f'\nChecking {len(usernames)} users...\n')
pairs = []
checked = 0

for username in sorted(usernames):
    try:
        data = requests.get(f'https://hacker-news.firebaseio.com/v0/user/{username}.json',
                           timeout=8, headers=HEADERS).json()
        if not data:
            time.sleep(0.1)
            continue
        about = decode_html(re.sub(r'<[^>]+>', ' ', data.get('about','') or ''))
        karma = data.get('karma', 0)
        if karma < 200:
            time.sleep(0.05)
            continue

        gh_handles = [h for h in re.findall(r'github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})', about)
                     if h.lower() not in IGNORE]

        for gh in gh_handles:
            key = f'hn:{username}|github:{gh}'
            if key in existing:
                continue
            n = check_github(gh)
            mtype = 'exact' if gh.lower()==username.lower() else 'bio-link'
            print(f'  hn:{username:22s} -> github:{gh:22s} karma={karma:6d} events={n} [{mtype}]')
            if n > 0:
                pairs.append((username, gh, karma, n))
            time.sleep(0.3)

        checked += 1
        if checked % 200 == 0:
            print(f'  [{checked}/{len(usernames)} checked, {len(pairs)} new pairs]')
        time.sleep(0.1)
    except:
        time.sleep(0.2)

    if len(pairs) >= 100:
        break

pairs.sort(key=lambda x: -x[2])
print(f'\n=== {len(pairs)} NEW pairs ===')
for u, g, k, n in pairs:
    print(f'  hn:{u:22s} <-> github:{g:22s} karma={k:6d} events={n}')

with open('discovered_pairs_new.json','w') as f:
    json.dump([{'hn_user':u,'github_user':g,'hn_karma':k,'github_events':n}
               for u,g,k,n in pairs], f, indent=2)
print(f'\nSaved to discovered_pairs_new.json')
