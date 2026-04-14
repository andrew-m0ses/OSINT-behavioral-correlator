"""
osint/collectors.py
Live data collection from public APIs.

Free/no-auth: Hacker News (Algolia), GitHub public events
Optional:     Reddit (OAuth), YouTube (API key), Twitter ($100/mo)

Each collector returns a normalized AccountData dict:
{
    platform: str,
    username: str,
    collected_at: str,
    posts: [{ id, text, timestamp, type, metadata }]
    post_count: int,
    profile: { bio, website, ... }
}
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional
import requests
from rich.console import Console

console = Console()

MIN_TEXT_LENGTH = 15   # Discard posts shorter than this
REQUEST_TIMEOUT = 12


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_get(url, params=None, headers=None, retries=3, delay=1.0):
    """GET with retries and polite rate limiting."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                console.print(f"[yellow]Rate limited — waiting {wait}s[/yellow]")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    return None


# ─── Hacker News ──────────────────────────────────────────────────────────────

class HNCollector:
    """
    Hacker News via Algolia API — completely free, no credentials needed.
    Returns full comment + story history per user.
    Docs: https://hn.algolia.com/api
    """
    ALGOLIA = "https://hn.algolia.com/api/v1"
    FIREBASE = "https://hacker-news.firebaseio.com/v0"

    def collect(self, username: str, max_posts: int = 500) -> dict:
        posts = []
        page = 0
        console.print(f"  [orange3]HN[/orange3] fetching [bold]{username}[/bold]...", end=" ")

        while len(posts) < max_posts:
            r = _safe_get(
                f"{self.ALGOLIA}/search_by_date",
                params={"tags": f"author_{username}", "hitsPerPage": 100, "page": page}
            )
            if not r:
                break
            hits = r.json().get("hits", [])
            if not hits:
                break

            for h in hits:
                raw = (h.get("comment_text") or h.get("story_text") or h.get("title") or "").strip()
                # Strip HTML tags
                text = re.sub(r"<[^>]+>", " ", raw).strip()
                if len(text) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": h.get("objectID"),
                        "text": text,
                        "timestamp": h.get("created_at"),
                        "type": "comment" if h.get("comment_text") else "story",
                        "metadata": {
                            "points": h.get("points", 0),
                            "parent_id": h.get("parent_id"),
                            "story_id": h.get("story_id"),
                        }
                    })
            page += 1
            if len(hits) < 100:
                break
            time.sleep(0.15)

        # Fetch profile bio
        profile = {}
        try:
            pr = _safe_get(f"{self.FIREBASE}/user/{username}.json")
            if pr:
                pdata = pr.json() or {}
                about_raw = pdata.get("about", "") or ""
                profile = {
                    "bio": re.sub(r"<[^>]+>", " ", about_raw).strip(),
                    "karma": pdata.get("karma", 0),
                    "created": pdata.get("created"),
                }
        except Exception:
            pass

        console.print(f"[green]{len(posts)} posts[/green]")
        return {
            "platform": "hn",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def discover_active_users(self, n: int = 200) -> list[str]:
        """Pull usernames from recent HN activity stream."""
        users = set()
        page = 0
        while len(users) < n:
            r = _safe_get(f"{self.ALGOLIA}/search_by_date",
                         params={"tags": "comment", "hitsPerPage": 100, "page": page})
            if not r:
                break
            for h in r.json().get("hits", []):
                if h.get("author"):
                    users.add(h["author"])
            page += 1
            if page > 10:
                break
            time.sleep(0.2)
        return list(users)[:n]

    def find_cross_platform_links(self, username: str) -> list[dict]:
        """Extract GitHub/Twitter links from HN profile bio."""
        links = []
        try:
            r = _safe_get(f"{self.FIREBASE}/user/{username}.json")
            if not r:
                return links
            about = (r.json() or {}).get("about", "") or ""
            about = re.sub(r"<[^>]+>", " ", about)
            for gh in re.findall(r"github\.com/([A-Za-z0-9_-]+)", about):
                links.append({"from": f"hn:{username}", "to": f"github:{gh}", "source": "hn_bio"})
            for tw in re.findall(r"twitter\.com/([A-Za-z0-9_]+)", about):
                links.append({"from": f"hn:{username}", "to": f"twitter:{tw}", "source": "hn_bio"})
        except Exception:
            pass
        return links


# ─── GitHub ───────────────────────────────────────────────────────────────────

class GitHubCollector:
    """
    GitHub public events + issue comments + commit messages.
    Free: 60 req/hr unauthenticated | 5,000 req/hr with GITHUB_TOKEN.
    Rich source of long-form technical writing (issues, PRs, commit msgs).
    """
    BASE = "https://api.github.com"

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
            self._authed = True
        else:
            self._authed = False

    def _get(self, path: str, params: dict = None):
        url = f"{self.BASE}{path}"
        r = _safe_get(url, params=params, headers=self.headers)
        if r and r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(5, reset - int(time.time()))
            console.print(f"[yellow]GitHub rate limit — waiting {wait}s[/yellow]")
            time.sleep(wait)
            r = _safe_get(url, params=params, headers=self.headers)
        return r

    def collect(self, username: str, max_posts: int = 300) -> dict:
        posts = []
        console.print(f"  [green]GitHub[/green] fetching [bold]{username}[/bold]...", end=" ")

        # 1. Issue comments via search API — goes back YEARS, not just 90 days
        #    This is the richest source of long-form writing on GitHub
        try:
            page = 1
            while len(posts) < max_posts:
                r = self._get("/search/issues", {
                    "q": f"commenter:{username} type:issue",
                    "sort": "updated", "order": "desc",
                    "per_page": 100, "page": page,
                })
                if not r: break
                data = r.json()
                items = data.get("items", [])
                if not items: break
                for item in items:
                    # Fetch the actual comments by this user on this issue
                    issue_comments = self._get_user_comments_on_issue(
                        username, item.get("comments_url", "")
                    )
                    posts.extend(issue_comments)
                if not data.get("incomplete_results") and len(items) < 100:
                    break
                page += 1
                if page > 5: break
                time.sleep(0.5)
        except Exception as e:
            pass

        # 2. PR review comments via search API
        try:
            page = 1
            while len(posts) < max_posts:
                r = self._get("/search/issues", {
                    "q": f"reviewed-by:{username} type:pr",
                    "sort": "updated", "order": "desc",
                    "per_page": 50, "page": page,
                })
                if not r: break
                data = r.json()
                items = data.get("items", [])
                if not items: break
                for item in items:
                    pr_url = item.get("pull_request", {}).get("url", "")
                    if pr_url:
                        reviews = self._get_user_pr_reviews(username, pr_url)
                        posts.extend(reviews)
                if len(items) < 50: break
                page += 1
                if page > 3: break
                time.sleep(0.5)
        except Exception:
            pass

        # 3. Fall back to public events if still thin
        if len(posts) < 30:
            for page in range(1, 11):
                r = self._get(f"/users/{username}/events/public",
                             {"per_page": 100, "page": page})
                if not r: break
                events = r.json()
                if not isinstance(events, list) or not events: break
                for ev in events:
                    text = self._extract_event_text(ev)
                    if text and len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": ev.get("id"),
                            "text": text,
                            "timestamp": ev.get("created_at"),
                            "type": ev.get("type", "unknown"),
                            "metadata": {"repo": ev.get("repo", {}).get("name")},
                        })
                if len(events) < 100: break
                if len(posts) >= max_posts: break
                time.sleep(0.3)

        # 4. Profile
        profile = {}
        try:
            pr = self._get(f"/users/{username}")
            if pr:
                pd = pr.json()
                profile = {
                    "bio": pd.get("bio") or "",
                    "twitter": pd.get("twitter_username"),
                    "blog": pd.get("blog") or "",
                    "company": pd.get("company") or "",
                    "location": pd.get("location") or "",
                    "public_repos": pd.get("public_repos", 0),
                    "followers": pd.get("followers", 0),
                }
        except Exception:
            pass

        # Deduplicate by id
        seen = set()
        deduped = []
        for p in posts:
            pid = str(p.get("id","")) + p.get("text","")[:50]
            if pid not in seen:
                seen.add(pid)
                deduped.append(p)

        console.print(f"[green]{min(len(deduped), max_posts)} posts[/green]")
        return {
            "platform": "github",
            "username": username,
            "collected_at": _now_iso(),
            "posts": deduped[:max_posts],
            "post_count": min(len(deduped), max_posts),
            "profile": profile,
        }

    def _get_user_comments_on_issue(self, username: str, comments_url: str) -> list:
        """Fetch comments on an issue/PR, filter to only this user's comments."""
        if not comments_url: return []
        posts = []
        try:
            # Strip base URL since _get prepends it
            path = comments_url.replace("https://api.github.com", "")
            r = self._get(path, {"per_page": 100})
            if not r: return []
            for comment in r.json():
                if not isinstance(comment, dict): continue
                if comment.get("user", {}).get("login", "").lower() != username.lower():
                    continue
                body = comment.get("body", "").strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": comment.get("id"),
                        "text": body[:3000],
                        "timestamp": comment.get("created_at"),
                        "type": "issue_comment",
                        "metadata": {"url": comment.get("html_url")},
                    })
            time.sleep(0.2)
        except Exception:
            pass
        return posts

    def _get_user_pr_reviews(self, username: str, pr_url: str) -> list:
        """Fetch PR review comments by this user."""
        if not pr_url: return []
        posts = []
        try:
            path = pr_url.replace("https://api.github.com", "") + "/reviews"
            r = self._get(path, {"per_page": 100})
            if not r: return []
            for review in r.json():
                if not isinstance(review, dict): continue
                if review.get("user", {}).get("login", "").lower() != username.lower():
                    continue
                body = review.get("body", "").strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": review.get("id"),
                        "text": body[:3000],
                        "timestamp": review.get("submitted_at"),
                        "type": "pr_review",
                        "metadata": {"state": review.get("state")},
                    })
            time.sleep(0.2)
        except Exception:
            pass
        return posts


    def _extract_event_text(self, event: dict) -> Optional[str]:
        t = event.get("type", "")
        p = event.get("payload", {})
        if t == "IssueCommentEvent":
            return p.get("comment", {}).get("body")
        if t == "IssuesEvent":
            iss = p.get("issue", {})
            return f"{iss.get('title','')} {iss.get('body','')}".strip()
        if t == "PullRequestReviewCommentEvent":
            return p.get("comment", {}).get("body")
        if t == "PullRequestEvent":
            pr = p.get("pull_request", {})
            return f"{pr.get('title','')} {pr.get('body','')}".strip()
        if t == "PushEvent":
            msgs = [c.get("message","") for c in p.get("commits", [])]
            return " ".join(msgs).strip() or None
        if t == "CreateEvent":
            return p.get("description") or None
        return None

    def find_cross_platform_links(self, username: str) -> list[dict]:
        """GitHub API exposes twitter_username directly."""
        links = []
        try:
            r = self._get(f"/users/{username}")
            if not r:
                return links
            pd = r.json()
            if pd.get("twitter_username"):
                links.append({
                    "from": f"github:{username}",
                    "to": f"twitter:{pd['twitter_username']}",
                    "source": "github_profile_api",
                    "confidence": "high",
                })
            bio = (pd.get("bio") or "") + " " + (pd.get("blog") or "")
            for hn in re.findall(r"news\.ycombinator\.com/user\?id=([A-Za-z0-9_-]+)", bio):
                links.append({
                    "from": f"github:{username}",
                    "to": f"hn:{hn}",
                    "source": "github_bio",
                    "confidence": "high",
                })
        except Exception:
            pass
        return links

    def discover_active_users(self, n: int = 100) -> list[str]:
        """Pull users from public GitHub event stream."""
        r = self._get("/events", {"per_page": 100})
        if not r:
            return []
        events = r.json()
        users = list({e["actor"]["login"] for e in events if "actor" in e})
        return users[:n]


# ─── Reddit ───────────────────────────────────────────────────────────────────

class RedditCollector:
    """
    Reddit via JSON API (no auth needed for public data, limited to ~25/page).
    For full access set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET.
    """

    def collect(self, username: str, max_posts: int = 200) -> dict:
        posts = []
        console.print(f"  [red]Reddit[/red] fetching [bold]{username}[/bold]...", end=" ")
        after = None
        headers = {"User-Agent": "OSINTTool/1.0 (research)"}

        for _ in range(10):
            params = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            try:
                r = _safe_get(
                    f"https://www.reddit.com/user/{username}/comments.json",
                    params=params, headers=headers
                )
                if not r:
                    break
                data = r.json().get("data", {})
                children = data.get("children", [])
                if not children:
                    break
                for child in children:
                    d = child.get("data", {})
                    text = d.get("body", "").strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": d.get("id"),
                            "text": text,
                            "timestamp": datetime.fromtimestamp(
                                d.get("created_utc", 0), tz=timezone.utc
                            ).isoformat(),
                            "type": "comment",
                            "metadata": {
                                "subreddit": d.get("subreddit"),
                                "score": d.get("score", 0),
                            }
                        })
                after = data.get("after")
                if not after or len(posts) >= max_posts:
                    break
                time.sleep(1.0)  # Reddit is strict about rate limits
            except Exception as e:
                console.print(f"[yellow] Reddit error: {e}[/yellow]")
                break

        console.print(f"[green]{len(posts[:max_posts])} posts[/green]")
        return {
            "platform": "reddit",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": {},
        }


# ─── YouTube ──────────────────────────────────────────────────────────────────

class YouTubeCollector:
    """
    YouTube comment collection via Data API v3.
    Requires YOUTUBE_API_KEY. Free quota: 10,000 units/day.
    """

    def __init__(self):
        self.key = os.getenv("YOUTUBE_API_KEY")
        self.BASE = "https://www.googleapis.com/youtube/v3"

    def collect_by_channel(self, channel_id: str, max_posts: int = 200) -> dict:
        if not self.key:
            console.print("[yellow]YouTube: set YOUTUBE_API_KEY env var[/yellow]")
            return {"platform": "youtube", "username": channel_id,
                    "collected_at": _now_iso(), "posts": [], "post_count": 0, "profile": {}}

        posts = []
        page_token = None
        console.print(f"  [red]YouTube[/red] fetching [bold]{channel_id}[/bold]...", end=" ")

        while len(posts) < max_posts:
            params = {
                "part": "snippet", "allThreadsRelatedToChannelId": channel_id,
                "maxResults": 100, "key": self.key,
            }
            if page_token:
                params["pageToken"] = page_token
            r = _safe_get(f"{self.BASE}/commentThreads", params=params)
            if not r:
                break
            data = r.json()
            for item in data.get("items", []):
                sn = item["snippet"]["topLevelComment"]["snippet"]
                text = sn.get("textOriginal", "").strip()
                if len(text) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": item["id"],
                        "text": text,
                        "timestamp": sn.get("publishedAt"),
                        "type": "comment",
                        "metadata": {"likes": sn.get("likeCount", 0)},
                    })
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.2)

        console.print(f"[green]{len(posts[:max_posts])} posts[/green]")
        return {
            "platform": "youtube",
            "username": channel_id,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": {},
        }


# ─── Collector Registry ───────────────────────────────────────────────────────

def get_collector(platform: str):
    return {
        "hn":      HNCollector,
        "stackoverflow": StackOverflowCollector,
        "github":  GitHubCollector,
        "reddit":  RedditCollector,
        "youtube": YouTubeCollector,
        "habr":    HabrCollector,
        "v2ex":    V2EXCollector,
        "twitter": TwitterCollector,
    }.get(platform)


def collect_account(platform: str, username: str, max_posts: int = 300) -> Optional[dict]:
    """
    Top-level entry point. Returns normalized account data or None.
    """
    cls = get_collector(platform)
    if cls is None:
        console.print(f"[red]No collector for platform: {platform}[/red]")
        return None
    try:
        collector = cls()
        if platform == "youtube":
            return collector.collect_by_channel(username, max_posts)
        return collector.collect(username, max_posts)
    except Exception as e:
        console.print(f"[red]Collection failed for {platform}:{username} — {e}[/red]")
        return None


def mine_identity_links(platform: str, username: str) -> list[dict]:
    """Mine cross-platform self-disclosures from profile bios."""
    cls = get_collector(platform)
    if cls is None:
        return []
    try:
        collector = cls()
        if hasattr(collector, "find_cross_platform_links"):
            return collector.find_cross_platform_links(username)
    except Exception:
        pass
    return []


# ─── Stack Overflow ───────────────────────────────────────────────────────────

class StackOverflowCollector:
    """
    Stack Overflow via public API v2.3. No auth needed (10k req/day).
    With API key (free): 100k req/day.
    Supports all Stack Exchange sites: stackoverflow.com, ru.stackoverflow.com, etc.
    """

    BASE = "https://api.stackexchange.com/2.3"

    def __init__(self, site: str = "stackoverflow"):
        self.site = site
        self.key = os.getenv("STACKOVERFLOW_KEY")  # optional free key

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        p = {"site": self.site, "filter": "withbody", "pagesize": 100}
        if self.key:
            p["key"] = self.key
        if params:
            p.update(params)
        try:
            r = _safe_get(f"{self.BASE}{path}", params=p)
            if not r:
                return None
            data = r.json()
            # Respect backoff
            if data.get("backoff"):
                time.sleep(data["backoff"])
            return data
        except Exception as e:
            console.print(f"[yellow]SO error: {e}[/yellow]")
            return None

    def get_user_id(self, username: str) -> Optional[int]:
        """Resolve display name or numeric ID to user ID."""
        if username.isdigit():
            return int(username)
        data = self._get("/users", {"inname": username, "pagesize": 5, "filter": "default"})
        if not data or not data.get("items"):
            return None
        # Find exact match first, then closest
        for item in data["items"]:
            if item["display_name"].lower() == username.lower():
                return item["user_id"]
        return data["items"][0]["user_id"]

    def get_profile(self, user_id: int) -> dict:
        """Get profile including GitHub link if set."""
        data = self._get(f"/users/{user_id}", {"filter": "!*236Ig7v1z2MLe(H"})
        if not data or not data.get("items"):
            return {}
        item = data["items"][0]
        return {
            "display_name": item.get("display_name", ""),
            "website": item.get("website_url", ""),
            "location": item.get("location", ""),
            "reputation": item.get("reputation", 0),
            "github": item.get("github", ""),  # SO has a dedicated GitHub field
            "about": re.sub(r"<[^>]+>", " ", item.get("about_me", "") or ""),
        }

    def collect(self, username: str, max_posts: int = 300) -> dict:
        console.print(f"  [blue]StackOverflow[/blue] fetching [bold]{username}[/bold]...", end=" ")

        user_id = self.get_user_id(username)
        if not user_id:
            console.print(f"[red]User not found[/red]")
            return {"platform": "stackoverflow", "username": username,
                    "collected_at": _now_iso(), "posts": [], "post_count": 0, "profile": {}}

        profile = self.get_profile(user_id)
        posts = []
        page = 1

        while len(posts) < max_posts:
            data = self._get(f"/users/{user_id}/answers",
                           {"page": page, "sort": "creation", "order": "desc"})
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                body = re.sub(r"<[^>]+>", " ", item.get("body", "") or "")
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": str(item.get("answer_id")),
                        "text": body[:2000],
                        "timestamp": datetime.fromtimestamp(
                            item.get("creation_date", 0), tz=timezone.utc
                        ).isoformat(),
                        "type": "answer",
                        "metadata": {
                            "score": item.get("score", 0),
                            "is_accepted": item.get("is_accepted", False),
                            "question_id": item.get("question_id"),
                        }
                    })
            if not data.get("has_more") or len(posts) >= max_posts:
                break
            page += 1
            time.sleep(0.5)

        # Also get comments
        page = 1
        while len(posts) < max_posts:
            data = self._get(f"/users/{user_id}/comments",
                           {"page": page, "sort": "creation", "order": "desc"})
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                body = re.sub(r"<[^>]+>", " ", item.get("body", "") or "")
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": str(item.get("comment_id")),
                        "text": body[:2000],
                        "timestamp": datetime.fromtimestamp(
                            item.get("creation_date", 0), tz=timezone.utc
                        ).isoformat(),
                        "type": "comment",
                        "metadata": {"score": item.get("score", 0)},
                    })
            if not data.get("has_more") or len(posts) >= max_posts:
                break
            page += 1
            time.sleep(0.5)

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "stackoverflow",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links in SO profile."""
        user_id = self.get_user_id(username)
        if not user_id:
            return []
        profile = self.get_profile(user_id)
        links = []
        # Check dedicated GitHub field
        gh = profile.get("github", "")
        if gh:
            gh_user = gh.strip("/").split("/")[-1].split("?")[0]
            if gh_user:
                links.append({
                    "from": f"stackoverflow:{username}",
                    "to": f"github:{gh_user}",
                    "source": "stackoverflow_github_field",
                    "confidence": "high",
                })
        # Also check website and about
        for text in [profile.get("website",""), profile.get("about","")]:
            for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text):
                gh_user = m.group(1).rstrip(".,;:)")
                if gh_user.lower() not in {"sponsors","orgs","apps","marketplace"}:
                    links.append({
                        "from": f"stackoverflow:{username}",
                        "to": f"github:{gh_user}",
                        "source": "stackoverflow_bio",
                        "confidence": "medium",
                    })
        return links


# ─── Habr (habr.com) ──────────────────────────────────────────────────────────

class HabrCollector:
    """
    Habr.com — Russian developer platform (like HN + Medium combined).
    Completely public API, no auth needed.
    Profiles often link GitHub directly.
    Language: Russian (primarily), some English articles.
    """
    BASE = "https://habr.com/api/v2"
    HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)", "Accept": "application/json"}

    def collect(self, username: str, max_posts: int = 300) -> dict:
        posts = []
        console.print(f"  [red]Habr[/red] fetching [bold]{username}[/bold]...", end=" ")

        profile = self._get_profile(username)

        # Fetch articles (long-form posts)
        page = 1
        while len(posts) < max_posts:
            try:
                r = requests.get(
                    f"{self.BASE}/users/{username}/articles",
                    params={"page": page, "perPage": 20, "sort": "published"},
                    headers=self.HEADERS, timeout=12
                )
                if r.status_code == 404:
                    break
                if not r.ok:
                    break
                data = r.json()
                items = data.get("articleRefs") or data.get("articles") or []
                if not items:
                    break
                for item in items:
                    # Get full article text
                    article_id = item.get("id") or item.get("alias")
                    if article_id:
                        text = self._get_article_text(article_id)
                        if text and len(text) >= MIN_TEXT_LENGTH:
                            posts.append({
                                "id": str(article_id),
                                "text": text[:3000],
                                "timestamp": item.get("timePublished") or item.get("publishedAt"),
                                "type": "article",
                                "metadata": {
                                    "title": item.get("titleHtml") or item.get("title", ""),
                                    "score": item.get("statistics", {}).get("score", 0),
                                }
                            })
                        time.sleep(0.3)
                page += 1
                if len(items) < 20:
                    break
                time.sleep(0.5)
            except Exception as e:
                break

        # Fetch comments
        page = 1
        while len(posts) < max_posts:
            try:
                r = requests.get(
                    f"{self.BASE}/users/{username}/comments",
                    params={"page": page, "perPage": 50},
                    headers=self.HEADERS, timeout=12
                )
                if not r.ok:
                    break
                data = r.json()
                items = data.get("commentRefs") or data.get("comments") or []
                if not items:
                    break
                for item in items:
                    text = re.sub(r"<[^>]+>", " ", item.get("textHtml") or item.get("text") or "")
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": str(item.get("id")),
                            "text": text[:2000],
                            "timestamp": item.get("timePublished") or item.get("publishedAt"),
                            "type": "comment",
                            "metadata": {"score": item.get("score", 0)},
                        })
                page += 1
                if len(items) < 50:
                    break
                time.sleep(0.5)
            except Exception:
                break

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "habr",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def _get_profile(self, username: str) -> dict:
        try:
            r = requests.get(f"{self.BASE}/users/{username}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return {}
            d = r.json()
            return {
                "alias": d.get("alias", ""),
                "fullname": d.get("fullname", ""),
                "bio": re.sub(r"<[^>]+>", " ", d.get("aboutHtml") or d.get("about") or ""),
                "website": d.get("contacts", {}).get("url", "") if isinstance(d.get("contacts"), dict) else "",
                "github": "",  # extracted separately in find_cross_platform_links
                "score": d.get("score", 0),
                "rating": d.get("rating", 0),
            }
        except Exception:
            return {}

    def _get_article_text(self, article_id) -> str:
        try:
            r = requests.get(f"{self.BASE}/articles/{article_id}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return ""
            d = r.json()
            html = d.get("textHtml") or d.get("text") or ""
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return ""

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links in Habr profile."""
        links = []
        try:
            r = requests.get(f"{self.BASE}/users/{username}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return []
            d = r.json()

            # Check social links/contacts
            contacts = d.get("contacts") or {}
            if isinstance(contacts, dict):
                github = contacts.get("github") or contacts.get("Github") or ""
                if github:
                    gh_user = github.strip("/").split("/")[-1]
                    if gh_user:
                        links.append({
                            "from": f"habr:{username}",
                            "to": f"github:{gh_user}",
                            "source": "habr_contacts_field",
                            "confidence": "high",
                        })

            # Check bio text
            bio = re.sub(r"<[^>]+>", " ", d.get("aboutHtml") or d.get("about") or "")
            for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", bio):
                gh_user = m.group(1).rstrip(".,;:)")
                links.append({
                    "from": f"habr:{username}",
                    "to": f"github:{gh_user}",
                    "source": "habr_bio",
                    "confidence": "medium",
                })
        except Exception:
            pass
        return links


# ─── V2EX (v2ex.com) ─────────────────────────────────────────────────────────

class V2EXCollector:
    """
    V2EX — Chinese developer forum, like HN for Chinese tech community.
    Completely public JSON API, no auth needed.
    Language: Chinese (primarily), some English.
    Profiles sometimes link GitHub.
    """
    BASE = "https://www.v2ex.com/api/v2"
    BASE_V1 = "https://www.v2ex.com/api"
    HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)"}

    def collect(self, username: str, max_posts: int = 300) -> dict:
        posts = []
        console.print(f"  [yellow]V2EX[/yellow] fetching [bold]{username}[/bold]...", end=" ")

        profile = self._get_profile(username)

        # V2EX API endpoints for replies/topics are dead — scrape HTML instead
        web_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        # Scrape replies pages
        for page in range(1, 11):
            if len(posts) >= max_posts:
                break
            try:
                r = requests.get(
                    f"https://www.v2ex.com/member/{username}/replies",
                    params={"p": page}, headers=web_headers, timeout=12
                )
                if not r.ok:
                    break
                # Extract reply content from HTML
                # Replies are in <div class="reply_content"> tags
                reply_texts = re.findall(
                    r'<div[^>]+class=["\']reply_content["\'][^>]*>(.*?)</div>',
                    r.text, re.DOTALL
                )
                if not reply_texts:
                    break
                for raw in reply_texts:
                    text = re.sub(r"<[^>]+>", " ", raw)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": f"{username}_reply_{len(posts)}",
                            "text": text[:2000],
                            "timestamp": None,
                            "type": "reply",
                            "metadata": {},
                        })
                time.sleep(0.8)
            except Exception:
                break

        # Scrape topics pages
        for page in range(1, 6):
            if len(posts) >= max_posts:
                break
            try:
                r = requests.get(
                    f"https://www.v2ex.com/member/{username}/topics",
                    params={"p": page}, headers=web_headers, timeout=12
                )
                if not r.ok:
                    break
                # Topic titles are in <span class="item_title"> and content in topic pages
                titles = re.findall(
                    r'<span[^>]+class=["\']item_title["\'][^>]*>.*?<a[^>]+>(.*?)</a>',
                    r.text, re.DOTALL
                )
                for title in titles:
                    text = re.sub(r"<[^>]+>", " ", title).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": f"{username}_topic_{len(posts)}",
                            "text": text[:2000],
                            "timestamp": None,
                            "type": "topic",
                            "metadata": {},
                        })
                if not titles:
                    break
                time.sleep(0.8)
            except Exception:
                break

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "v2ex",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def _get_profile(self, username: str) -> dict:
        try:
            r = requests.get(f"{self.BASE_V1}/members/show.json",
                           params={"username": username},
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return {}
            d = r.json()
            return {
                "id": d.get("id"),
                "username": d.get("username", ""),
                "bio": d.get("bio") or "",
                "website": d.get("website") or "",
                "github": d.get("github") or "",  # V2EX has a dedicated github field!
                "twitter": d.get("twitter") or "",
                "location": d.get("location") or "",
                "tagline": d.get("tagline") or "",
            }
        except Exception:
            return {}

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links — V2EX has a dedicated github field."""
        links = []
        try:
            profile = self._get_profile(username)

            # Dedicated github field (highest confidence)
            gh = profile.get("github", "").strip()
            if gh:
                gh_user = gh.strip("/").split("/")[-1].split("?")[0]
                if gh_user and len(gh_user) >= 2:
                    links.append({
                        "from": f"v2ex:{username}",
                        "to": f"github:{gh_user}",
                        "source": "v2ex_github_field",
                        "confidence": "high",
                    })

            # Bio/website
            for text in [profile.get("bio", ""), profile.get("website", "")]:
                for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text or ""):
                    gh_user = m.group(1).rstrip(".,;:)")
                    links.append({
                        "from": f"v2ex:{username}",
                        "to": f"github:{gh_user}",
                        "source": "v2ex_bio",
                        "confidence": "medium",
                    })
        except Exception:
            pass
        return links


# ─── Twitter/X via Apify ──────────────────────────────────────────────────────

class TwitterCollector:
    """
    Twitter/X via Apify automation-lab/twitter-scraper.
    Requires APIFY_TOKEN env var.
    Requires TWITTER_AUTH_TOKEN + TWITTER_CT0 env vars (from browser cookies).
    Cost: ~$0.30 per 1,000 tweets.
    """
    BASE = "https://api.apify.com/v2"

    def __init__(self):
        self.token = os.getenv("APIFY_TOKEN")
        if not self.token:
            raise ValueError("Set APIFY_TOKEN env var")
        self.auth_token = os.getenv("TWITTER_AUTH_TOKEN", "")
        self.ct0 = os.getenv("TWITTER_CT0", "")

    def collect(self, username: str, max_posts: int = 500) -> dict:
        console.print(f"  [cyan]Twitter[/cyan] fetching [bold]{username}[/bold]...", end=" ")

        run_input = {
            "usernames": [username],
            "tweetsDesired": max_posts,
            "mode": "user-tweets",
        }
        if self.auth_token and self.ct0:
            run_input["cookies"] = f"auth_token={self.auth_token}; ct0={self.ct0}"

        posts = []
        try:
            r = requests.post(
                f"{self.BASE}/acts/automation-lab~twitter-scraper/runs",
                params={"token": self.token},
                json={"input": run_input},
                timeout=30,
            )
            if not r.ok:
                console.print(f"[red]Failed to start: {r.status_code}[/red]")
                return self._empty(username)

            run_id = r.json().get("data", {}).get("id")
            if not run_id:
                console.print(f"[red]No run ID[/red]")
                return self._empty(username)

            # Poll for completion
            for _ in range(60):
                time.sleep(5)
                s = requests.get(f"{self.BASE}/actor-runs/{run_id}",
                               params={"token": self.token}, timeout=15).json()
                status = s.get("data", {}).get("status", "")
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    break

            if status != "SUCCEEDED":
                console.print(f"[red]Run {status}[/red]")
                return self._empty(username)

            dataset_id = s.get("data", {}).get("defaultDatasetId")
            items_r = requests.get(
                f"{self.BASE}/datasets/{dataset_id}/items",
                params={"token": self.token, "limit": max_posts, "format": "json"},
                timeout=30,
            )
            items = items_r.json()
            if not isinstance(items, list):
                return self._empty(username)

            for item in items:
                # Handle various field names from different actors
                text = (item.get("text") or item.get("full_text") or
                        item.get("tweetText") or item.get("content") or "").strip()
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                ts = (item.get("created_at") or item.get("createdAt") or
                      item.get("timestamp") or item.get("date") or "")
                posts.append({
                    "id": str(item.get("id") or item.get("tweetId") or len(posts)),
                    "text": text[:2000],
                    "timestamp": ts,
                    "type": "tweet",
                    "metadata": {
                        "likes": item.get("likeCount") or item.get("favorite_count", 0),
                        "retweets": item.get("retweetCount") or item.get("retweet_count", 0),
                        "lang": item.get("lang", ""),
                        "is_reply": bool(item.get("inReplyToId") or item.get("in_reply_to_status_id")),
                    }
                })

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return self._empty(username)

        console.print(f"[green]{len(posts)} posts[/green]")
        return {
            "platform": "twitter",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": {},
        }

    def _empty(self, username):
        return {"platform": "twitter", "username": username,
                "collected_at": _now_iso(), "posts": [], "post_count": 0, "profile": {}}


# ─── Collector Registry ───────────────────────────────────────────────────────

def get_collector(platform: str):
    return {
        "hn":      HNCollector,
        "stackoverflow": StackOverflowCollector,
        "github":  GitHubCollector,
        "reddit":  RedditCollector,
        "youtube": YouTubeCollector,
        "habr":    HabrCollector,
        "v2ex":    V2EXCollector,
        "twitter": TwitterCollector,
    }.get(platform)


def collect_account(platform: str, username: str, max_posts: int = 300) -> Optional[dict]:
    """
    Top-level entry point. Returns normalized account data or None.
    """
    cls = get_collector(platform)
    if cls is None:
        console.print(f"[red]No collector for platform: {platform}[/red]")
        return None
    try:
        collector = cls()
        if platform == "youtube":
            return collector.collect_by_channel(username, max_posts)
        return collector.collect(username, max_posts)
    except Exception as e:
        console.print(f"[red]Collection failed for {platform}:{username} — {e}[/red]")
        return None


def mine_identity_links(platform: str, username: str) -> list[dict]:
    """Mine cross-platform self-disclosures from profile bios."""
    cls = get_collector(platform)
    if cls is None:
        return []
    try:
        collector = cls()
        if hasattr(collector, "find_cross_platform_links"):
            return collector.find_cross_platform_links(username)
    except Exception:
        pass
    return []


# ─── Stack Overflow ───────────────────────────────────────────────────────────

class StackOverflowCollector:
    """
    Stack Overflow via public API v2.3. No auth needed (10k req/day).
    With API key (free): 100k req/day.
    Supports all Stack Exchange sites: stackoverflow.com, ru.stackoverflow.com, etc.
    """

    BASE = "https://api.stackexchange.com/2.3"

    def __init__(self, site: str = "stackoverflow"):
        self.site = site
        self.key = os.getenv("STACKOVERFLOW_KEY")  # optional free key

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        p = {"site": self.site, "filter": "withbody", "pagesize": 100}
        if self.key:
            p["key"] = self.key
        if params:
            p.update(params)
        try:
            r = _safe_get(f"{self.BASE}{path}", params=p)
            if not r:
                return None
            data = r.json()
            # Respect backoff
            if data.get("backoff"):
                time.sleep(data["backoff"])
            return data
        except Exception as e:
            console.print(f"[yellow]SO error: {e}[/yellow]")
            return None

    def get_user_id(self, username: str) -> Optional[int]:
        """Resolve display name or numeric ID to user ID."""
        if username.isdigit():
            return int(username)
        data = self._get("/users", {"inname": username, "pagesize": 5, "filter": "default"})
        if not data or not data.get("items"):
            return None
        # Find exact match first, then closest
        for item in data["items"]:
            if item["display_name"].lower() == username.lower():
                return item["user_id"]
        return data["items"][0]["user_id"]

    def get_profile(self, user_id: int) -> dict:
        """Get profile including GitHub link if set."""
        data = self._get(f"/users/{user_id}", {"filter": "!*236Ig7v1z2MLe(H"})
        if not data or not data.get("items"):
            return {}
        item = data["items"][0]
        return {
            "display_name": item.get("display_name", ""),
            "website": item.get("website_url", ""),
            "location": item.get("location", ""),
            "reputation": item.get("reputation", 0),
            "github": item.get("github", ""),  # SO has a dedicated GitHub field
            "about": re.sub(r"<[^>]+>", " ", item.get("about_me", "") or ""),
        }

    def collect(self, username: str, max_posts: int = 300) -> dict:
        console.print(f"  [blue]StackOverflow[/blue] fetching [bold]{username}[/bold]...", end=" ")

        user_id = self.get_user_id(username)
        if not user_id:
            console.print(f"[red]User not found[/red]")
            return {"platform": "stackoverflow", "username": username,
                    "collected_at": _now_iso(), "posts": [], "post_count": 0, "profile": {}}

        profile = self.get_profile(user_id)
        posts = []
        page = 1

        while len(posts) < max_posts:
            data = self._get(f"/users/{user_id}/answers",
                           {"page": page, "sort": "creation", "order": "desc"})
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                body = re.sub(r"<[^>]+>", " ", item.get("body", "") or "")
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": str(item.get("answer_id")),
                        "text": body[:2000],
                        "timestamp": datetime.fromtimestamp(
                            item.get("creation_date", 0), tz=timezone.utc
                        ).isoformat(),
                        "type": "answer",
                        "metadata": {
                            "score": item.get("score", 0),
                            "is_accepted": item.get("is_accepted", False),
                            "question_id": item.get("question_id"),
                        }
                    })
            if not data.get("has_more") or len(posts) >= max_posts:
                break
            page += 1
            time.sleep(0.5)

        # Also get comments
        page = 1
        while len(posts) < max_posts:
            data = self._get(f"/users/{user_id}/comments",
                           {"page": page, "sort": "creation", "order": "desc"})
            if not data or not data.get("items"):
                break
            for item in data["items"]:
                body = re.sub(r"<[^>]+>", " ", item.get("body", "") or "")
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) >= MIN_TEXT_LENGTH:
                    posts.append({
                        "id": str(item.get("comment_id")),
                        "text": body[:2000],
                        "timestamp": datetime.fromtimestamp(
                            item.get("creation_date", 0), tz=timezone.utc
                        ).isoformat(),
                        "type": "comment",
                        "metadata": {"score": item.get("score", 0)},
                    })
            if not data.get("has_more") or len(posts) >= max_posts:
                break
            page += 1
            time.sleep(0.5)

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "stackoverflow",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links in SO profile."""
        user_id = self.get_user_id(username)
        if not user_id:
            return []
        profile = self.get_profile(user_id)
        links = []
        # Check dedicated GitHub field
        gh = profile.get("github", "")
        if gh:
            gh_user = gh.strip("/").split("/")[-1].split("?")[0]
            if gh_user:
                links.append({
                    "from": f"stackoverflow:{username}",
                    "to": f"github:{gh_user}",
                    "source": "stackoverflow_github_field",
                    "confidence": "high",
                })
        # Also check website and about
        for text in [profile.get("website",""), profile.get("about","")]:
            for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text):
                gh_user = m.group(1).rstrip(".,;:)")
                if gh_user.lower() not in {"sponsors","orgs","apps","marketplace"}:
                    links.append({
                        "from": f"stackoverflow:{username}",
                        "to": f"github:{gh_user}",
                        "source": "stackoverflow_bio",
                        "confidence": "medium",
                    })
        return links


# ─── Habr (habr.com) ──────────────────────────────────────────────────────────

class HabrCollector:
    """
    Habr.com — Russian developer platform (like HN + Medium combined).
    Completely public API, no auth needed.
    Profiles often link GitHub directly.
    Language: Russian (primarily), some English articles.
    """
    BASE = "https://habr.com/api/v2"
    HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)", "Accept": "application/json"}

    def collect(self, username: str, max_posts: int = 300) -> dict:
        posts = []
        console.print(f"  [red]Habr[/red] fetching [bold]{username}[/bold]...", end=" ")

        profile = self._get_profile(username)

        # Fetch articles (long-form posts)
        page = 1
        while len(posts) < max_posts:
            try:
                r = requests.get(
                    f"{self.BASE}/users/{username}/articles",
                    params={"page": page, "perPage": 20, "sort": "published"},
                    headers=self.HEADERS, timeout=12
                )
                if r.status_code == 404:
                    break
                if not r.ok:
                    break
                data = r.json()
                items = data.get("articleRefs") or data.get("articles") or []
                if not items:
                    break
                for item in items:
                    # Get full article text
                    article_id = item.get("id") or item.get("alias")
                    if article_id:
                        text = self._get_article_text(article_id)
                        if text and len(text) >= MIN_TEXT_LENGTH:
                            posts.append({
                                "id": str(article_id),
                                "text": text[:3000],
                                "timestamp": item.get("timePublished") or item.get("publishedAt"),
                                "type": "article",
                                "metadata": {
                                    "title": item.get("titleHtml") or item.get("title", ""),
                                    "score": item.get("statistics", {}).get("score", 0),
                                }
                            })
                        time.sleep(0.3)
                page += 1
                if len(items) < 20:
                    break
                time.sleep(0.5)
            except Exception as e:
                break

        # Fetch comments
        page = 1
        while len(posts) < max_posts:
            try:
                r = requests.get(
                    f"{self.BASE}/users/{username}/comments",
                    params={"page": page, "perPage": 50},
                    headers=self.HEADERS, timeout=12
                )
                if not r.ok:
                    break
                data = r.json()
                items = data.get("commentRefs") or data.get("comments") or []
                if not items:
                    break
                for item in items:
                    text = re.sub(r"<[^>]+>", " ", item.get("textHtml") or item.get("text") or "")
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": str(item.get("id")),
                            "text": text[:2000],
                            "timestamp": item.get("timePublished") or item.get("publishedAt"),
                            "type": "comment",
                            "metadata": {"score": item.get("score", 0)},
                        })
                page += 1
                if len(items) < 50:
                    break
                time.sleep(0.5)
            except Exception:
                break

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "habr",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def _get_profile(self, username: str) -> dict:
        try:
            r = requests.get(f"{self.BASE}/users/{username}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return {}
            d = r.json()
            return {
                "alias": d.get("alias", ""),
                "fullname": d.get("fullname", ""),
                "bio": re.sub(r"<[^>]+>", " ", d.get("aboutHtml") or d.get("about") or ""),
                "website": d.get("contacts", {}).get("url", "") if isinstance(d.get("contacts"), dict) else "",
                "github": "",  # extracted separately in find_cross_platform_links
                "score": d.get("score", 0),
                "rating": d.get("rating", 0),
            }
        except Exception:
            return {}

    def _get_article_text(self, article_id) -> str:
        try:
            r = requests.get(f"{self.BASE}/articles/{article_id}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return ""
            d = r.json()
            html = d.get("textHtml") or d.get("text") or ""
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return ""

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links in Habr profile."""
        links = []
        try:
            r = requests.get(f"{self.BASE}/users/{username}",
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return []
            d = r.json()

            # Check social links/contacts
            contacts = d.get("contacts") or {}
            if isinstance(contacts, dict):
                github = contacts.get("github") or contacts.get("Github") or ""
                if github:
                    gh_user = github.strip("/").split("/")[-1]
                    if gh_user:
                        links.append({
                            "from": f"habr:{username}",
                            "to": f"github:{gh_user}",
                            "source": "habr_contacts_field",
                            "confidence": "high",
                        })

            # Check bio text
            bio = re.sub(r"<[^>]+>", " ", d.get("aboutHtml") or d.get("about") or "")
            for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", bio):
                gh_user = m.group(1).rstrip(".,;:)")
                links.append({
                    "from": f"habr:{username}",
                    "to": f"github:{gh_user}",
                    "source": "habr_bio",
                    "confidence": "medium",
                })
        except Exception:
            pass
        return links


# ─── V2EX (v2ex.com) ─────────────────────────────────────────────────────────

class V2EXCollector:
    """
    V2EX — Chinese developer forum, like HN for Chinese tech community.
    Completely public JSON API, no auth needed.
    Language: Chinese (primarily), some English.
    Profiles sometimes link GitHub.
    """
    BASE = "https://www.v2ex.com/api/v2"
    BASE_V1 = "https://www.v2ex.com/api"
    HEADERS = {"User-Agent": "OSINTResearch/1.0 (academic)"}

    def collect(self, username: str, max_posts: int = 300) -> dict:
        posts = []
        console.print(f"  [yellow]V2EX[/yellow] fetching [bold]{username}[/bold]...", end=" ")

        profile = self._get_profile(username)

        # V2EX API endpoints for replies/topics are dead — scrape HTML instead
        web_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        # Scrape replies pages
        for page in range(1, 11):
            if len(posts) >= max_posts:
                break
            try:
                r = requests.get(
                    f"https://www.v2ex.com/member/{username}/replies",
                    params={"p": page}, headers=web_headers, timeout=12
                )
                if not r.ok:
                    break
                # Extract reply content from HTML
                # Replies are in <div class="reply_content"> tags
                reply_texts = re.findall(
                    r'<div[^>]+class=["\']reply_content["\'][^>]*>(.*?)</div>',
                    r.text, re.DOTALL
                )
                if not reply_texts:
                    break
                for raw in reply_texts:
                    text = re.sub(r"<[^>]+>", " ", raw)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": f"{username}_reply_{len(posts)}",
                            "text": text[:2000],
                            "timestamp": None,
                            "type": "reply",
                            "metadata": {},
                        })
                time.sleep(0.8)
            except Exception:
                break

        # Scrape topics pages
        for page in range(1, 6):
            if len(posts) >= max_posts:
                break
            try:
                r = requests.get(
                    f"https://www.v2ex.com/member/{username}/topics",
                    params={"p": page}, headers=web_headers, timeout=12
                )
                if not r.ok:
                    break
                # Topic titles are in <span class="item_title"> and content in topic pages
                titles = re.findall(
                    r'<span[^>]+class=["\']item_title["\'][^>]*>.*?<a[^>]+>(.*?)</a>',
                    r.text, re.DOTALL
                )
                for title in titles:
                    text = re.sub(r"<[^>]+>", " ", title).strip()
                    if len(text) >= MIN_TEXT_LENGTH:
                        posts.append({
                            "id": f"{username}_topic_{len(posts)}",
                            "text": text[:2000],
                            "timestamp": None,
                            "type": "topic",
                            "metadata": {},
                        })
                if not titles:
                    break
                time.sleep(0.8)
            except Exception:
                break

        console.print(f"[green]{min(len(posts), max_posts)} posts[/green]")
        return {
            "platform": "v2ex",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": profile,
        }

    def _get_profile(self, username: str) -> dict:
        try:
            r = requests.get(f"{self.BASE_V1}/members/show.json",
                           params={"username": username},
                           headers=self.HEADERS, timeout=10)
            if not r.ok:
                return {}
            d = r.json()
            return {
                "id": d.get("id"),
                "username": d.get("username", ""),
                "bio": d.get("bio") or "",
                "website": d.get("website") or "",
                "github": d.get("github") or "",  # V2EX has a dedicated github field!
                "twitter": d.get("twitter") or "",
                "location": d.get("location") or "",
                "tagline": d.get("tagline") or "",
            }
        except Exception:
            return {}

    def find_cross_platform_links(self, username: str) -> list:
        """Find GitHub links — V2EX has a dedicated github field."""
        links = []
        try:
            profile = self._get_profile(username)

            # Dedicated github field (highest confidence)
            gh = profile.get("github", "").strip()
            if gh:
                gh_user = gh.strip("/").split("/")[-1].split("?")[0]
                if gh_user and len(gh_user) >= 2:
                    links.append({
                        "from": f"v2ex:{username}",
                        "to": f"github:{gh_user}",
                        "source": "v2ex_github_field",
                        "confidence": "high",
                    })

            # Bio/website
            for text in [profile.get("bio", ""), profile.get("website", "")]:
                for m in re.finditer(r"github\.com/([A-Za-z0-9][A-Za-z0-9_-]{1,38})", text or ""):
                    gh_user = m.group(1).rstrip(".,;:)")
                    links.append({
                        "from": f"v2ex:{username}",
                        "to": f"github:{gh_user}",
                        "source": "v2ex_bio",
                        "confidence": "medium",
                    })
        except Exception:
            pass
        return links


# ─── Twitter/X via Apify ──────────────────────────────────────────────────────

class TwitterCollector:
    """
    Twitter/X via Apify's tweet scraper actors.
    Requires APIFY_TOKEN env var.
    Cost: ~$0.25-0.40 per 1,000 tweets.
    
    Collects full tweet history by username using advanced search.
    """
    # Best value actor for profile scraping
    ACTOR_ID = "apidojo/tweet-scraper"
    BASE = "https://api.apify.com/v2"

    def __init__(self):
        self.token = os.getenv("APIFY_TOKEN")
        if not self.token:
            raise ValueError("Set APIFY_TOKEN env var")

    def collect(self, username: str, max_posts: int = 500) -> dict:
        console.print(f"  [cyan]Twitter[/cyan] fetching [bold]{username}[/bold]...", end=" ")

        # Use Twitter advanced search to get user's tweets
        # from:username gives full history, not just recent
        # Use pay-per-result actor — works correctly on Starter plan
        # $0.25/1000 tweets, no demo mode cap
        run_input = {
            "usernames": [username],
            "maxItems": max_posts,
            "queryType": "Latest",
        }

        posts = []
        try:
            # Start the actor run
            r = requests.post(
                f"{self.BASE}/acts/kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest/runs",
                params={"token": self.token},
                json={"input": run_input},
                timeout=30,
            )
            if not r.ok:
                # Fallback to scweet actor
                run_input2 = {
                    "profile_urls": [f"https://twitter.com/{username}"],
                    "max_tweets": max_posts,
                }
                r = requests.post(
                    f"{self.BASE}/acts/altimis~scweet/runs",
                    params={"token": self.token},
                    json={"input": run_input2},
                    timeout=30,
                )
            if not r.ok:
                console.print(f"[red]Failed to start actor: {r.status_code}[/red]")
                return self._empty(username)

            run_id = r.json().get("data", {}).get("id")
            if not run_id:
                console.print(f"[red]No run ID returned[/red]")
                return self._empty(username)

            # Wait for completion (poll every 5s, max 5 min)
            for _ in range(60):
                time.sleep(5)
                status_r = requests.get(
                    f"{self.BASE}/actor-runs/{run_id}",
                    params={"token": self.token},
                    timeout=15,
                )
                if not status_r.ok:
                    continue
                status = status_r.json().get("data", {}).get("status", "")
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    break

            if status != "SUCCEEDED":
                console.print(f"[red]Run {status}[/red]")
                return self._empty(username)

            # Fetch results
            dataset_id = status_r.json().get("data", {}).get("defaultDatasetId")
            items_r = requests.get(
                f"{self.BASE}/datasets/{dataset_id}/items",
                params={"token": self.token, "limit": max_posts, "format": "json"},
                timeout=30,
            )
            if not items_r.ok:
                return self._empty(username)

            for item in items_r.json():
                text = item.get("text") or item.get("full_text") or item.get("tweetText") or ""
                text = text.strip()
                if len(text) < MIN_TEXT_LENGTH:
                    continue

                # Parse timestamp
                ts = (item.get("created_at") or item.get("createdAt") or
                      item.get("timestamp") or "")

                posts.append({
                    "id": str(item.get("id") or item.get("tweetId") or len(posts)),
                    "text": text[:2000],
                    "timestamp": ts,
                    "type": "tweet",
                    "metadata": {
                        "likes": item.get("likeCount") or item.get("favorite_count", 0),
                        "retweets": item.get("retweetCount") or item.get("retweet_count", 0),
                        "lang": item.get("lang", ""),
                        "is_reply": bool(item.get("inReplyToId") or item.get("in_reply_to_status_id")),
                    }
                })

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return self._empty(username)

        console.print(f"[green]{len(posts)} posts[/green]")
        return {
            "platform": "twitter",
            "username": username,
            "collected_at": _now_iso(),
            "posts": posts[:max_posts],
            "post_count": min(len(posts), max_posts),
            "profile": {},
        }

    def _empty(self, username: str) -> dict:
        return {
            "platform": "twitter", "username": username,
            "collected_at": _now_iso(), "posts": [], "post_count": 0, "profile": {}
        }
