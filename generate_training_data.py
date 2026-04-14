"""
generate_training_data.py
Generates realistic synthetic account data for training when live APIs
are unavailable (sandbox / offline mode).

Each "identity" gets a stable behavioral fingerprint seeded from their name.
Same identity on two platforms = slightly different surface text, same deep style.
Different identity = different fingerprint entirely.

Run: python generate_training_data.py
"""

import json
import random
import math
from pathlib import Path

# ─── Identity templates ───────────────────────────────────────────────────────
# Each identity has: style, topics, function_word_tics, hour_peak, verbosity
# These map to real behavioral archetypes found on HN/GitHub

IDENTITIES = [
    # (name, style_profile)
    # Verbose security researcher — long sentences, "however", "actually", late night
    ("tptacek",      "verbose_security",   [22,23,0,1],    "hn",      "github"),
    ("pg",           "terse_wise",         [9,10,20,21],   "hn",      "github"),
    ("dang",         "measured_moderator", [10,11,14,15],  "hn",      "github"),
    ("patio11",      "businessy_verbose",  [8,9,10,21,22], "hn",      "github"),
    ("jacquesm",     "dutch_technical",    [7,8,9,14],     "hn",      "github"),
    ("simonw",       "builder_terse",      [9,10,11,15],   "hn",      "github"),
    ("feross",       "open_source_casual", [14,15,22,23],  "hn",      "github"),
    ("jaredpalmer",  "frontend_energetic", [9,10,23,0],    "hn",      "github"),
    ("ingve",        "nordic_precise",     [8,9,10,14],    "hn",      "github"),
    ("vidarh",       "systems_thoughtful", [21,22,23,0],   "hn",      "github"),
    ("zdw",          "ops_terse",          [9,10,11,14],   "hn",      "github"),
    ("mooreds",      "auth_practical",     [8,9,10,15],    "hn",      "github"),
    ("danso",        "data_journalist",    [10,11,14,15],  "hn",      "github"),
    ("swyx",         "dx_enthusiast",      [9,10,22,23],   "hn",      "github"),
    ("ColinWright",  "juggler_precise",    [7,8,9,14],     "hn",      "github"),
    # Extra identities for negative pool variety
    ("user_alpha",   "startup_casual",     [9,10,11],      "hn",      "reddit"),
    ("user_beta",    "academic_formal",    [14,15,16],     "hn",      "github"),
    ("user_gamma",   "sysadmin_terse",     [9,10,11,14],   "reddit",  "github"),
    ("user_delta",   "vc_jargon",          [8,9,10],       "hn",      "reddit"),
    ("user_epsilon", "ml_researcher",      [22,23,0,1],    "hn",      "github"),
    ("user_zeta",    "devops_practical",   [9,10,11,15],   "reddit",  "github"),
    ("user_eta",     "frontend_casual",    [14,15,22,23],  "hn",      "reddit"),
    ("user_theta",   "security_paranoid",  [22,23,0,1,2],  "hn",      "github"),
    ("user_iota",    "entrepreneur_hyped", [8,9,10,21],    "hn",      "reddit"),
    ("user_kappa",   "rust_evangelist",    [21,22,23,0],   "hn",      "github"),
    ("user_lambda",  "python_pragmatist",  [9,10,11,14],   "reddit",  "github"),
    ("user_mu",      "database_nerd",      [8,9,14,15],    "hn",      "github"),
    ("user_nu",      "infra_lead",         [9,10,11],      "reddit",  "github"),
    ("user_xi",      "product_manager",    [9,10,11,14],   "hn",      "reddit"),
    ("user_omicron", "open_source_vet",    [21,22,23,0],   "hn",      "github"),
]

STYLE_TEMPLATES = {
    "verbose_security": {
        "hn": [
            "However, this approach fundamentally misunderstands the threat model. The actual attack surface here is considerably larger than you'd expect.",
            "tbh the security implications here are non-trivial. I've seen this exact pattern exploited in production environments.",
            "Actually, if you read the RFC carefully, the behavior you're describing is explicitly undefined. This has real consequences for implementations.",
            "The real problem is that people keep conflating authentication with authorization. These are genuinely different problems that require different solutions.",
            "I'd push back on this framing. The naive implementation fails in at least three ways I can immediately think of.",
            "This is a well-known class of vulnerability. The standard mitigation doesn't work here because the attacker controls the timing.",
            "However, the attack requires a man-in-the-middle position, which changes the threat model considerably.",
            "The interesting thing about this is that the obvious fix makes things worse, not better. The correct solution is counterintuitive.",
        ],
        "github": [
            "This PR addresses the authentication bypass we discussed in the security review. The fix is minimal but the implications are significant.",
            "Reviewed. The cryptographic assumptions here are fragile — we should add a comment explaining why this is safe in our specific threat model.",
            "LGTM modulo the nit about error handling. We should never silently swallow crypto errors.",
            "The timing attack surface in this implementation is real. Added a constant-time comparison function.",
            "Fixed the session fixation vulnerability. Tests updated. The root cause was incorrect trust boundary.",
        ],
    },
    "terse_wise": {
        "hn": [
            "The best founders are often wrong about why they succeed.",
            "Investors pattern-match on founders, not ideas. This is mostly correct.",
            "The hard part isn't the idea. Everyone has ideas.",
            "Most startup advice is wrong because it's averages of survivors.",
            "The real question is whether users want this, not whether it's technically impressive.",
            "Pivots are usually reframings, not changes.",
            "The best time to raise is when you don't need to.",
            "Most moats are temporary. The ones that aren't are usually network effects.",
        ],
        "github": [
            "Simplified. The original was overengineered.",
            "Fixed.",
            "This approach works but there's a cleaner way. See alternative branch.",
            "Added tests. The edge case was subtle.",
            "Reverted. The complexity isn't worth it.",
        ],
    },
    "measured_moderator": {
        "hn": [
            "Please keep comments civil and on-topic. We're trying to have a substantive discussion here.",
            "This thread has gone off the rails a bit. Let's focus on the technical merits.",
            "That's an interesting point, though I'd note the article actually says something slightly different.",
            "We've had this discussion before. The community consensus seems to be roughly what you'd expect.",
            "I appreciate the passion but the tone isn't helping the discussion.",
            "To be fair to the original poster, the question is a reasonable one even if the framing is imperfect.",
        ],
        "github": [
            "Thanks for the contribution. A few questions before we merge.",
            "This looks good. Can you add a test for the edge case?",
            "I'll take a look at this when I get a chance.",
            "Closing as won't fix — the current behavior is intentional.",
        ],
    },
    "businessy_verbose": {
        "hn": [
            "This is actually a really important insight that most people in the industry miss. The business model implications are significant.",
            "I've seen this pattern play out at several companies. The failure mode is almost always the same: you optimize for the metric instead of the underlying thing the metric was supposed to measure.",
            "The interesting thing here is that the unit economics only work at scale, which means you need to get to scale before you can prove the model works, which is a chicken-and-egg problem.",
            "tbh most of the advice in this space is cargo-culting from companies that succeeded for very different reasons than people think.",
            "This is a solved problem in Japan. The interesting question is why the US hasn't adopted it.",
            "The regulatory capture angle here is underappreciated. This isn't just about technology.",
            "Actually the more interesting question is why this hasn't happened sooner. The technology has been available for years.",
        ],
        "github": [
            "Adds support for the enterprise pricing tier. The implementation is straightforward but required touching several files.",
            "Fixed the billing edge case. The root cause was an off-by-one error in the proration calculation.",
            "This PR implements the feature requested in multiple customer conversations. Tests passing.",
            "Refactored the pricing engine. The old implementation was correct but impossible to reason about.",
        ],
    },
    "builder_terse": {
        "hn": [
            "I built this. Happy to answer questions.",
            "The hard part was the incremental indexing, not the query engine.",
            "SQLite is underrated for this use case.",
            "Shipped this yesterday. 3k users in 24 hours, mostly from HN.",
            "The interesting finding was that users didn't want the feature we thought they wanted.",
            "Source is on GitHub if anyone wants to dig into the implementation.",
            "Wrote a blog post explaining the architecture decisions.",
        ],
        "github": [
            "Added the feature. It's a bit hacky but it works.",
            "Fixed the performance regression. The culprit was an N+1 query.",
            "Pushed a new release. Changelog in the PR description.",
            "Tests are now passing on all platforms.",
            "This was requested a lot. Finally got around to it.",
        ],
    },
    "dutch_technical": {
        "hn": [
            "This is interesting but the scalability story is unclear. What happens at 10x the current load?",
            "I've done similar work in the past. The main gotcha is that the naive approach fails silently.",
            "The paper this is based on has some serious methodological issues that the authors don't address.",
            "In my experience the operational complexity is the killer, not the initial implementation.",
            "This is technically correct but misses the practical constraints that make the obvious solution unworkable.",
        ],
        "github": [
            "The implementation is correct but the error handling needs work. We should never return an empty list on failure.",
            "Reviewed. The algorithm is O(n log n) but the constant factor is too high for the expected input sizes.",
            "Fixed the race condition. The original code had an implicit assumption about ordering that wasn't guaranteed.",
            "This works but we should add documentation explaining the invariants.",
        ],
    },
    "open_source_casual": {
        "hn": [
            "Just shipped v2 of the library. Breaking changes but it was necessary.",
            "The npm ecosystem has real problems but the alternatives are worse in different ways.",
            "WebAssembly is finally ready for production use in my opinion.",
            "The browser security model is simultaneously too restrictive and not restrictive enough.",
            "lol at the irony of a security company shipping vulnerable code.",
        ],
        "github": [
            "BREAKING: removed the deprecated API. Migration guide in the readme.",
            "Added TypeScript types. Finally.",
            "Fixed the memory leak that's been annoying everyone.",
            "v2.0.0 is out. Thanks to everyone who contributed.",
        ],
    },
    "startup_casual": {
        "hn": [
            "We're building in this space. Happy to share what we've learned.",
            "The market is bigger than people think.",
            "Launched last week. Onboarding 50 users/day.",
            "We tried this. Didn't work for us. Here's why.",
            "The enterprise version of this problem is actually unsolved.",
        ],
        "reddit": [
            "Founder here. AMA.",
            "We just raised a seed. Still looking for our first 10 customers.",
            "The advice I'd give myself: talk to users earlier.",
            "YC was worth it for us. YMMV.",
        ],
    },
    "academic_formal": {
        "hn": [
            "The methodology in this paper is sound but the conclusions are overstated.",
            "There is a substantial body of prior work in this area that the authors fail to cite.",
            "The statistical analysis would benefit from a more careful treatment of confounding variables.",
            "This result is consistent with what has been observed in controlled settings.",
            "However, the sample size is insufficient to draw the conclusions the authors claim.",
        ],
        "github": [
            "Implementation of the algorithm described in the referenced paper.",
            "The time complexity is O(n^2) in the worst case but O(n log n) in practice.",
            "Added references to the relevant literature in the documentation.",
            "Corrected the implementation to match the formal specification.",
        ],
    },
    "sysadmin_terse": {
        "reddit": [
            "Just use ansible.",
            "This is a solved problem. Terraform.",
            "Have you tried turning it off and on again? Seriously though, it's probably DNS.",
            "Your firewall rules are wrong. They're always wrong.",
            "Check your logs. All of them.",
        ],
        "github": [
            "Added the systemd service file.",
            "Fixed the init script.",
            "The Dockerfile was broken. Fixed.",
            "Updated the config management playbook.",
        ],
    },
    "ml_researcher": {
        "hn": [
            "The loss landscape intuition is wrong here. The actual geometry is considerably more complex.",
            "Attention is not what people think it is. The interpretability story is very shaky.",
            "This benchmark is gameable and everyone who works in the area knows it.",
            "The scaling hypothesis has held up remarkably well. That's the actually surprising thing.",
            "Transformers work despite the theory, not because of it.",
            "The interesting question is whether the representations are meaningful or just useful.",
        ],
        "github": [
            "Implemented the training loop. The gradient accumulation was tricky.",
            "Added mixed precision training. 40% speedup.",
            "Fixed the evaluation code. The original had a subtle data leakage issue.",
            "Checkpointing now works correctly with distributed training.",
        ],
    },
    "rust_evangelist": {
        "hn": [
            "Rewrite it in Rust. I'm only half joking.",
            "The ownership model prevents this entire class of bug. That's the point.",
            "The compile times are worth it. You catch these bugs at compile time instead of 3am.",
            "Go is fine but the lack of algebraic types is a constant frustration.",
            "Unsafe Rust is still safer than safe C.",
            "The async story has improved dramatically in the last year.",
        ],
        "github": [
            "Rewrote the hot path in unsafe Rust. 10x speedup.",
            "Added the zero-copy parser. No allocations in the common case.",
            "Clippy is now happy.",
            "Replaced the mutex with a lock-free structure. Benchmarks attached.",
        ],
    },
    "frontend_energetic": {
        "hn": [
            "This is going to change how we think about state management.",
            "React is showing its age. The new primitives are much better.",
            "The DX improvements in this release are huge.",
            "Hot take: CSS-in-JS was a mistake and we're all slowly admitting it.",
            "The TypeScript 5.0 features are genuinely exciting.",
        ],
        "reddit": [
            "Just tried this. It's actually incredible.",
            "The bundle size is still too big but the DX is worth it for me.",
            "Anyone else noticing the performance improvements in the latest release?",
            "The docs have gotten so much better.",
        ],
    },
    "nordic_precise": {
        "hn": [
            "The implementation details matter here more than the headline claim.",
            "This is well-engineered. The edge cases are handled correctly.",
            "I would add one caveat: this approach doesn't work if the data is sparse.",
            "The paper is worth reading carefully. The details in the appendix are important.",
            "Precise and correct. Nothing to add.",
        ],
        "github": [
            "The implementation is clean. One suggestion: add an explicit invariant comment.",
            "Edge case handled. Good.",
            "The naming could be more descriptive but the logic is correct.",
            "Added the missing boundary check.",
        ],
    },
    "systems_thoughtful": {
        "hn": [
            "The systems implications of this are more interesting than the application-level story.",
            "Memory models are genuinely hard. This is a good introduction.",
            "The cache coherency story here is subtle and the post glosses over it.",
            "Actually, the interesting thing about this architecture is what it implies about the failure modes.",
            "NUMA effects will kill you at scale. This is often underappreciated.",
            "The lock-free data structure is wrong. It has an ABA problem.",
        ],
        "github": [
            "The memory ordering is wrong here. We need acquire/release semantics, not relaxed.",
            "Added the cache line padding. The false sharing was causing 40% throughput reduction.",
            "Replaced the spinlock with a parking lot mutex. Better latency distribution.",
            "The allocation pattern is suboptimal for NUMA systems. Refactored.",
        ],
    },
    "ops_terse": {
        "hn": [
            "This doesn't work at scale. Source: running this in prod.",
            "The operational complexity is the problem, not the theory.",
            "Kubernetes is the right answer to a question most people aren't asking.",
            "Monitoring is more important than the system being monitored.",
            "If you haven't been paged at 3am by your own code you haven't shipped enough.",
        ],
        "github": [
            "Fixed the deployment script.",
            "Updated the runbook.",
            "The health check was wrong. Fixed.",
            "Added the missing alert.",
        ],
    },
    "auth_practical": {
        "hn": [
            "OAuth is a framework, not a protocol. The details matter enormously.",
            "Most authorization bugs are authorization bugs pretending to be authentication bugs.",
            "Passkeys are the right direction. The UX is still rough.",
            "The interesting thing about FusionAuth is what it reveals about the enterprise requirements.",
            "Delegated authorization is genuinely hard to get right.",
        ],
        "github": [
            "Implemented the PKCE flow. The original was vulnerable to authorization code injection.",
            "Added the token rotation. The refresh token lifetime was too long.",
            "Fixed the redirect URI validation. It was accepting partial matches.",
            "The JWT verification was missing the algorithm check. Critical fix.",
        ],
    },
    "data_journalist": {
        "hn": [
            "The methodology section doesn't support the headline claim.",
            "The interesting finding is buried in table 4 of the supplementary material.",
            "This data is publicly available if you know where to look.",
            "The visualization is misleading — the y-axis doesn't start at zero.",
            "I scraped this data last year. Happy to share the cleaned version.",
        ],
        "github": [
            "Added the data cleaning script. The raw data had several inconsistencies.",
            "Updated the analysis with the corrected figures.",
            "The notebook is reproducible. All data sources are linked.",
            "Fixed the date parsing. The original was silently dropping records.",
        ],
    },
    "dx_enthusiast": {
        "hn": [
            "The developer experience here is genuinely excellent.",
            "This is the API design I've been wanting for years.",
            "The error messages alone make this worth using.",
            "Hot take: most developer tools are built by people who don't use them.",
            "The docs are as important as the code. This team gets that.",
        ],
        "github": [
            "Improved the error messages. They should tell you what to do, not just what went wrong.",
            "Added the helpful hint when the common mistake is detected.",
            "The getting started experience was too hard. Simplified.",
            "Added examples for the common use cases.",
        ],
    },
    "juggler_precise": {
        "hn": [
            "The mathematical structure here is more interesting than it appears on the surface.",
            "This is related to a result in combinatorics that most people haven't seen.",
            "The proof is elegant but the construction is what's surprising.",
            "I've been thinking about this problem for years. The key insight is counterintuitive.",
            "Precise and worth reading carefully.",
        ],
        "github": [
            "The algorithm is correct. The proof is in the attached note.",
            "Edge case: what happens when n=0? Added explicit handling.",
            "The complexity analysis in the comment was wrong. Fixed.",
            "Added the mathematical invariant as a comment.",
        ],
    },
    "vc_jargon": {
        "hn": [
            "This is a massive TAM if the go-to-market is right.",
            "The defensibility story is weak. Margins will compress.",
            "Interesting wedge. What's the expansion motion?",
            "The unit economics need to work at the enterprise tier.",
            "Classic blitzscaling playbook.",
        ],
        "reddit": [
            "The seed round was oversubscribed. Good signal.",
            "This is exactly what Series A investors are looking for right now.",
            "The founder market fit is strong.",
        ],
    },
    "python_pragmatist": {
        "reddit": [
            "Just use pandas. The performance is good enough.",
            "Poetry has made dependency management actually bearable.",
            "Type hints have improved the codebase significantly.",
            "FastAPI is the right answer for most API use cases.",
            "The GIL is only a problem if you're doing CPU-bound work in threads, which you shouldn't be.",
        ],
        "github": [
            "Replaced the subprocess call with a proper library.",
            "Added type hints throughout. mypy is now happy.",
            "The list comprehension was cleaner. Simplified.",
            "Added the __slots__ to reduce memory usage.",
        ],
    },
    "database_nerd": {
        "hn": [
            "The query planner made the wrong choice here. You need an index hint.",
            "MVCC is doing interesting things to your write amplification.",
            "The interesting thing about this benchmark is what it reveals about the storage engine's assumptions.",
            "Normalized schemas are correct. Denormalization is an optimization, not a design.",
            "The window function makes this query 10x cleaner.",
        ],
        "github": [
            "Added the missing index. Explain plan was doing a full table scan.",
            "Replaced the N+1 with a join. Should help with the performance issue.",
            "The migration is reversible. Rollback script included.",
            "Vacuum and analyze after this. The statistics were stale.",
        ],
    },
    "infra_lead": {
        "reddit": [
            "The networking layer is always the problem. Always.",
            "eBPF is going to change everything and we're not ready.",
            "Service mesh adds latency. Make sure you actually need it.",
            "The right number of microservices is one fewer than you have.",
        ],
        "github": [
            "Updated the terraform modules. Tested in staging.",
            "The network policy was too permissive. Tightened.",
            "Added the circuit breaker. The dependency was flaky.",
            "Consolidated the helm charts.",
        ],
    },
    "product_manager": {
        "hn": [
            "The user research didn't support the engineering team's intuition. Surprised everyone.",
            "We shipped this and learned we were solving the wrong problem.",
            "The metric was going up but the underlying behavior was getting worse.",
            "Most product decisions are actually distribution decisions.",
            "The best PRDs are the ones nobody reads because the problem is clear.",
        ],
        "reddit": [
            "Roadmap is set by what customers are actually paying for, not what they ask for.",
            "The discovery process matters more than the execution.",
            "We killed a feature that 30% of users used but 100% hated.",
        ],
    },
    "open_source_vet": {
        "hn": [
            "I've been maintaining open source for 20 years. The sustainability problem is real.",
            "The license choice matters more than people think when you're starting.",
            "Governance is the hard part. The code is easy.",
            "Most OSS projects die of success, not failure.",
            "The tragedy of the commons plays out in dependencies every day.",
        ],
        "github": [
            "CODEOWNERS updated.",
            "Added the contributor covenant. Let's be explicit about expectations.",
            "The release process is now documented.",
            "Deprecated this API. Migration guide attached.",
        ],
    },
    "security_paranoid": {
        "hn": [
            "Assume breach. Design accordingly.",
            "The supply chain attack surface is enormous and mostly unmonitored.",
            "Your threat model is wrong. Everyone's threat model is wrong.",
            "Defense in depth is not a buzzword. It's the only thing that works.",
            "The insider threat is the one nobody wants to talk about.",
            "Zero trust means zero trust. Not 'trust but verify'.",
        ],
        "github": [
            "Pinned the dependency hashes. Supply chain hygiene.",
            "Removed the third-party script. Not worth the attack surface.",
            "Added SAST to the CI pipeline.",
            "The secret was in the git history. Rotated and removed.",
        ],
    },
    "entrepreneur_hyped": {
        "hn": [
            "This is going to be huge. We're still early.",
            "The timing is finally right for this.",
            "We tried this 5 years ago and failed. The infrastructure wasn't there yet.",
            "Disruption is real. The incumbents are too slow to respond.",
            "The AI wave is different this time.",
        ],
        "reddit": [
            "We're hiring. DM me.",
            "The traction is real. 10k users in the first week.",
            "Bootstrapped to $1M ARR. Happy to share what worked.",
        ],
    },
    "devops_practical": {
        "reddit": [
            "CI/CD is the foundation. Everything else is optional.",
            "The deployment should be boring. If it's exciting you've done something wrong.",
            "Observability is not monitoring. The difference matters.",
            "Runbooks should be so clear that someone who's never seen the system can use them.",
        ],
        "github": [
            "Fixed the pipeline. The caching was causing stale builds.",
            "Added the smoke test. Catches the obvious failures before they hit prod.",
            "Updated the deployment workflow. Now uses OIDC instead of long-lived credentials.",
        ],
    },
    "frontend_casual": {
        "hn": [
            "The DX is great but the runtime performance is still too slow for my use case.",
            "I switched from webpack to vite and I'm never going back.",
            "The React Server Components mental model is hard to teach.",
            "CSS has gotten genuinely good in the last three years.",
        ],
        "reddit": [
            "Just use Tailwind. The debate is over.",
            "The component library is fine but the accessibility is terrible.",
            "TypeScript is worth the overhead. Source: learned this the hard way.",
            "The bundle is too big. Every project's bundle is too big.",
        ],
    },
}


def seeded_rng(seed: int):
    """Simple seeded PRNG for reproducibility."""
    class Rng:
        def __init__(self, s): self.s = s
        def rand(self):
            self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
            return self.s / 0xFFFFFFFF
        def randint(self, lo, hi): return int(self.rand() * (hi - lo)) + lo
        def choice(self, lst): return lst[int(self.rand() * len(lst))]
        def sample(self, lst, k): 
            lst2 = list(lst)
            out = []
            for _ in range(k):
                i = int(self.rand() * len(lst2))
                out.append(lst2.pop(i))
            return out
    return Rng(seed)


def str_hash(s: str) -> int:
    h = 5381
    for c in s:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
    return h


def generate_account(name: str, platform: str, style_key: str, hours: list, n_posts: int = 180) -> dict:
    """Generate realistic post data for one account."""
    seed = str_hash(f"{name}:{platform}:{style_key}")
    rng = seeded_rng(seed)

    style = STYLE_TEMPLATES.get(style_key, STYLE_TEMPLATES["builder_terse"])
    templates = style.get(platform, list(style.values())[0])

    posts = []
    for i in range(n_posts):
        h = rng.choice(hours)
        # Add some hour jitter
        h = (h + rng.randint(-1, 2)) % 24
        weekday = rng.randint(0, 6)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        minute = rng.randint(0, 59)
        ts = f"2023-{month:02d}-{day:02d}T{h:02d}:{minute:02d}:00Z"

        # Pick base template and add variation
        base = rng.choice(templates)
        # Occasionally add a second sentence from another template
        if rng.rand() > 0.6:
            extra = rng.choice(templates)
            text = base + " " + extra
        else:
            text = base

        posts.append({
            "id": str(i),
            "text": text,
            "timestamp": ts,
            "type": "comment" if rng.rand() > 0.3 else "story",
            "metadata": {
                "parent_id": str(i - 1) if i > 0 and rng.rand() > 0.4 else None,
                "score": rng.randint(0, 200),
            }
        })

    return {
        "platform": platform,
        "username": name if name.startswith("user_") else name,
        "collected_at": "2024-01-01T00:00:00Z",
        "posts": posts,
        "post_count": len(posts),
        "profile": {
            "bio": f"{name} on {platform}",
            "karma": rng.randint(100, 50000),
        }
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from osint.features import extract as extract_features
    from osint.pipeline import (
        save_raw, save_features, append_link,
        build_identity_groups, build_positive_pairs,
        build_random_negatives, build_hard_negatives,
        identity_split, save_pairs, load_all_features,
        LINKS_FILE, RAW_DIR, FEAT_DIR,
    )
    import shutil

    # Clean slate
    for d in [RAW_DIR, FEAT_DIR, Path("data/pairs")]:
        if d.exists(): shutil.rmtree(d)
    if LINKS_FILE.exists(): LINKS_FILE.unlink()

    print(f"Generating synthetic training data for {len(IDENTITIES)} identities...")
    print()

    all_accounts = {}

    for name, style, hours, plat_a, plat_b in IDENTITIES:
        n_posts = 150 + str_hash(name) % 100   # Vary post count per identity

        for plat in [plat_a, plat_b]:
            print(f"  {plat:8s}:{name} ({style})", end=" ... ")
            data = generate_account(name, plat, style, hours, n_posts=n_posts)
            save_raw(data)
            feat = extract_features(data, use_embeddings=False)
            save_features(feat)
            key = f"{plat}:{name}"
            all_accounts[key] = feat
            print(f"{feat['post_count']} posts | peak {feat['summary']['peak_hour']:02d}:00 | vocab {feat['summary']['vocab_richness']:.2f}")

        # Register the identity link
        link = {
            "from": f"{plat_a}:{name}",
            "to":   f"{plat_b}:{name}",
            "source": "synthetic_ground_truth",
            "confidence": "high",
        }
        append_link(link)

    print(f"\n  Total accounts: {len(all_accounts)}")
    print(f"  Total links: {len(IDENTITIES)}")

    print("\nBuilding training pairs...")
    all_feats = load_all_features()
    groups    = build_identity_groups(all_feats)
    all_keys  = list(all_feats.keys())
    print(f"  Identity groups: {len(groups)}")

    pos_pairs  = build_positive_pairs(groups)
    rand_negs  = build_random_negatives(groups, all_keys, n=len(pos_pairs) * 12)
    hard_negs  = build_hard_negatives(all_feats, groups, n=len(pos_pairs) * 6)
    all_negs   = rand_negs + hard_negs

    print(f"  Positives:       {len(pos_pairs)}")
    print(f"  Random negs:     {len(rand_negs)}")
    print(f"  Hard negs:       {len(hard_negs)}")
    print(f"  Total negatives: {len(all_negs)}")

    split = identity_split(groups, pos_pairs, all_negs)
    print("\nSplit (by identity — no leakage):")
    save_pairs(split, all_feats)
    print("\nData generation complete.")


if __name__ == "__main__":
    main()
