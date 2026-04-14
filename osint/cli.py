"""
osint/cli.py
Main CLI for the OSINT Behavioral Pattern Aggregator.

Commands:
  collect    — fetch posts from one or more accounts
  analyze    — compare two accounts, report similarity + explanation
  search     — find best matches for an account across all collected data
  mine-links — crawl profile bios for cross-platform self-disclosures
  pipeline   — full collect → extract → pair run
  train      — train the Siamese model
  report     — print a structured report for a case
  discover   — auto-discover active users from a platform

Examples:
  python -m osint collect hn:dang github:dhh reddit:spez
  python -m osint analyze hn:pg github:paulgrahm
  python -m osint search hn:tptacek --top 10
  python -m osint discover hn --n 200
  python -m osint pipeline hn:pg github:paulgrahm hn:dang github:dhh
  python -m osint train --epochs 50
  python -m osint report hn:pg
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_account(s: str) -> tuple[str, str]:
    """'hn:username' → ('hn', 'username')"""
    if ":" not in s:
        raise click.BadParameter(f"Format must be platform:username (e.g. hn:pg) — got: {s}")
    parts = s.split(":", 1)
    return parts[0].strip().lower(), parts[1].strip()


def load_feat(platform: str, username: str):
    from .pipeline import load_features
    feat = load_features(platform, username)
    if feat is None:
        console.print(f"[yellow]No features for {platform}:{username} — collecting now...[/yellow]")
        from .pipeline import collect_and_save
        feat = collect_and_save(platform, username)
    if feat is None:
        console.print(f"[red]Could not load or collect {platform}:{username}[/red]")
    return feat


def confidence_label(score: float) -> tuple[str, str]:
    if score >= 0.88: return "VERY HIGH", "bold green"
    if score >= 0.78: return "HIGH",      "green"
    if score >= 0.68: return "MEDIUM",    "yellow"
    if score >= 0.55: return "LOW",       "dim yellow"
    return "UNLIKELY",  "dim"


def verdict(score: float, threshold: float = 0.75) -> str:
    return "LIKELY SAME PERSON" if score >= threshold else "LIKELY DIFFERENT PEOPLE"


# ─── CLI Group ────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """OSINT Behavioral Pattern Aggregator — cross-platform persona correlation."""
    pass


# ─── collect ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("accounts", nargs=-1, required=True)
@click.option("--max-posts", default=300, show_default=True, help="Max posts per account")
@click.option("--force",     is_flag=True, help="Re-collect even if cached")
def collect(accounts, max_posts, force):
    """
    Collect and fingerprint one or more accounts.

    ACCOUNTS: one or more platform:username identifiers
    (e.g. hn:dang github:dhh reddit:spez)
    """
    from .pipeline import collect_and_save
    console.rule("[bold]Collecting accounts[/bold]")

    results = []
    for raw in accounts:
        try:
            platform, username = parse_account(raw)
        except click.BadParameter as e:
            console.print(f"[red]{e}[/red]")
            continue

        feat = collect_and_save(platform, username, max_posts=max_posts, force=force)
        if feat:
            results.append(feat)

    if not results:
        console.print("[red]No accounts collected.[/red]")
        return

    table = Table(title="Collection Results", box=box.SIMPLE)
    table.add_column("Account")
    table.add_column("Posts", justify="right")
    table.add_column("Peak hour", justify="right")
    table.add_column("Weekend %", justify="right")
    table.add_column("Burst score", justify="right")
    table.add_column("Vocab richness", justify="right")

    for f in results:
        s = f.get("summary", {})
        table.add_row(
            f"{f['platform']}:{f['username']}",
            str(f["post_count"]),
            f"{s.get('peak_hour',0):02d}:00",
            f"{s.get('weekend_ratio',0)*100:.0f}%",
            f"{s.get('burst_score',0):.2f}",
            f"{s.get('vocab_richness',0):.2f}",
        )
    console.print(table)


# ─── analyze ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("account_a")
@click.argument("account_b")
@click.option("--threshold", default=0.75, show_default=True)
@click.option("--model",     default="data/models/best.pt", help="Trained model path")
@click.option("--no-model",  is_flag=True, help="Skip model, use raw feature cosine similarity")
@click.option("--json-out",  is_flag=True, help="Output raw JSON")
def analyze(account_a, account_b, threshold, model, no_model, json_out):
    """
    Compare two accounts and report similarity score with full explanation.

    Example:
      python -m osint analyze hn:pg github:paulgrahm
    """
    from .features import feature_breakdown, cosine_similarity as cos_sim

    platform_a, username_a = parse_account(account_a)
    platform_b, username_b = parse_account(account_b)

    feat_a = load_feat(platform_a, username_a)
    feat_b = load_feat(platform_b, username_b)

    if feat_a is None or feat_b is None:
        sys.exit(1)

    # Score
    model_used = False
    if not no_model and Path(model).exists():
        try:
            from .model import load_model, score_pair
            m = load_model(model)
            score = score_pair(m, feat_a["vector"], feat_b["vector"])
            model_used = True
        except Exception as e:
            console.print(f"[yellow]Model load failed ({e}) — using feature cosine similarity[/yellow]")
            score = cos_sim(feat_a["vector"], feat_b["vector"])
    else:
        score = cos_sim(feat_a["vector"], feat_b["vector"])

    breakdown = feature_breakdown(feat_a, feat_b)

    if json_out:
        out = {
            "account_a": account_a,
            "account_b": account_b,
            "score": round(score, 4),
            "verdict": verdict(score, threshold),
            "model_used": model_used,
            "threshold": threshold,
            "features": breakdown,
            "summary_a": feat_a.get("summary",{}),
            "summary_b": feat_b.get("summary",{}),
        }
        print(json.dumps(out, indent=2))
        return

    # Pretty output
    conf, style = confidence_label(score)
    verd = verdict(score, threshold)
    border = "green" if score >= threshold else "red"

    console.print()
    console.print(Panel(
        f"[bold]{account_a}[/bold]   ↔   [bold]{account_b}[/bold]\n\n"
        f"Similarity score : [{style}]{score:.4f}[/{style}]   (threshold {threshold})\n"
        f"Confidence       : [{style}]{conf}[/{style}]\n"
        f"Verdict          : [bold {border}]{verd}[/bold {border}]\n"
        f"Scored by        : {'trained model' if model_used else 'feature cosine similarity'}",
        title="[bold]Persona Correlation Analysis[/bold]",
        border_style=border,
    ))

    # Feature breakdown
    feat_table = Table(title="Feature Similarity Breakdown", box=box.SIMPLE, show_header=True)
    feat_table.add_column("Feature")
    feat_table.add_column("Score", justify="right")
    feat_table.add_column("Signal")

    for item in breakdown:
        s = item["score"]
        bar = "█" * int(s * 12) + "░" * (12 - int(s * 12))
        sig_col = "green" if item["signal"] == "strong match" else ("yellow" if item["signal"] == "partial match" else "red")
        feat_table.add_row(
            item["feature"],
            f"{s:.3f}  {bar}",
            f"[{sig_col}]{item['signal']}[/{sig_col}]",
        )
    console.print(feat_table)

    # Side-by-side stats
    stats_table = Table(title="Behavioral Profile Comparison", box=box.SIMPLE)
    stats_table.add_column("Metric")
    stats_table.add_column(account_a, justify="right")
    stats_table.add_column(account_b, justify="right")

    sa = feat_a.get("summary", {})
    sb = feat_b.get("summary", {})
    rows = [
        ("Posts analyzed",    str(feat_a["post_count"]),         str(feat_b["post_count"])),
        ("Peak active hour",  f"{sa.get('peak_hour',0):02d}:00", f"{sb.get('peak_hour',0):02d}:00"),
        ("Weekend posting",   f"{sa.get('weekend_ratio',0)*100:.0f}%", f"{sb.get('weekend_ratio',0)*100:.0f}%"),
        ("Burst score",       f"{sa.get('burst_score',0):.2f}",  f"{sb.get('burst_score',0):.2f}"),
        ("Avg sentence len",  f"{sa.get('avg_sent_len',0):.1f} words", f"{sb.get('avg_sent_len',0):.1f} words"),
        ("Vocab richness",    f"{sa.get('vocab_richness',0):.2f}", f"{sb.get('vocab_richness',0):.2f}"),
        ("Ellipsis frequency",f"{sa.get('ellipsis_freq',0):.4f}", f"{sb.get('ellipsis_freq',0):.4f}"),
        ("Reply ratio",       f"{sa.get('reply_ratio',0)*100:.0f}%", f"{sb.get('reply_ratio',0)*100:.0f}%"),
    ]
    for row in rows:
        stats_table.add_row(*row)
    console.print(stats_table)


# ─── search ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("account")
@click.option("--top",       default=10,    show_default=True, help="Number of results")
@click.option("--threshold", default=0.60,  show_default=True, help="Min similarity to show")
@click.option("--model",     default="data/models/best.pt")
@click.option("--platform",  default=None,  help="Restrict results to platform")
def search(account, top, threshold, model, platform):
    """
    Find accounts most similar to ACCOUNT across all collected data.

    Example:
      python -m osint search hn:tptacek --top 10
      python -m osint search hn:tptacek --platform github
    """
    from .pipeline import load_all_features
    from .features import cosine_similarity as cos_sim

    plat, uname = parse_account(account)
    feat = load_feat(plat, uname)
    if feat is None:
        sys.exit(1)

    all_feats = load_all_features()
    query_key = f"{plat}:{uname}"

    # Load model if available
    use_model = False
    model_obj = None
    if Path(model).exists():
        try:
            from .model import load_model, rank_candidates
            model_obj = load_model(model)
            use_model = True
        except Exception:
            pass

    # Score all candidates
    results = []
    for key, cand_feat in all_feats.items():
        if key == query_key:
            continue
        if platform and cand_feat["platform"] != platform:
            continue
        if use_model:
            from .model import score_pair
            score = score_pair(model_obj, feat["vector"], cand_feat["vector"])
        else:
            score = cos_sim(feat["vector"], cand_feat["vector"])
        if score >= threshold:
            results.append((key, score, cand_feat))

    results.sort(key=lambda x: -x[1])

    console.rule(f"[bold]Search results for {account}[/bold]")

    if not results:
        console.print(f"[dim]No accounts found above threshold {threshold}[/dim]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Account")
    table.add_column("Score", justify="right")
    table.add_column("Confidence")
    table.add_column("Posts", justify="right")
    table.add_column("Peak hr", justify="right")

    for rank, (key, score, cf) in enumerate(results[:top], 1):
        conf, style = confidence_label(score)
        s = cf.get("summary", {})
        table.add_row(
            str(rank),
            key,
            f"{score:.4f}",
            f"[{style}]{conf}[/{style}]",
            str(cf["post_count"]),
            f"{s.get('peak_hour',0):02d}:00",
        )

    console.print(table)
    console.print(f"\n[dim]Scored using {'trained model' if use_model else 'feature cosine similarity'}[/dim]")
    console.print(f"[dim]Run [bold]python -m osint analyze {account} <account>[/bold] for full breakdown[/dim]")


# ─── mine-links ──────────────────────────────────────────────────────────────

@cli.command("mine-links")
@click.option("--platforms", default="hn,github", show_default=True)
def mine_links(platforms):
    """
    Crawl collected profiles for cross-platform self-disclosure links.
    These become the positive (same-person) training pairs.
    """
    from .pipeline import load_all_features, append_link
    from .collectors import mine_identity_links

    console.rule("[bold]Mining identity links[/bold]")
    all_feats = load_all_features()
    plat_list = [p.strip() for p in platforms.split(",")]
    new_links  = 0

    for key, feat in all_feats.items():
        plat = feat["platform"]
        if plat not in plat_list:
            continue
        links = mine_identity_links(plat, feat["username"])
        for link in links:
            append_link(link)
            console.print(f"  [green]Link found:[/green] {link['from']} → {link['to']} ({link.get('source','')})")
            new_links += 1

    console.print(f"\n[green]{new_links} new identity links saved[/green]")


# ─── pipeline ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("accounts", nargs=-1, required=True)
@click.option("--max-posts",      default=300, show_default=True)
@click.option("--neg-ratio",      default=10,  show_default=True, help="Random negatives per positive")
@click.option("--hard-neg-ratio", default=5,   show_default=True, help="Hard negatives per positive")
@click.option("--force",          is_flag=True)
def pipeline(accounts, max_posts, neg_ratio, hard_neg_ratio, force):
    """
    Run full data pipeline: collect → extract → pair.

    Example:
      python -m osint pipeline hn:pg github:paulgrahm hn:dang github:dhh
    """
    from .pipeline import run_pipeline

    parsed = []
    for raw in accounts:
        try:
            parsed.append(parse_account(raw))
        except click.BadParameter as e:
            console.print(f"[red]{e}[/red]")

    console.rule("[bold]Full Pipeline[/bold]")
    ok = run_pipeline(
        accounts=parsed,
        max_posts=max_posts,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        force_collect=force,
    )
    if ok:
        console.print("\n[green]Pipeline complete. Run [bold]python -m osint train[/bold] to train the model.[/green]")


# ─── train ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--epochs",     default=50,   show_default=True)
@click.option("--batch-size", default=256,  show_default=True)
@click.option("--lr",         default=3e-4, show_default=True)
@click.option("--margin",     default=0.3,  show_default=True)
@click.option("--patience",   default=8,    show_default=True)
@click.option("--pairs-dir",  default="data/pairs",  show_default=True)
@click.option("--models-dir", default="data/models", show_default=True)
def train(epochs, batch_size, lr, margin, patience, pairs_dir, models_dir):
    """
    Train the Siamese network on collected pairs.
    Requires pipeline to have been run first.
    """
    from .model import train as run_train

    console.rule("[bold]Training Siamese Network[/bold]")
    run_train(
        pairs_dir=pairs_dir,
        models_dir=models_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        patience=patience,
        verbose=True,
    )


# ─── discover ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("platform", type=click.Choice(["hn", "github"]))
@click.option("--n",         default=100,  show_default=True, help="Users to discover")
@click.option("--max-posts", default=200,  show_default=True)
def discover(platform, n, max_posts):
    """
    Auto-discover and collect active users from a platform.

    Example:
      python -m osint discover hn --n 200
      python -m osint discover github --n 100
    """
    from .pipeline import discover_and_collect
    console.rule(f"[bold]Discovering {platform} users[/bold]")
    discover_and_collect(platform, n=n, max_posts=max_posts)


# ─── report ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("account")
@click.option("--json-out", is_flag=True)
def report(account, json_out):
    """
    Print full behavioral profile for an account.

    Example:
      python -m osint report hn:dang
    """
    plat, uname = parse_account(account)
    feat = load_feat(plat, uname)
    if feat is None:
        sys.exit(1)

    if json_out:
        out = {k: v for k, v in feat.items() if k != "vector"}  # Don't dump 484 floats
        out["vector_dim"] = feat["dim"]
        print(json.dumps(out, indent=2))
        return

    s = feat.get("summary", {})
    console.rule(f"[bold]Behavioral Profile: {account}[/bold]")
    console.print(Panel(
        f"Platform    : [bold]{feat['platform']}[/bold]\n"
        f"Username    : [bold]{feat['username']}[/bold]\n"
        f"Posts       : {feat['post_count']}\n"
        f"Feature dim : {feat['dim']}",
        title="Account"
    ))

    table = Table(title="Behavioral Fingerprint", box=box.SIMPLE)
    table.add_column("Feature")
    table.add_column("Value", justify="right")

    rows = [
        ("Peak active hour",    f"{s.get('peak_hour',0):02d}:00"),
        ("Weekend posting ratio", f"{s.get('weekend_ratio',0)*100:.1f}%"),
        ("Burst posting score",  f"{s.get('burst_score',0):.3f}"),
        ("Avg sentence length",  f"{s.get('avg_sent_len',0):.1f} words"),
        ("Vocabulary richness",  f"{s.get('vocab_richness',0):.3f}"),
        ("Ellipsis frequency",   f"{s.get('ellipsis_freq',0):.5f}"),
        ("Reply ratio",          f"{s.get('reply_ratio',0)*100:.1f}%"),
        ("Link sharing rate",    f"{s.get('tech_affinity',0)*100:.1f}%"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
