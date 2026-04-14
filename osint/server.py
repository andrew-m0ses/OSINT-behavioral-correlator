"""
osint/server.py
FastAPI REST server — bridges the frontend UI to the Python backend.

Run:
    python -m osint.server
    # or
    uvicorn osint.server:app --reload --port 8000

Endpoints:
    GET  /api/status                     — server health + model info
    POST /api/collect                    — collect + fingerprint an account
    GET  /api/accounts                   — list all collected accounts
    GET  /api/accounts/{platform}/{user} — get one account's fingerprint
    DELETE /api/accounts/{platform}/{user}
    POST /api/analyze                    — compare two accounts
    POST /api/search                     — find matches for an account
    POST /api/mine-links                 — mine cross-platform identity links
    GET  /api/links                      — list known identity links
    POST /api/pipeline                   — run full collect→extract→pair
    POST /api/train                      — kick off model training (async)
    GET  /api/train/status               — training job status
    GET  /api/graph                      — graph data for all accounts
    GET  /api/export/{platform}/{user}   — export account report as JSON
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Import our modules ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from osint.collectors import collect_account, mine_identity_links
from osint.features import extract as extract_features, feature_breakdown, cosine_similarity
from osint.pipeline import (
    save_raw, load_raw, save_features, load_features,
    load_all_features, append_link, load_links,
    build_identity_groups, build_positive_pairs,
    build_random_negatives, build_hard_negatives,
    identity_split, save_pairs, MIN_POSTS,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OSINT Behavioral Correlator API",
    version="1.0.0",
    description="Cross-platform pseudonym correlation via behavioral fingerprinting",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_model      = None          # Loaded SiameseCorrelator, or None
_model_path = "data/models/best.pt"
_model_meta = {}

_train_job = {
    "running": False,
    "started_at": None,
    "epoch": 0,
    "total_epochs": 0,
    "best_f1": 0.0,
    "log": [],
    "finished": False,
    "error": None,
}


def _try_load_model():
    global _model, _model_meta
    if not Path(_model_path).exists():
        return
    try:
        from osint.model import load_model
        _model = load_model(_model_path)
        ckpt = __import__("torch").load(_model_path, map_location="cpu")
        _model_meta = ckpt.get("metrics", {})
        _model_meta["loaded_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[server] Model loaded from {_model_path}")
    except Exception as e:
        print(f"[server] Could not load model: {e}")


# Load model at startup if it exists
_try_load_model()


# ── Request / Response models ─────────────────────────────────────────────────

class CollectRequest(BaseModel):
    platform: str
    username: str
    max_posts: int = 300
    force: bool = False

class AnalyzeRequest(BaseModel):
    account_a: str          # "platform:username"
    account_b: str
    threshold: float = 0.75

class SearchRequest(BaseModel):
    account: str            # "platform:username"
    platform_filter: Optional[str] = None
    min_score: float = 0.55
    top_k: int = 20

class PipelineRequest(BaseModel):
    accounts: list[str]     # ["hn:pg", "github:dhh", ...]
    max_posts: int = 300
    neg_ratio: int = 10
    hard_neg_ratio: int = 5
    force_collect: bool = False

class TrainRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    margin: float = 0.3
    patience: int = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_key(key: str) -> tuple[str, str]:
    if ":" not in key:
        raise HTTPException(400, f"Invalid account key '{key}'. Use platform:username format.")
    parts = key.split(":", 1)
    return parts[0].strip().lower(), parts[1].strip()


def _score_pair(feat_a: dict, feat_b: dict) -> float:
    if _model is not None:
        try:
            from osint.model import score_pair
            return score_pair(_model, feat_a["vector"], feat_b["vector"])
        except Exception:
            pass
    return cosine_similarity(feat_a["vector"], feat_b["vector"])


def _feature_summary(feat: dict) -> dict:
    """Slim version of feature dict safe to return to client."""
    return {
        "platform":   feat["platform"],
        "username":   feat["username"],
        "post_count": feat["post_count"],
        "dim":        feat["dim"],
        "summary":    feat.get("summary", {}),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    all_feats = load_all_features()
    links     = load_links()
    return {
        "status":        "ok",
        "accounts":      len(all_feats),
        "identity_links":len(links),
        "model": {
            "loaded":    _model is not None,
            "path":      _model_path if Path(_model_path).exists() else None,
            "metrics":   _model_meta,
            "mode":      "siamese_model" if _model else "feature_cosine",
        },
        "training": {
            "running":   _train_job["running"],
            "finished":  _train_job["finished"],
        },
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/collect")
def collect(req: CollectRequest):
    """
    Collect posts for an account, extract behavioral features, cache to disk.
    Returns the behavioral fingerprint summary.
    """
    platform = req.platform.lower()
    username = req.username.strip()

    if not username:
        raise HTTPException(400, "username is required")

    # Check cache first (unless force=True)
    if not req.force:
        feat = load_features(platform, username)
        if feat:
            return {
                "cached":    True,
                "account":   f"{platform}:{username}",
                **_feature_summary(feat),
                "links_found": [],
            }

    # Live collection
    data = collect_account(platform, username, max_posts=req.max_posts)
    if data is None:
        raise HTTPException(502, f"Collection failed for {platform}:{username}")
    if data["post_count"] < MIN_POSTS:
        raise HTTPException(422, f"Too few posts ({data['post_count']}) for {platform}:{username} — minimum {MIN_POSTS}")

    # Extract features
    feat = extract_features(data, use_embeddings=True)
    save_raw(data)
    save_features(feat)

    # Mine cross-platform links
    links = mine_identity_links(platform, username)
    for link in links:
        append_link(link)

    return {
        "cached":      False,
        "account":     f"{platform}:{username}",
        "links_found": links,
        **_feature_summary(feat),
    }


@app.get("/api/accounts")
def list_accounts(platform: Optional[str] = None):
    """List all collected accounts with their behavioral summaries."""
    all_feats = load_all_features()
    result = []
    for key, feat in all_feats.items():
        if platform and feat["platform"] != platform:
            continue
        result.append({
            "key":     key,
            **_feature_summary(feat),
        })
    result.sort(key=lambda x: x["key"])
    return {"accounts": result, "count": len(result)}


@app.get("/api/accounts/{platform}/{username}")
def get_account(platform: str, username: str):
    """Get full fingerprint for one account."""
    feat = load_features(platform, username)
    if feat is None:
        raise HTTPException(404, f"{platform}:{username} not found — collect it first")
    # Return everything except the raw 484-dim vector (too big for UI)
    out = {k: v for k, v in feat.items() if k != "vector"}
    out["vector_dim"] = feat["dim"]
    return out


@app.delete("/api/accounts/{platform}/{username}")
def delete_account(platform: str, username: str):
    """Remove a collected account from cache."""
    raw_path  = Path("data/raw")      / platform / f"{username}.json"
    feat_path = Path("data/features") / platform / f"{username}.json"
    deleted = []
    for p in [raw_path, feat_path]:
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if not deleted:
        raise HTTPException(404, f"{platform}:{username} not found")
    return {"deleted": deleted}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Compare two accounts. Returns similarity score, verdict, and
    per-feature breakdown for explainability.
    """
    plat_a, user_a = _parse_key(req.account_a)
    plat_b, user_b = _parse_key(req.account_b)

    feat_a = load_features(plat_a, user_a)
    feat_b = load_features(plat_b, user_b)

    if feat_a is None:
        raise HTTPException(404, f"{req.account_a} not found — collect it first")
    if feat_b is None:
        raise HTTPException(404, f"{req.account_b} not found — collect it first")

    score   = round(_score_pair(feat_a, feat_b), 6)
    breakdown = feature_breakdown(feat_a, feat_b)

    if score >= req.threshold:
        verdict, confidence = "LIKELY SAME PERSON", "high" if score >= 0.88 else "medium"
    elif score >= 0.55:
        verdict, confidence = "INCONCLUSIVE", "low"
    else:
        verdict, confidence = "LIKELY DIFFERENT PEOPLE", "high"

    return {
        "account_a":  req.account_a,
        "account_b":  req.account_b,
        "score":      score,
        "threshold":  req.threshold,
        "verdict":    verdict,
        "confidence": confidence,
        "model_used": _model is not None,
        "features":   breakdown,
        "summary_a":  feat_a.get("summary", {}),
        "summary_b":  feat_b.get("summary", {}),
        "post_count_a": feat_a["post_count"],
        "post_count_b": feat_b["post_count"],
    }


@app.post("/api/search")
def search(req: SearchRequest):
    """
    Find all accounts most similar to the query account.
    Returns ranked list with scores.
    """
    plat, user = _parse_key(req.account)
    feat_q = load_features(plat, user)
    if feat_q is None:
        raise HTTPException(404, f"{req.account} not found — collect it first")

    all_feats = load_all_features()
    results   = []

    for key, feat in all_feats.items():
        if key == req.account:
            continue
        if req.platform_filter and feat["platform"] != req.platform_filter:
            continue
        score = _score_pair(feat_q, feat)
        if score >= req.min_score:
            results.append({
                "key":        key,
                "score":      round(score, 6),
                "confidence": "high" if score >= 0.78 else "medium" if score >= 0.65 else "low",
                **_feature_summary(feat),
            })

    results.sort(key=lambda x: -x["score"])
    return {
        "query":      req.account,
        "results":    results[:req.top_k],
        "total_found":len(results),
        "model_used": _model is not None,
    }


@app.post("/api/mine-links")
def mine_links_endpoint(platforms: list[str] = None):
    """
    Crawl all collected account profiles for cross-platform self-disclosures.
    Appends found links to data/identity_links.jsonl.
    """
    all_feats  = load_all_features()
    plat_filter = set(platforms) if platforms else None
    new_links  = []

    for key, feat in all_feats.items():
        if plat_filter and feat["platform"] not in plat_filter:
            continue
        links = mine_identity_links(feat["platform"], feat["username"])
        for link in links:
            append_link(link)
            new_links.append(link)

    return {
        "new_links":   new_links,
        "total_found": len(new_links),
        "total_links": len(load_links()),
    }


@app.get("/api/links")
def get_links():
    """Return all known cross-platform identity links."""
    return {"links": load_links(), "count": len(load_links())}


@app.post("/api/pipeline")
async def run_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Full collect → extract → pair pipeline for a list of accounts.
    Runs synchronously (use background_tasks for large runs).
    """
    results = {"collected": [], "failed": [], "pairs_built": False}

    for raw in req.accounts:
        try:
            plat, user = _parse_key(raw)
        except HTTPException:
            results["failed"].append({"account": raw, "reason": "invalid format"})
            continue

        if not req.force_collect:
            feat = load_features(plat, user)
            if feat:
                results["collected"].append({"account": raw, "cached": True, "post_count": feat["post_count"]})
                continue

        data = collect_account(plat, user, max_posts=req.max_posts)
        if data is None or data["post_count"] < MIN_POSTS:
            results["failed"].append({"account": raw, "reason": f"only {data['post_count'] if data else 0} posts"})
            continue

        feat = extract_features(data, use_embeddings=True)
        save_raw(data)
        save_features(feat)
        links = mine_identity_links(plat, user)
        for link in links:
            append_link(link)
        results["collected"].append({
            "account": raw, "cached": False,
            "post_count": feat["post_count"],
            "links_found": len(links),
        })

    # Build pairs
    all_feats = load_all_features()
    if len(all_feats) >= 2:
        groups     = build_identity_groups(all_feats)
        all_keys   = list(all_feats.keys())
        pos_pairs  = build_positive_pairs(groups)
        rand_negs  = build_random_negatives(groups, all_keys, n=max(len(pos_pairs)*req.neg_ratio, 500))
        hard_negs  = build_hard_negatives(all_feats, groups, n=max(len(pos_pairs)*req.hard_neg_ratio, 200))
        split      = identity_split(groups, pos_pairs, rand_negs + hard_negs)
        save_pairs(split, all_feats)
        results["pairs_built"] = True
        results["pair_counts"]  = {k: len(v) for k, v in split.items()}
        results["identity_groups"] = len(groups)

    return results


@app.post("/api/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model training in a background thread.
    Poll /api/train/status for progress.
    """
    global _train_job
    if _train_job["running"]:
        raise HTTPException(409, "Training already in progress")

    pairs_dir = Path("data/pairs")
    if not (pairs_dir / "train" / "X_a.npy").exists():
        raise HTTPException(422, "No training pairs found. Run /api/pipeline first.")

    _train_job = {
        "running": True, "started_at": datetime.now(timezone.utc).isoformat(),
        "epoch": 0, "total_epochs": req.epochs, "best_f1": 0.0,
        "log": [], "finished": False, "error": None,
    }

    def _do_train():
        global _model, _model_meta, _train_job
        try:
            from osint.model import train as run_train

            class _Logger:
                """Capture print output into _train_job log."""
                def write(self, msg):
                    msg = msg.strip()
                    if msg:
                        _train_job["log"].append(msg)
                        if "Epoch" in msg:
                            parts = msg.split("|")
                            try:
                                ep = int(parts[0].split()[1].split("/")[0])
                                _train_job["epoch"] = ep
                                f1_part = [p for p in parts if "f1=" in p]
                                if f1_part:
                                    f1 = float(f1_part[0].split("=")[1].strip())
                                    _train_job["best_f1"] = max(_train_job["best_f1"], f1)
                            except Exception:
                                pass
                def flush(self): pass

            old_stdout = sys.stdout
            sys.stdout = _Logger()
            try:
                run_train(
                    pairs_dir="data/pairs",
                    models_dir="data/models",
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    lr=req.lr,
                    margin=req.margin,
                    patience=req.patience,
                    verbose=True,
                )
            finally:
                sys.stdout = old_stdout

            # Reload model
            _try_load_model()
            _train_job["finished"] = True
            _train_job["running"]  = False
            _train_job["log"].append("Training complete.")

        except Exception as e:
            _train_job["error"]    = str(e)
            _train_job["running"]  = False
            _train_job["finished"] = True
            _train_job["log"].append(f"ERROR: {e}")

    thread = threading.Thread(target=_do_train, daemon=True)
    thread.start()

    return {
        "started":      True,
        "total_epochs": req.epochs,
        "message":      "Training started. Poll /api/train/status for progress.",
    }


@app.get("/api/train/status")
def training_status():
    """Poll for training progress."""
    return {
        "running":      _train_job["running"],
        "finished":     _train_job["finished"],
        "started_at":   _train_job["started_at"],
        "epoch":        _train_job["epoch"],
        "total_epochs": _train_job["total_epochs"],
        "best_f1":      round(_train_job["best_f1"], 4),
        "error":        _train_job["error"],
        "recent_log":   _train_job["log"][-20:],
        "model_loaded": _model is not None,
    }


@app.post("/api/model/reload")
def reload_model():
    """Reload the model from disk (e.g. after training finishes)."""
    _try_load_model()
    return {
        "loaded":  _model is not None,
        "path":    _model_path,
        "metrics": _model_meta,
    }


@app.get("/api/export/{platform}/{username}")
def export_account(platform: str, username: str):
    """Export full account report as JSON."""
    feat = load_features(platform, username)
    raw  = load_raw(platform, username)
    if feat is None:
        raise HTTPException(404, f"{platform}:{username} not found")

    report = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "account": f"{platform}:{username}",
        "fingerprint": {k: v for k, v in feat.items() if k != "vector"},
        "vector_dim": feat["dim"],
        "profile": raw.get("profile", {}) if raw else {},
        "sample_posts": (raw.get("posts", [])[:5] if raw else []),
    }
    return report


# ── Serve static frontend ─────────────────────────────────────────────────────
# Put your built frontend in osint/static/
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ── Dev entrypoint ────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "osint.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
