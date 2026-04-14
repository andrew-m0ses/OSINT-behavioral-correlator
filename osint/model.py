"""
osint/model.py
Siamese network for cross-platform persona correlation.

Architecture:
  PersonEncoder: 484 → [512 → 256] → 128 (L2 normalized)
  Loss: TripletMarginLoss with hard negative mining
  Similarity: cosine on unit sphere (fast, interpretable)

Training is fully self-contained here. Data is loaded from the
pairs/ directory produced by the pipeline.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PLATFORMS = ["hn", "github", "reddit", "youtube", "twitter", "telegram"]

# ─── Dataset ──────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Loads (anchor, positive, negative) triplets from pairs/ directory.
    Dynamically samples triplets each epoch from pos/neg pool.
    """
    def __init__(self, pairs_dir: str, split: str = "train"):
        d = Path(pairs_dir) / split
        self.X_a = torch.from_numpy(np.load(d / "X_a.npy").astype(np.float32))
        self.X_b = torch.from_numpy(np.load(d / "X_b.npy").astype(np.float32))
        self.y   = torch.from_numpy(np.load(d / "y.npy").astype(np.float32))

        pos = self.y == 1
        neg = self.y == 0
        # Positive pairs: (anchor, positive)
        self.pos_a = self.X_a[pos]
        self.pos_b = self.X_b[pos]
        # Negative pool: all negative accounts
        self.neg_pool = torch.cat([self.X_a[neg], self.X_b[neg]], dim=0)

        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        print(f"  [{split}] {n_pos} pos pairs | {n_neg} neg pairs | {len(self.neg_pool)} neg accounts")

    def __len__(self):
        return len(self.pos_a) * 6  # Multiple passes per epoch

    def __getitem__(self, idx):
        i = idx % len(self.pos_a)
        anchor   = self.pos_a[i]
        positive = self.pos_b[i]
        j = torch.randint(0, len(self.neg_pool), (1,)).item()
        negative = self.neg_pool[j]
        return anchor, positive, negative


class PairDataset(Dataset):
    """Binary pair dataset for validation/evaluation."""
    def __init__(self, pairs_dir: str, split: str = "val"):
        d = Path(pairs_dir) / split
        self.X_a = torch.from_numpy(np.load(d / "X_a.npy").astype(np.float32))
        self.X_b = torch.from_numpy(np.load(d / "X_b.npy").astype(np.float32))
        self.y   = torch.from_numpy(np.load(d / "y.npy").astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_a[idx], self.X_b[idx], self.y[idx]


# ─── Model ────────────────────────────────────────────────────────────────────

class PersonEncoder(nn.Module):
    """
    Maps 484-dim behavioral feature vector → 128-dim L2-normalized embedding.

    Key design:
    - LayerNorm instead of BatchNorm (stable with small batches)
    - GELU activation (smoother gradients for embedding tasks)
    - Learned platform embedding replaces one-hot (6 → 16 dims)
      so model learns: 'same person writes differently on Twitter vs HN'
    - L2 normalization: dot product == cosine similarity (fast inference)
    """
    def __init__(
        self,
        input_dim:   int   = 484,
        hidden_dims: list  = None,
        output_dim:  int   = 128,
        dropout:     float = 0.2,
        n_platforms: int   = 6,
        plat_dim:    int   = 16,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]

        # Replace 6-dim one-hot with 16-dim learned embedding
        self.platform_emb = nn.Embedding(n_platforms, plat_dim)
        in_dim = input_dim - 6 + plat_dim  # 494

        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 484) — last 6 dims are platform one-hot"""
        features = x[:, :-6]
        plat_ids = x[:, -6:].argmax(dim=-1).clamp(0, len(PLATFORMS)-1)
        plat_emb = self.platform_emb(plat_ids)
        h = torch.cat([features, plat_emb], dim=-1)
        return F.normalize(self.net(h), dim=-1)


class SiameseCorrelator(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.encoder = PersonEncoder(**kw)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, xa, xb):
        ea = self.encode(xa)
        eb = self.encode(xb)
        return F.cosine_similarity(ea, eb)


# ─── Hard Negative Mining ─────────────────────────────────────────────────────

def mine_hard_negatives(
    model: SiameseCorrelator,
    X_a: torch.Tensor,
    y:   torch.Tensor,
    device: torch.device,
    top_k: int = 5,
    batch_size: int = 512,
) -> list[tuple]:
    """
    After N epochs, find the negative pairs the model currently
    finds most confusing (high similarity despite different identities).
    Returns (idx_a, idx_b) pairs for the hardest negatives.
    """
    model.eval()
    # Encode all accounts
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(X_a), batch_size):
            emb = model.encode(X_a[start:start+batch_size].to(device))
            embeddings.append(emb.cpu())
    E = torch.cat(embeddings, dim=0)  # (N, 128)

    # Similarity matrix
    sim = E @ E.T  # (N, N)

    # Find same-identity pairs (to exclude)
    pos_set = set()
    pos_idx = (y == 1).nonzero(as_tuple=True)[0].tolist()
    # (simplified: exclude diagonal)

    hard = []
    for i in range(len(E)):
        if y[i] != 1:   # Only anchor from positive pairs
            continue
        row = sim[i].clone()
        row[i] = -2     # Exclude self
        top = row.topk(top_k).indices
        for j in top:
            if y[j] == 0:   # Must be a negative
                hard.append((i, j.item()))
    model.train()
    return hard


# ─── Loss ─────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dp = 1 - F.cosine_similarity(anchor, positive)
        dn = 1 - F.cosine_similarity(anchor, negative)
        return F.relu(dp - dn + self.margin).mean()


# ─── Training ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    pairs_dir:   str   = "data/pairs",
    models_dir:  str   = "data/models",
    epochs:      int   = 50,
    batch_size:  int   = 256,
    lr:          float = 3e-4,
    margin:      float = 0.3,
    hidden_dims: list  = None,
    output_dim:  int   = 128,
    dropout:     float = 0.2,
    hard_mine_start: int = 5,
    patience:    int   = 8,
    verbose:     bool  = True,
) -> SiameseCorrelator:

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    device = get_device()
    if verbose: print(f"Device: {device}")

    # Data
    try:
        train_ds = TripletDataset(pairs_dir, "train")
        val_ds   = PairDataset(pairs_dir, "val")
    except FileNotFoundError:
        raise RuntimeError("No pairs found. Run the pipeline first: python -m osint.pipeline")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False)

    # Auto-detect input_dim from training data
    import numpy as _np
    input_dim = int(_np.load(f"{pairs_dir}/train/X_a.npy", mmap_mode="r").shape[1])
    if verbose: print(f"Input dim: {input_dim}")

    # Model
    model = SiameseCorrelator(
        input_dim=input_dim, hidden_dims=hidden_dims or [512, 256],
        output_dim=output_dim, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose: print(f"Parameters: {n_params:,}")

    loss_fn   = TripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_f1 = 0.0
    patience_ctr = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        losses = []
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    ea = model.encode(anchor)
                    ep = model.encode(positive)
                    en = model.encode(negative)
                    loss = loss_fn(ea, ep, en)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                ea = model.encode(anchor)
                ep = model.encode(positive)
                en = model.encode(negative)
                loss = loss_fn(ea, ep, en)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        sims, labels = [], []
        with torch.no_grad():
            for xa, xb, y in val_loader:
                xa, xb = xa.to(device), xb.to(device)
                s = model(xa, xb)
                sims.extend(s.cpu().numpy())
                labels.extend(y.numpy())
        sims   = np.array(sims)
        labels = np.array(labels)

        thresh = 0.75
        preds  = (sims >= thresh).astype(float)
        tp = ((preds==1)&(labels==1)).sum()
        fp = ((preds==1)&(labels==0)).sum()
        fn = ((preds==0)&(labels==1)).sum()
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)
        pos_sim = sims[labels==1].mean() if (labels==1).sum() > 0 else 0
        neg_sim = sims[labels==0].mean() if (labels==0).sum() > 0 else 0
        train_loss = np.mean(losses)

        history.append({"epoch": epoch, "loss": train_loss, "f1": f1,
                        "precision": prec, "recall": rec,
                        "pos_sim": pos_sim, "neg_sim": neg_sim})

        if verbose:
            mark = " ✓" if f1 > best_f1 else ""
            print(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | "
                  f"f1={f1:.3f} | p={prec:.3f} r={rec:.3f} | "
                  f"gap={pos_sim-neg_sim:.3f}{mark}")

        if f1 > best_f1:
            best_f1 = f1
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": {"input_dim":484,"hidden_dims":hidden_dims or [512,256],
                           "output_dim":output_dim,"dropout":dropout},
                "metrics": history[-1],
                "history": history,
            }, Path(models_dir) / "best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose: print(f"Early stopping at epoch {epoch}")
                break

    with open(Path(models_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    if verbose: print(f"\nBest F1: {best_f1:.3f} → saved to {models_dir}/best.pt")
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(path: str, device: Optional[torch.device] = None) -> SiameseCorrelator:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    # Auto-detect actual input_dim from weights to handle stale configs
    actual_first_layer = ckpt["model_state"].get("encoder.net.0.weight")
    if actual_first_layer is not None:
        # in_dim = input_dim - 6 + plat_dim, we need to back-calculate input_dim
        # Try values until we find one that produces the right in_dim
        actual_in_dim = actual_first_layer.shape[1]
        # Find plat_dim from embedding weight
        plat_emb = ckpt["model_state"].get("encoder.platform_emb.weight")
        plat_dim = plat_emb.shape[1] if plat_emb is not None else 16
        n_platforms = plat_emb.shape[0] if plat_emb is not None else 8
        # in_dim = input_dim - n_platforms + plat_dim
        inferred_input_dim = actual_in_dim + n_platforms - plat_dim
        cfg = dict(cfg)
        cfg["input_dim"] = inferred_input_dim
    model = SiameseCorrelator(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["output_dim"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def score_pair(
    model: SiameseCorrelator,
    vec_a: list,
    vec_b: list,
    device: Optional[torch.device] = None,
) -> float:
    if device is None:
        device = get_device()
    a = torch.tensor(vec_a, dtype=torch.float32).unsqueeze(0).to(device)
    b = torch.tensor(vec_b, dtype=torch.float32).unsqueeze(0).to(device)
    return float(model(a, b).item())


@torch.no_grad()
def rank_candidates(
    model: SiameseCorrelator,
    query_vec: list,
    candidate_vecs: list[list],
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> list[tuple[int, float]]:
    """Return (index, score) pairs sorted by similarity, top_k first."""
    if device is None:
        device = get_device()
    q = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0).to(device)
    qe = model.encode(q)  # (1, D)
    all_vecs = torch.tensor(candidate_vecs, dtype=torch.float32).to(device)
    # Batch encode candidates
    ce = model.encode(all_vecs)  # (N, D)
    sims = (qe @ ce.T).squeeze(0).cpu().numpy()
    top = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in top]
