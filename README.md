# OSINT Behavioral Correlator

Cross-platform pseudonym correlation via behavioral fingerprinting. Identifies whether two accounts on different platforms belong to the same person — no username or email overlap required, purely behavioral.

Built as a research tool for studying online identity and influence operations.

---

## How it works

1. **Collect** — fetches post history from Hacker News, GitHub, Reddit, and more via public APIs
2. **Fingerprint** — extracts a 486-dim behavioral vector per account:
   - 31-dim temporal histogram (posting hours, weekday patterns)
   - 55-dim stylometric vector (sentence length, vocabulary richness, function word frequencies, punctuation habits)
   - 8-dim behavioral vector (burst score, weekend ratio, reply ratio, topic diversity)
   - 384-dim semantic embedding (mean-pooled `all-MiniLM-L6-v2`)
   - 8-dim learned platform embedding
3. **Score** — a trained Siamese network maps accounts to a unit sphere; cosine similarity between embeddings indicates identity match probability
4. **Explain** — per-feature breakdown shows which behavioral signals match or mismatch

## Model performance

Trained on 164 verified cross-platform identity groups (HN ↔ GitHub):

| Metric | Value |
|--------|-------|
| Embedding gap (same vs. different person) | 0.49 |
| F1 @ threshold 0.70 | 0.28 |
| Known pairs correctly identified | simonw: 0.90, geerlingguy: 0.82, minimaxir: 0.76 |
| Known non-match correctly rejected | simonw vs tptacek: 0.43 |

---

## Quickstart

```bash
pip install -r requirements.txt
export GITHUB_TOKEN=your_token   # optional but recommended
python -m osint.server           # starts server at http://localhost:8000
```

A pre-trained model checkpoint is included at `data/models/best.pt`.

---

## CLI

```bash
# Collect accounts
python -m osint collect hn:username github:username --max-posts 500

# Compare two accounts
python -m osint analyze hn:simonw github:simonw

# Find best matches for an account across all collected accounts
python -m osint search hn:simonw --top 10

# Build training pairs and train
python -m osint pipeline hn:user1 github:user1 hn:user2 github:user2
python -m osint train --epochs 150 --batch-size 64 --lr 0.0001 --patience 25
```

---

## Project structure

```
osint/
├── collectors.py     HN, GitHub, Reddit, V2EX, Habr, Twitter collectors
├── features.py       486-dim behavioral fingerprint extraction
├── model.py          Siamese network + triplet loss
├── pipeline.py       pair building, hard negative mining, balanced sampling
├── cli.py            command-line interface
├── server.py         FastAPI REST server
└── static/
    └── index.html    browser UI

find_pairs*.py        pair discovery scripts for each platform
data/
├── models/best.pt    trained model checkpoint
└── identity_links.jsonl  verified cross-platform identity links
```

---

## Environment variables

```bash
export GITHUB_TOKEN=your_token        # raises GitHub rate limit 60 → 5000 req/hr
export APIFY_TOKEN=your_token         # for Twitter/X collection via Apify
export TWITTER_AUTH_TOKEN=your_token  # Twitter browser cookie
export TWITTER_CT0=your_token         # Twitter browser cookie
```

---

## API

The FastAPI server exposes endpoints for collection, analysis, search, training, and graph visualization. Interactive docs at `http://localhost:8000/docs` when server is running.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/collect | Collect and fingerprint an account |
| POST | /api/analyze | Compare two accounts |
| POST | /api/search | Find matches for an account |
| GET | /api/accounts | List all collected accounts |
| GET | /api/graph | Network graph data |
| POST | /api/train | Start training run |
| POST | /api/model/reload | Hot-reload model checkpoint |

---

## Ethical use

This tool is designed for academic research, journalism, and security research into online influence operations and sockpuppet detection. It works on publicly available data only.

It should not be used for stalking, harassment, or surveillance of private individuals.
