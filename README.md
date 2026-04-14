# OSINT Behavioral Correlator

Siamese neural network for cross-platform persona analysis: estimates whether two accounts on different platforms belong to the same person or not

---

## how it works

1) **collect** — fetches post history from public APIs
2) **fingerprint** — extracts a 486-dim behavioral vector per account:
   - 31-dim temporal histogram (posting hours, weekday patterns)
   - 55-dim stylometric vector (sentence length, vocabulary richness, function word frequencies, punctuation habits)
   - 8-dim behavioral vector (burst score, weekend ratio, reply ratio, topic diversity)
   - 384-dim semantic embedding (mean-pooled `all-MiniLM-L6-v2`)
   - 8-dim learned platform embedding
3) **score** — a trained Siamese network maps accounts to a unit sphere; cosine similarity between embeddings indicates identity match probability
4) **explain** — per-feature breakdown shows which behavioral signals match or mismatch

## model performance

trained on 164 verified cross-platform identity groups (HN ↔ GitHub):

| Metric | Value |
|--------|-------|
| embedding gap (same vs. different person) | 0.49 |
| F1 @ threshold 0.70 | 0.28 |

---

## quickstart

```bash
pip install -r requirements.txt
export GITHUB_TOKEN=your_token   # optional but recommended
python -m osint.server           # starts server at http://localhost:8000
```

a pre-trained model checkpoint is included at `data/models/best.pt`.

---

## CLI

```bash
# collect accounts
python -m osint collect hn:username github:username --max-posts 500

# compare two accounts
python -m osint analyze hn:simonw github:simonw

# find best matches for an account across all collected accounts
python -m osint search hn:simonw --top 10

# build training pairs and train
python -m osint pipeline hn:user1 github:user1 hn:user2 github:user2
python -m osint train --epochs 150 --batch-size 64 --lr 0.0001 --patience 25
```

---

## project structure

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

## environment variables

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
| POST | /api/collect | collect and fingerprint an account |
| POST | /api/analyze | compare two accounts |
| POST | /api/search | find matches for an account |
| GET | /api/accounts | list all collected accounts |
| GET | /api/graph | network graph data |
| POST | /api/train | start training run |
| POST | /api/model/reload | hot-reload model checkpoint |

---
forthcoming modifications include a repaired network graph system and a new best.pt trained on Chinese and RuNet sites, as well as other languages between X and GitHub, e.g.

---
<img width="1720" height="898" alt="Screenshot 2026-04-13 at 6 32 32 PM" src="https://github.com/user-attachments/assets/34865966-84a2-4900-a621-993b1a2e5880" />
<img width="1721" height="899" alt="Screenshot 2026-04-13 at 6 32 10 PM" src="https://github.com/user-attachments/assets/92a08b60-c9c6-4478-b17d-dedc81dabe9a" />
