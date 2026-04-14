# OSINT Behavioral Correlator

Cross-platform pseudonym correlation via behavioral fingerprinting.
No username or email overlap needed — purely behavioral.

---

## Quickstart (3 commands)

```bash
pip install -r requirements.txt
python generate_training_data.py    # optional — regenerates synthetic training data
python -m osint.server              # starts server + opens UI at http://localhost:8000
```

Then open http://localhost:8000 in your browser.

---

## Full structure

```
osint_complete/
├── osint/
│   ├── collectors.py     live data: HN, GitHub, Reddit, YouTube
│   ├── features.py       484-dim behavioral fingerprint extraction
│   ├── model.py          Siamese network + triplet loss training
│   ├── pipeline.py       data management, pair building, hard negatives
│   ├── cli.py            command-line interface
│   ├── server.py         FastAPI REST server (19 endpoints)
│   └── static/
│       └── index.html    browser UI (served automatically)
├── data/
│   └── models/
│       └── best.pt       pre-trained model checkpoint
├── generate_training_data.py   synthetic data generator
├── seed_accounts.jsonl         34 verified cross-platform identity pairs
├── requirements.txt
└── setup.py
```

---

## Environment variables (optional)

Create a `.env` file or export before running:

```bash
export GITHUB_TOKEN=your_token   # free at github.com/settings/tokens — no scopes needed
                                  # raises rate limit from 60 to 5000 req/hr
export YOUTUBE_API_KEY=your_key  # only needed for YouTube collection
export REDDIT_CLIENT_ID=xxx      # only needed for Reddit OAuth
export REDDIT_CLIENT_SECRET=xxx
```

---

## CLI commands

```bash
# Collect accounts
python -m osint collect hn tptacek --max-posts 500
python -m osint collect github tptacek --max-posts 500
python -m osint collect reddit username --max-posts 300

# Compare two accounts
python -m osint analyze hn:tptacek github:tptacek

# Find all matches for an account
python -m osint search hn:tptacek --top 10

# Auto-discover + collect active users
python -m osint discover hn --n 200
python -m osint discover github --n 100

# Mine cross-platform identity links from bios
python -m osint mine-links

# Build training pairs
python -m osint pipeline hn:tptacek github:tptacek hn:pg github:paulgrahm

# Train the model
python -m osint train --epochs 100 --patience 15

# Print behavioral profile
python -m osint report hn:tptacek
```

---

## Training from scratch (real data)

```bash
# 1. Collect seed accounts (~20 min with GitHub token)
python -m osint collect hn tptacek --max-posts 500
python -m osint collect github tptacek --max-posts 500
# ... repeat for all pairs in seed_accounts.jsonl

# 2. Mine identity links
python -m osint mine-links --platforms hn,github

# 3. Expand negative pool
python -m osint discover hn --n 300 --max-posts 200
python -m osint discover github --n 200 --max-posts 200

# 4. Build pairs
python -m osint pipeline hn:tptacek github:tptacek hn:pg github:paulgrahm \
  --neg-ratio 12 --hard-neg-ratio 8

# 5. Train
python -m osint train --epochs 100 --batch-size 128 --lr 0.0003 --patience 15

# 6. Server auto-reloads the new model
curl -X POST http://localhost:8000/api/model/reload
```

---

## API endpoints (when server is running)

| Method | Path | What |
|--------|------|------|
| GET | /api/status | server health, model info |
| POST | /api/collect | collect + fingerprint account |
| GET | /api/accounts | list all collected accounts |
| POST | /api/analyze | compare two accounts |
| POST | /api/search | find matches for account |
| GET | /api/graph | network graph data |
| POST | /api/pipeline | full collect→pair pipeline |
| POST | /api/train | start training (background) |
| GET | /api/train/status | poll training progress |
| POST | /api/model/reload | hot-reload model |

Full interactive docs: http://localhost:8000/docs

---

## How it works

1. **Collect** — fetches post history from HN (Algolia API), GitHub (REST API),
   Reddit (JSON API), YouTube (Data API v3)

2. **Extract** — builds a 484-dim behavioral fingerprint per account:
   - 31-dim temporal histogram (posting hours + weekday distribution)
   - 55-dim stylometric vector (sentence length, vocab richness, 37 function
     word frequencies, punctuation habits, code block usage)
   - 8-dim behavioral vector (burst score, weekend ratio, reply ratio, etc.)
   - 384-dim sentence embedding (mean-pooled all-MiniLM-L6-v2)
   - 6-dim platform one-hot

3. **Score** — cosine similarity on unit-sphere embeddings from the trained
   Siamese network. Same person → score near 1.0. Different people → score
   near 0.0. Threshold 0.75 by default.

4. **Explain** — per-feature breakdown shows which behavioral signals
   match or mismatch between two accounts.
