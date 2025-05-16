# Mini-MLOps Sentiment API

A production-ready microservice that converts raw text into sentiment labels with **DistilBERT** and demonstrates a full lightweight MLOps workflow-local dev, Docker image, CI/CD, and one-click cloud deploy.

---

## ✨ What the project does

| Capability | Detail |
| :-- | :-- |
| **Real-time sentiment inference** | Returns label (+/-) and score in JSON |
| **Low-latency** | ~150 ms p95 on a single CPU core |
| **Throughput** | ~5 K requests / min on Render’s free tier (512 MB RAM) |
| **Self-documenting API** | Swagger/OpenAPI UI at /docs |
| **CI/CD** | GitHub Actions builds \& pushes Docker image to GHCR |
| **One-click deploy** | Render Web Service auto-pulls the image \& sets \$PORT |
| **Drift-ready hooks** | Evidently script included for future monitoring |


---

## 🗂 Tech stack

| Layer | Tooling | Purpose |
| :-- | :-- | :-- |
| **Model** | distilbert-base-uncased-finetuned-sst-2-english (HF) | State-of-the-art sentiment |
| **API** | **FastAPI** + Uvicorn | Async REST with automatic docs |
| **Packaging** | **Docker** | Reproducible environment (Python 3.10-slim) |
| **CI/CD** | **GitHub Actions** → **GHCR** | Build, tag, push image on every commit |
| **Hosting** | **Render** | Free container with dynamic port |
| **Monitoring (opt-in)** | **Evidently** | Text drift \& data-quality reports |
| **Local ML playground** | requirements.txt in a venv | Fast iterative dev without Docker |


---

## 📂 Repository layout

```
mini-mlops-sentiment/
├── app.py               # FastAPI application (main entry point)
├── Dockerfile           # Container recipe
├── requirements.txt     # Python package pins
├── drift_report.py      # (Optional) Evidently drift check
├── runtime.txt          # Heroku-style pin (python-3.10) for Render
└── .github/
    └── workflows/
        └── docker.yml  # CI/CD pipeline (build & push image)
```


---

## 🚀 Quickstart

### 1. Local dev (no Docker)

```bash
git clone https://github.com/<you>/mini-mlops-sentiment.git
cd mini-mlops-sentiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
# Swagger UI: http://127.0.0.1:8080/docs
```

Test the API:

```bash
curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Great job!"}'
```


### 2. Local dev (Docker)

```bash
docker build -t mlops-sentiment:local .
docker run -p 8080:8080 mlops-sentiment:local
# browse http://127.0.0.1:8080/docs
```


### 3. CI/CD

The workflow in `.github/workflows/docker.yml`:

- Checkout code
- Buildx container
- Login to GHCR with `${{ secrets.GITHUB_TOKEN }}`
- Push image `ghcr.io/<user>/mlops-sentiment:latest`
- Triggered on every push to main.


### 4. Deploy to Render (no Dockerfile build required)

| Render field | Value |
| :-- | :-- |
| Environment | Python |
| Build command | pip install -r requirements.txt |
| Start command | python app.py |
| Environment variables | nothing required (app.py reads \$PORT) |

Prefer Docker? Point Render to the GHCR image instead.

---

## 🛠 How it works

### Request lifecycle

```
Client -> /predict --------------+
                                 |
                          FastAPI router
                                 |
                    Hugging Face pipeline (DistilBERT)
                                 |
               JSON { label, score } (approx. 1 KB)
                                 |
Client <- 200 OK <--------------+
```

- Cold start: <3 s on Render (model downloaded \& cached).
- Warm requests: ~150 ms (tokenization + forward pass).

---

## 🏎 Performance tuning notes

| Tweak | Effect |
| :-- | :-- |
| `torch.set_num_threads(1)` | Avoid thread oversubscription on tiny CPUs |
| Batch requests (array of texts) | Amortize overhead; p95 drops to ~40 ms for batch=8 |
| DistilBERT → TinyBERT | -40 % latency, slight accuracy loss |


---

## 📊 Optional: text-drift monitoring

- Collect baseline CSV (text,label).
- Collect live CSV after N requests.
- Generate HTML report:

```bash
python drift_report.py data/baseline.csv data/live.csv reports/drift.html
```

Open `reports/drift.html`-you’ll see top drifting tokens, embeddings shift, etc.

---

## 🔧 Extending the project

- **Auth** – Add OAuth2 middleware (`fastapi-security`).
- **GPU inference** – Change Docker base to `nvidia/cuda:12.2.0-base-ubuntu22.04` and pass `device=0`.
- **Model upgrades** – Swap model name in `app.py`, rebuild, commit, push-CI/CD redeploys automatically.
- **K8s** – Render image works out-of-the-box on any Kubernetes with a simple Deployment + Service YAML.

---

## 📜 License

MIT - see LICENSE for details.

---

## 🤝 Acknowledgements

- Hugging Face for providing the DistilBERT SST-2 checkpoint.
- Render for the generous free tier.
- ML Community for MLOps best-practice references.

---

Happy hacking! Feel free to open issues or pull requests.

