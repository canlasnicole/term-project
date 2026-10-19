# AI + Data Analysis Integrated Web Service

One web service that combines five AI / data-analysis functions.
**FastAPI** is the backend (AI models + data), **Streamlit** is the frontend (user interface).

| Live service | URL |
|---|---|
| Web service (Streamlit + FastAPI) | `https://YOUR-APP.streamlit.app` |
| GitHub | `https://github.com/YOUR-GITHUB/term-project` |
| Docker image | `YOUR-DOCKERHUB/term-project:latest` (or the submitted .tar file) |

## 1. Architecture

```
  User's browser (PC / phone)
            │
            ▼
┌─────────────────────────────┐        HTTP + JSON (requests)        ┌───────────────────────────────────┐
│  Streamlit frontend (VIEW)  │  ──────────────────────────────────▶ │  FastAPI backend                   │
│  frontend/app.py            │  ◀──────────────────────────────────  │                                   │
│  5 tabs, forms, charts      │                                       │  main.py      routes (CONTROLLER) │
│                             │                                       │  schemas.py   data shapes (MODEL) │
└─────────────────────────────┘                                       │  storage.py   SQLite (MODEL)      │
                                                                      │  *_service.py AI logic            │
                                                                      └──────────────┬────────────────────┘
                                                                                     │
                     ┌──────────────┬──────────────┬─────────────┬───────────────────┼──────────────┐
                     ▼              ▼              ▼             ▼                   ▼              ▼
                Gemini API     scikit-learn    ResNet-18      YOLOv8n      Gemini embeddings     FAISS
                (replies, RAG)  + XGBoost      (PyTorch)    (Ultralytics)  (gemini-embedding-001) (vector DB)
```

**Deployment:** both parts run on **Streamlit Community Cloud** (free). The Streamlit app starts the FastAPI server in a background thread of the same machine, and every button still calls FastAPI through HTTP (`http://127.0.0.1:8000`). Locally or with Docker, the two parts run as two separate servers. The code is the same in both cases; only `BACKEND_URL` decides the mode.

### MVC (Model - View - Controller)

| Part | Files | Job |
|---|---|---|
| **Model** (data) | `backend/schemas.py`, `backend/storage.py` | Defines what data looks like (Pydantic) and saves it (SQLite). |
| **View** (screen) | `frontend/app.py` | Shows tabs, forms and charts. It never runs a model; it only sends HTTP requests. |
| **Controller** (traffic) | `backend/main.py` | Receives each request, checks it with a schema, calls the right service, returns JSON. |
| Services (business logic) | `backend/*_service.py` | The actual AI work: Gemini, ML, CNN, YOLO, RAG. The controller calls them. |

Because the View only talks to the Controller through HTTP, the frontend and backend can run on two different servers (local / Docker) or on one server (Streamlit Cloud) without changing any code.

## 2. Files

```
term-project/
├── backend/
│   ├── __init__.py
│   ├── main.py            # Controller: all FastAPI routes
│   ├── schemas.py         # Model: Pydantic request/response models
│   ├── storage.py         # Model: SQLite storage for comments
│   ├── config.py          # settings from .env / secrets
│   ├── gemini_service.py  # 2.1 Gemini auto-reply
│   ├── ml_service.py      # 2.2 EDA + model comparison (Titanic, MovieLens)
│   ├── image_utils.py     # open image from upload or URL
│   ├── cnn_service.py     # 2.3 ResNet-18 classification
│   ├── yolo_service.py    # 2.4 YOLOv8n detection
│   ├── rag_service.py     # 2.5 RAG with LangChain + FAISS + Gemini
│   └── precomputed/       # small saved MovieLens summary (saves memory on the server)
├── frontend/
│   └── app.py             # View: Streamlit UI (+ starts FastAPI when BACKEND_URL is empty)
├── .streamlit/config.toml # theme and upload size
├── docs/                  # deployment guide, test checklist, presentation script
├── .env.example           # copy to .env
├── requirements.txt       # all Python packages (CPU-only PyTorch)
├── packages.txt           # 2 Linux libraries that OpenCV needs on Streamlit Cloud
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 3. The five modules

| Tab | What it does | Backend endpoint(s) |
|---|---|---|
| 1. Gemini auto-reply | A webhook receives a comment, saves it, and Gemini writes a reply. Dashboard, webhook simulator, settings. | `POST /webhook/comment`, `GET /comments`, `GET /comments/{id}`, `POST /comments/{id}/reply` |
| 2. Data analysis & ML | Titanic EDA, MovieLens EDA, compare 6 models with Recall / F1 / AUROC / PR-AUC, final model, predictions for `test.csv`, one-passenger form | `GET /ml/eda/titanic`, `GET /ml/eda/movielens`, `GET /ml/compare`, `POST /ml/predict` |
| 3. CNN | ResNet-18 (ImageNet) classifies an uploaded image or URL; top-5 chart; dog/cat check by class ID | `POST /cnn/classify` |
| 4. YOLO | YOLOv8n finds objects, draws boxes, lists class + confidence | `POST /yolo/detect` |
| 5. RAG | Upload PDF/TXT → chunks → Gemini embeddings → FAISS → top-k chunks → Gemini answer + sources | `POST /rag/upload`, `POST /rag/ask` |

**Important note about Titanic `test.csv`:** it has **no `Survived` column** (it is the Kaggle competition file). So it cannot be scored. The app compares models with 5-fold cross-validation on 80% of `train.csv`, reports final performance on the other 20% (hold-out), then retrains on all of `train.csv` and only **predicts** `test.csv` (downloadable CSV).

## 4. Run locally (no Docker)

Requirements: Python 3.11 or 3.12.

```bash
cd term-project
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # Mac/Linux: cp .env.example .env   -> then put your Gemini key in .env
```

Terminal 1 (backend):
```bash
uvicorn backend.main:app --reload --port 8000
```
Open http://localhost:8000/docs to see and test every endpoint.

Terminal 2 (frontend):
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501.

The first run downloads the Titanic data and model weights (ResNet-18 45 MB, YOLOv8n 6 MB). The MovieLens summary is already saved in `backend/precomputed/`.

**One-terminal mode:** remove the `BACKEND_URL` line from `.env` and run only `streamlit run frontend/app.py`. The app then starts FastAPI by itself (this is how the cloud version runs).

## 5. Run with Docker

```bash
docker compose up --build
```
Frontend: http://localhost:8501 · Backend: http://localhost:8000/docs

Backend only:
```bash
docker build -t term-project .
docker run -p 8000:7860 --env-file .env term-project
```

## 6. Environment variables

| Name | Where | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | backend | Free key from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | backend (optional) | Default `gemini-flash-latest` |
| `EMBEDDING_MODEL` | backend (optional) | Default `gemini-embedding-001` |
| `BACKEND_URL` | frontend (optional) | URL of a separate FastAPI backend. Empty = start FastAPI inside the Streamlit app |

## 7. Deployment

Full beginner steps: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Summary: one free app on Streamlit Community Cloud, connected to GitHub, with `GEMINI_API_KEY` as a secret.

Test steps: [`docs/TEST_CHECKLIST.md`](docs/TEST_CHECKLIST.md). Presentation: [`docs/PRESENTATION.md`](docs/PRESENTATION.md).

## 8. Data sources

- Titanic and MovieLens 1M from *Python for Data Analysis*, 3rd ed. (Wes McKinney), https://github.com/wesm/pydata-book/tree/3rd-edition/datasets
- ResNet-18 ImageNet weights (torchvision), YOLOv8n COCO weights (Ultralytics), Gemini API (text and embeddings)
