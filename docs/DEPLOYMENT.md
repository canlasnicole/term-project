# Step-by-step: run locally, Docker, and deploy (free)

Plan: the whole service (Streamlit + FastAPI) runs as **one free app on Streamlit Community Cloud**.
No credit card is needed.

Why not Hugging Face or Render? Since July 2026, Hugging Face needs a paid plan for Docker and Gradio Spaces, and Render's free plan has only 512 MB of memory, which is too small for PyTorch + YOLO.

How it works on one server: `frontend/app.py` checks `BACKEND_URL`. If it is empty, the app starts the FastAPI server in a background thread and calls it over HTTP at `http://127.0.0.1:8000`. The frontend and backend are still two separate programs that talk only through HTTP, so the MVC design does not change.

---

## Step 0. Gemini API key

Get a free key at https://aistudio.google.com/apikey and put it in `.env` as `GEMINI_API_KEY=...`.

## Step 1. Test locally

Activate the virtual environment first (every new terminal):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Two-terminal mode** (separate backend, shows the `/docs` page): keep `BACKEND_URL=http://localhost:8000` in `.env`, then
```powershell
python -m uvicorn backend.main:app --reload --port 8000     # terminal 1
python -m streamlit run frontend/app.py                     # terminal 2
```

**One-terminal mode** (exactly like the cloud): put `#` in front of the `BACKEND_URL` line in `.env`, then
```powershell
python -m streamlit run frontend/app.py
```
The sidebar shows "Mode: FastAPI runs inside this server (embedded)".

Test every tab with `TEST_CHECKLIST.md`.

## Step 2. Push to GitHub

```powershell
git add .
git status          # CHECK: .env and .venv must NOT be in the list
git commit -m "Run FastAPI inside the Streamlit app for free deployment"
git push
```

## Step 3. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
   - Repository: `YOUR-GITHUB/term-project`
   - Branch: `main`
   - Main file path: `frontend/app.py`
   - App URL: choose a name, for example `nicole-ai-term-project`
3. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets (TOML format), and do **not** add `BACKEND_URL`:
```toml
GEMINI_API_KEY = "your-real-key-here"
```
4. Click **Deploy**. The first build installs PyTorch and takes about 5-15 minutes. Click **Manage app** (bottom right) to see the log.
5. When the app opens, the sidebar should show **FastAPI backend connected** and **Gemini API key set**.

To change the key later: **Manage app → ⋮ → Settings → Secrets**, save, then **Reboot app**.

## Step 4. Test from another device

Open the `.streamlit.app` URL on your phone using mobile data (not school Wi-Fi), and do one quick test in each tab. The first CNN, YOLO and RAG requests are slower because models load.

## Step 5. Docker image (for submission)

Install Docker Desktop, then:
```powershell
docker compose up --build
```
Open http://localhost:8501 (frontend) and http://localhost:8000/docs (backend). To submit:
```powershell
docker save -o term-project-image.tar term-project:latest
```
or push it to Docker Hub (`docker tag`, then `docker push`). Ask your professor which one they want.

## Step 6. URL checklist for the course Google Sheet

| Item | Example | Done |
|---|---|---|
| Web service URL | `https://nicole-ai-term-project.streamlit.app` | ☐ |
| GitHub repository (public) | `https://github.com/YOUR-GITHUB/term-project` | ☐ |
| Docker image (Hub link or .tar) | `YOUR-DOCKERHUB/term-project:latest` | ☐ |
| URLs written in the README table | | ☐ |
| Opened from another device, all 5 tabs work | | ☐ |

## Troubleshooting

| Problem | Fix |
|---|---|
| Build log: error while installing packages | Copy the last red lines of the log and check `requirements.txt` for typos. |
| `libGL.so.1: cannot open shared object file` | `packages.txt` is missing from the repo root. Push it and reboot. |
| "This app has gone over its resource limits" | Too much memory. **Reboot app**, then use the tabs one by one. CNN + YOLO + ML together fit; avoid very large PDFs. |
| Sidebar "Gemini API key missing" | Add `GEMINI_API_KEY` in Secrets, then **Reboot app**. |
| Sidebar "Backend not connected" | Make sure there is **no** `BACKEND_URL` in Secrets. Reboot the app. |
| App is asleep ("Yes, get this app back up") | Free apps sleep after a period without visitors. Click the button and wait 1-2 minutes. Open the app before the demo. |
| Gemini error "model not found" | Add `GEMINI_MODEL = "gemini-2.5-flash"` (or another model listed in AI Studio) to Secrets. |
| Gemini error 429 | Free-tier rate limit. Wait one minute. |
| RAG: "No text found" | Scanned PDF without a text layer. Use a PDF with selectable text or a .txt file. |
