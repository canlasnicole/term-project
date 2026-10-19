"""Shared settings. All values come from environment variables (.env file locally,
"Secrets" / "Variables" on the hosting service)."""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (does nothing if the file is missing, e.g. in the cloud)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# "gemini-flash-latest" is an alias that Google keeps pointing to the newest Flash model,
# so the app keeps working when an older model is shut down. You can override it in .env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

# Gemini embedding model for the RAG module (runs on Google's servers -> uses no RAM here)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Folder for downloaded datasets, cached results, and the comments database
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Small pre-computed results that are committed to GitHub (e.g. the MovieLens summary)
PRECOMPUTED_DIR = Path(__file__).resolve().parent / "precomputed"

# Ultralytics (YOLO) needs a writable settings folder; the temp folder works everywhere
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "Ultralytics"))

# Where the datasets come from (book: "Python for Data Analysis", 3rd edition)
PYDATA_BASE = "https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets"

# Headers used when downloading images from a URL (same as the professor's notebook)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
HTTP_TIMEOUT = 10
