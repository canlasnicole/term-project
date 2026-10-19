"""Controller layer: FastAPI routes.
Each route checks the input (schemas.py), calls a service, and returns JSON.
Run locally:  uvicorn backend.main:app --reload --port 8000   ->  open http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from . import gemini_service, ml_service, rag_service, storage
from .config import GEMINI_API_KEY, GEMINI_MODEL
from .image_utils import load_image
from .schemas import (AskIn, AskOut, CNNOut, CommentIn, CommentOut, HealthOut,
                      Passenger, PredictionOut, RagStatus, YOLOOut)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()          # runs once at startup: create the comments table
    yield


app = FastAPI(title="AI + Data Analysis Term Project API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/", tags=["system"])
def root():
    return {"message": "Backend is running. See /docs for all endpoints."}


@app.get("/health", response_model=HealthOut, tags=["system"])
def health():
    return {"status": "ok", "gemini_key_set": bool(GEMINI_API_KEY), "gemini_model": GEMINI_MODEL}


# ======================= 2.1 Webhook + Gemini auto-reply =======================
def _reply(comment: dict) -> dict:
    """Ask Gemini for a reply and save it. On error, save the error message."""
    try:
        text = gemini_service.reply_to_comment(comment["platform"], comment["author"],
                                               comment["content"])
        return storage.set_reply(comment["comment_id"], text, "replied")
    except gemini_service.GeminiError as e:
        return storage.set_reply(comment["comment_id"], str(e), "error")


@app.post("/webhook/comment", response_model=CommentOut, tags=["2.1 comments"])
def receive_comment(comment: CommentIn, auto_reply: bool = True):
    """A platform (YouTube, Instagram ...) calls this URL when a new comment arrives."""
    saved = storage.save_comment(comment.model_dump())
    return _reply(saved) if auto_reply else saved


@app.get("/comments", response_model=List[CommentOut], tags=["2.1 comments"])
def list_comments():
    return storage.list_comments()


@app.get("/comments/{comment_id}", response_model=CommentOut, tags=["2.1 comments"])
def get_comment(comment_id: str):
    c = storage.get_comment(comment_id)
    if not c:
        raise HTTPException(404, "Comment not found")
    return c


@app.post("/comments/{comment_id}/reply", response_model=CommentOut, tags=["2.1 comments"])
def regenerate_reply(comment_id: str):
    c = storage.get_comment(comment_id)
    if not c:
        raise HTTPException(404, "Comment not found")
    return _reply(c)


@app.delete("/comments", tags=["2.1 comments"])
def clear_comments():
    return {"deleted": storage.delete_all()}


@app.post("/gemini/test", tags=["2.1 comments"])
def gemini_test():
    try:
        return {"ok": True, "reply": gemini_service.generate("Say hello in one short sentence.")}
    except gemini_service.GeminiError as e:
        return {"ok": False, "reply": str(e)}


# ======================= 2.2 Data analysis + ML =======================
@app.get("/ml/eda/titanic", tags=["2.2 ml"])
def eda_titanic():
    return ml_service.titanic_eda()


@app.get("/ml/eda/movielens", tags=["2.2 ml"])
def eda_movielens():
    return ml_service.movielens_eda()


@app.get("/ml/compare", tags=["2.2 ml"])
def compare_models():
    return ml_service.train_and_compare()


@app.post("/ml/predict", response_model=PredictionOut, tags=["2.2 ml"])
def predict(p: Passenger):
    return ml_service.predict_passenger(p.model_dump())


@app.get("/ml/test-predictions.csv", response_class=PlainTextResponse, tags=["2.2 ml"])
def test_predictions():
    return ml_service.test_predictions_csv()


# ======================= 2.3 CNN + 2.4 YOLO =======================
async def _read_image(file: Optional[UploadFile], url: Optional[str]):
    try:
        data = await file.read() if file is not None else None
        return load_image(data=data, url=url)
    except Exception as e:
        raise HTTPException(400, f"Could not load image: {e}")


@app.post("/cnn/classify", response_model=CNNOut, tags=["2.3 cnn"])
async def cnn_classify(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    img = await _read_image(file, url)
    from . import cnn_service              # import here: PyTorch loads only when needed
    return cnn_service.classify(img)


@app.post("/yolo/detect", response_model=YOLOOut, tags=["2.4 yolo"])
async def yolo_detect(file: Optional[UploadFile] = File(None), url: Optional[str] = Form(None),
                      conf: float = Form(0.25)):
    img = await _read_image(file, url)
    from . import yolo_service
    return yolo_service.detect(img, conf=conf)


# ======================= 2.5 RAG =======================
@app.post("/rag/upload", response_model=RagStatus, tags=["2.5 rag"])
async def rag_upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(400, "File is larger than 15 MB.")
    try:
        return rag_service.ingest(file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/rag/status", response_model=RagStatus, tags=["2.5 rag"])
def rag_status():
    return rag_service.status()


@app.post("/rag/ask", response_model=AskOut, tags=["2.5 rag"])
def rag_ask(body: AskIn):
    try:
        return rag_service.ask(body.question, body.k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Gemini error: {e}")
