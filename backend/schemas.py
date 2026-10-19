"""Model layer: Pydantic models describe the shape of every request and response.
FastAPI uses them to check input data and to build the /docs page automatically."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- 2.1 Webhook + comments
class CommentIn(BaseModel):
    comment_id: str = Field(..., examples=["cmt_001"])
    platform: str = Field("youtube", examples=["youtube", "instagram", "facebook"])
    post_id: str = Field(..., examples=["post_123"])
    author: str = Field(..., examples=["Nicole"])
    content: str = Field(..., min_length=1, examples=["How long does delivery take?"])


class CommentOut(CommentIn):
    status: str                      # pending / replied / error
    ai_reply: Optional[str] = None
    received_at: str
    replied_at: Optional[str] = None


class HealthOut(BaseModel):
    status: str
    gemini_key_set: bool
    gemini_model: str


# ---------- 2.2 Machine learning
class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3] = 3
    Sex: Literal["male", "female"] = "male"
    Age: float = Field(30, ge=0, le=100)
    SibSp: int = Field(0, ge=0, le=10)     # siblings / spouses on board
    Parch: int = Field(0, ge=0, le=10)     # parents / children on board
    Fare: float = Field(15.0, ge=0)
    Embarked: Literal["S", "C", "Q"] = "S"


class PredictionOut(BaseModel):
    survived: int
    probability: float
    model: str


# ---------- 2.3 CNN
class TopK(BaseModel):
    class_id: int
    class_name: str
    probability: float


class CNNOut(BaseModel):
    class_id: int
    class_name: str
    probability: float
    top5: List[TopK]
    is_dog_or_cat: bool
    animal: Optional[str] = None     # "dog", "cat" or None
    message: str


# ---------- 2.4 YOLO
class Detection(BaseModel):
    class_name: str
    confidence: float
    box: List[float]                 # [x1, y1, x2, y2] in pixels


class YOLOOut(BaseModel):
    count: int
    detections: List[Detection]
    annotated_image_base64: str      # JPEG image with boxes, base64 text


# ---------- 2.5 RAG
class AskIn(BaseModel):
    question: str = Field(..., min_length=2)
    k: int = Field(4, ge=1, le=10)   # how many chunks to retrieve


class SourceChunk(BaseModel):
    rank: int
    text: str
    page: Optional[int] = None
    score: float                     # distance: smaller = more similar


class AskOut(BaseModel):
    answer: str
    sources: List[SourceChunk]


class RagStatus(BaseModel):
    ready: bool
    filename: Optional[str] = None
    chunks: int = 0
