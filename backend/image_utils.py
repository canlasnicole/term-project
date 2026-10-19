"""Shared helper: open an image from uploaded bytes OR from a URL (CNN and YOLO use it)."""
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from .config import HTTP_HEADERS, HTTP_TIMEOUT

MAX_BYTES = 10 * 1024 * 1024        # 10 MB limit keeps memory low


def load_image(data: Optional[bytes] = None, url: Optional[str] = None) -> Image.Image:
    if data is None and not url:
        raise ValueError("Send an image file or an image URL.")
    if data is None:
        # Same as the professor's notebook: browser User-Agent + timeout=10
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.content
    if len(data) > MAX_BYTES:
        raise ValueError("Image is larger than 10 MB.")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError("The file or URL is not a valid image.") from e
