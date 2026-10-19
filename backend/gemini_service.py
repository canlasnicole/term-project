"""Module 2.1 - Gemini auto-reply. Uses the official Google Gen AI SDK (google-genai)."""
from google import genai

from .config import GEMINI_API_KEY, GEMINI_MODEL

_client = None

REPLY_PROMPT = """You are a friendly customer-care manager answering a comment on {platform}.
Write a short, polite reply (2-4 sentences) in the SAME language as the comment.
Thank the person, answer the question if you can, and never invent prices, dates or policies.
If you do not know the answer, say a staff member will follow up.

Author: {author}
Comment: {content}

Reply:"""


class GeminiError(Exception):
    pass


def get_client():
    """Create the Gemini client once (lazy) and reuse it."""
    global _client
    if not GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not set. Add it to .env or the server secrets.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate(prompt: str) -> str:
    try:
        resp = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (resp.text or "").strip()
    except GeminiError:
        raise
    except Exception as e:                      # quota, wrong model name, network ...
        raise GeminiError(f"Gemini call failed ({GEMINI_MODEL}): {e}") from e


def reply_to_comment(platform: str, author: str, content: str) -> str:
    return generate(REPLY_PROMPT.format(platform=platform, author=author, content=content))
