import os
import time
import uuid
from typing import Dict, Tuple, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "3600"))  # 1 hour default
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "alloy")
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED", "1.05"))

# -----------------------------
# App & Client
# -----------------------------
app = FastAPI(title="Personal Voice TTS", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory store: audio_id -> (created_ts, mp3_bytes)
AUDIO_STORE: Dict[str, Tuple[float, bytes]] = {}


# -----------------------------
# Models
# -----------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to speak")
    style: Optional[str] = Field(
        default="",
        description="Style guide for how the voice should sound",
    )
    voice: Optional[str] = Field(default=DEFAULT_VOICE, description="Voice name")
    speed: Optional[float] = Field(default=DEFAULT_SPEED, description="Speech speed")


class TTSResponse(BaseModel):
    id: str
    audio_url: str
    expires_in_seconds: int


# -----------------------------
# Helpers
# -----------------------------
def cleanup() -> None:
    """Remove expired audio from memory."""
    now = time.time()
    expired = [k for k, (ts, _) in AUDIO_STORE.items() if now - ts > TTL_SECONDS]
    for k in expired:
        AUDIO_STORE.pop(k, None)


def get_base_url(request: Request) -> str:
    """Prefer PUBLIC_BASE_URL; otherwise infer from incoming request."""
    env_url = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    # infer from request (works on Render behind proxy)
    return str(request.base_url).rstrip("/")


def build_tts_input(style: str, text: str) -> str:
    style = (style or "").strip()
    text = (text or "").strip()
    if style:
        return f"[STYLE]\n{style}\n\n[SCRIPT]\n{text}\n"
    return text


def create_speech_mp3(input_text: str, voice: str, speed: float) -> bytes:
    """
    Generate MP3 bytes via OpenAI TTS.
    - Uses response_format="mp3"
    - Tries speed; if SDK/model doesn't accept speed, retries without it.
    """
    voice = (voice or DEFAULT_VOICE).strip()
    if not voice:
        voice = DEFAULT_VOICE

    # 1) Try with speed
    try:
        audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=input_text,
            response_format="mp3",
            speed=float(speed),
        )
    except TypeError:
        # 2) Retry without speed if not supported
        audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=input_text,
            response_format="mp3",
        )

    # openai-python returns a response-like object; handle both patterns.
    if hasattr(audio, "read"):
        return audio.read()
    # fallback
    return getattr(audio, "content", b"")


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"ok": True, "docs": "/docs"}


@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest, request: Request):
    cleanup()

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    voice = (req.voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE
    speed = float(req.speed or DEFAULT_SPEED)

    tts_input = build_tts_input(req.style or "", text)

    try:
        audio_bytes = create_speech_mp3(tts_input, voice, speed)
        if not audio_bytes:
            raise RuntimeError("Empty audio returned")

        audio_id = uuid.uuid4().hex
        AUDIO_STORE[audio_id] = (time.time(), audio_bytes)

        base_url = get_base_url(request)
        return TTSResponse(
            id=audio_id,
            audio_url=f"{base_url}/audio/{audio_id}.mp3",
            expires_in_seconds=TTL_SECONDS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get("/audio/{audio_id}.mp3")
def get_audio(audio_id: str):
    cleanup()
    item = AUDIO_STORE.get(audio_id)
    if not item:
        raise HTTPException(status_code=404, detail="audio not found (expired or invalid id)")

    _, audio_bytes = item
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )
