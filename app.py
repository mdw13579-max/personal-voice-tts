import os, time, uuid
from typing import Dict, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Personal Voice TTS", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_STYLE = (
    "Upbeat American middle-aged radio host / coach vibe. "
    "Warm, confident, lightly humorous, clear enunciation, medium-fast pace, "
    "smile in the voice. Avoid sounding like a young influencer."
)

# 간단한 in-memory 저장소 (개인용으로 충분 / 서버 재시작하면 사라짐)
# id -> (created_at_epoch, audio_bytes)
AUDIO_STORE: Dict[str, Tuple[float, bytes]] = {}
TTL_SECONDS = 60 * 30  # 30분 보관

def cleanup():
    now = time.time()
    dead = [k for k, (t, _) in AUDIO_STORE.items() if now - t > TTL_SECONDS]
    for k in dead:
        AUDIO_STORE.pop(k, None)

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to speak")
    style: str = Field(default=DEFAULT_STYLE)
    voice: str = Field(default="alloy", description="e.g. alloy / aria / verse (availability may vary)")
    speed: float = Field(default=1.05, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    id: str
    audio_url: str
    expires_in_seconds: int

@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    cleanup()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    tts_input = f"[STYLE]\n{req.style}\n\n[SCRIPT]\n{req.text.strip()}\n"

    try:
        # OpenAI Audio API (TTS). gpt-4o-mini-tts 지원. :contentReference[oaicite:1]{index=1}
        audio = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="mp3",   # ✅ 여기
            # instructions=... (있으면 그대로)
        )

        if hasattr(audio, "read"):
            audio_bytes = audio.read()
        else:
            audio_bytes = audio.content

        audio_id = uuid.uuid4().hex
        AUDIO_STORE[audio_id] = (time.time(), audio_bytes)

        base_url = os.getenv("PUBLIC_BASE_URL")  # Render에서 서비스 URL 넣기
        if not base_url:
            # 로컬 테스트용
            base_url = "http://localhost:8000"

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
        raise HTTPException(status_code=404, detail="Audio expired or not found")

    _, audio_bytes = item
    return Response(content=audio_bytes, media_type="audio/mpeg")
