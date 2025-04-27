# holdwatcher_server.py – one‑file prototype for detecting a live airline agent
# -----------------------------------------------------------------------------
# ❶  Twilio ⇒ WebSocket ⇒ async server (20‑ms μ‑law chunks)
# ❷  VAD + music‑on‑hold filter (Silero)
# ❸  Streaming ASR (faster‑whisper) + greeting‑keyword check
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES (set on **your** server, *never* in GitHub!)
#   TWILIO_ACCOUNT_SID   your Twilio account SID (optional, for future outbound calls)
#   TWILIO_AUTH_TOKEN    your auth token
# -----------------------------------------------------------------------------
"""
Quick‑start
===========
$ python -m venv venv && source venv/bin/activate
$ pip install -r requirements.txt
$ python ingest/holdwatcher_server.py 0.0.0.0 443

Point your Twilio <Stream url="wss://YOUR_DOMAIN:443"/> at it.
"""

from __future__ import annotations
import asyncio, base64, json, logging, os, ssl, sys
from pathlib import Path

import numpy as np
import websockets

# -----------------------------------------------------------------------------
SAMPLE_RATE = 16_000
CHUNK_MS = 20
KEYWORDS = {
    "hello": 0.8,
    "how can i help": 0.6,
    "thank you for calling": 0.6,
    "this is": 0.6,
}

ACCOUNT_SID: str | None = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN: str | None = os.getenv("TWILIO_AUTH_TOKEN")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# μ‑law → 16‑bit PCM (float32)
# -----------------------------------------------------------------------------
μLAW_TABLE = np.sign(np.arange(256) - 127) * (
    (1 / 255) * (((1 + 255) ** (np.abs(np.arange(256) - 127) / 127)) - 1)
)

def mulaw_decode(mu_bytes: bytes) -> np.ndarray:
    idx = np.frombuffer(mu_bytes, dtype=np.uint8)
    return μLAW_TABLE[idx].astype(np.float32)

# -----------------------------------------------------------------------------
# ➊  VAD using Silero (CPU‑only, fast)
# -----------------------------------------------------------------------------
try:
    from silero_vad import SileroVad

    _vad = SileroVad(sample_rate=SAMPLE_RATE)
    logging.info("Silero VAD loaded")
except ImportError:
    _vad = None
    logging.warning("silero‑vad not installed – treating everything as speech")

def is_speech(pcm: np.ndarray) -> bool:
    if _vad is None:
        return True
    return bool((_vad(pcm, return_seconds=False) > 0.5).any())

# -----------------------------------------------------------------------------
# ➋  Streaming ASR – faster‑whisper tiny (CPU‑friendly)
# -----------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel

    _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    logging.info("Whisper model loaded")
except ImportError:
    _whisper = None
    logging.warning("faster‑whisper not installed – ASR disabled")

class ASRStream:
    """Collect 200‑ms windows and emit running transcripts."""

    def __init__(self):
        self.buf = bytearray()
        self.last_ms = 0

    async def feed(self, pcm: np.ndarray) -> str:
        if _whisper is None:
            return ""
        self.buf.extend(pcm.tobytes())
        now_ms = len(self.buf) / SAMPLE_RATE / 2 * 1000  # 16‑bit → half samples
        if now_ms - self.last_ms < 200:
            return ""
        self.last_ms = now_ms
        segments, _ = _whisper.transcribe(np.frombuffer(self.buf, dtype=np.int16), beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segments).lower()

# -----------------------------------------------------------------------------
# ➌  WebSocket handler for Twilio Media Streams
# -----------------------------------------------------------------------------
async def handle_call(ws: websockets.WebSocketServerProtocol):
    logging.info("Twilio stream connected")
    asr = ASRStream()
    async for message in ws:
        data = json.loads(message)
        if data.get("event") != "media":
            continue
        pcm = mulaw_decode(base64.b64decode(data["media"]["payload"]))
        if not is_speech(pcm):
            continue  # still music or silence
        text = await asr.feed(pcm)
        for kw, thresh in KEYWORDS.items():
            if kw in text:
                logging.info(f"AGENT DETECTED – keyword '{kw}' in '{text[:80]}…'")
                # TODO: bridge caller or hit webhook here
                return

# -----------------------------------------------------------------------------
async def run_server(host: str, port: int):
    cert, key = Path("cert.pem"), Path("key.pem")
    if not cert.exists() or not key.exists():
        logging.error("TLS certificate files 'cert.pem' and 'key.pem' are required for wss://")
        sys.exit(1)
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain(cert, key)

    async with websockets.serve(handle_call, host, port, ssl=ssl_ctx, max_size=2**18):
        logging.info(f"HoldWatcher listening on wss://{host}:{port}")
        await asyncio.Future()  # run forever

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ingest/holdwatcher_server.py <host> <port>")
        sys.exit(1)
    asyncio.run(run_server(sys.argv[1], int(sys.argv[2])))
