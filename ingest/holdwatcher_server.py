"""
holdwatcher_server.py – detects when a real agent starts speaking
──────────────────────────────────────────────────────────────────
Twilio → WebSocket → async server  (16 kHz µ-law 20 ms frames)
• Silero VAD to ignore hold music / silence
• Gap-based buffering (flush after ≥300 ms silence)
• faster-whisper tiny → streaming ASR
• Keyword match → macOS “say” alert + log

Run:
    python ingest/holdwatcher_server.py 0.0.0.0 8000
Environment:
    TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN  (optional, not used yet)
"""

from __future__ import annotations
import asyncio, base64, json, logging, os, platform, sys, time, pathlib

import numpy as np, websockets            # pip install websockets
from silero_vad import SileroVad          # pip install silero-vad
from faster_whisper import WhisperModel   # pip install faster-whisper

# ────────────────────────────── parameters ──────────────────────────────
SAMPLE_RATE = 16_000
SILENCE_GAP_MS = 300          # flush when this much silence is detected
KEYWORDS = {
    "hello": 0.6,
    "how can i help": 0.5,
    "thank you for calling": 0.5,
    "this is": 0.5,
}
# ─────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s")

decode_pcm = lambda b: np.frombuffer(
    base64.b64decode(b), dtype="<i2"
).astype(np.float32) / 32768

vad = SileroVad(sample_rate=SAMPLE_RATE)
whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

# ────────────────────────────── ASR helper ──────────────────────────────
class GapASR:
    def __init__(self):
        self.buf = bytearray()
        self.last_voiced_ms: float | None = None

    async def feed(self, pcm: np.ndarray) -> list[str]:
        voiced = (vad(pcm, return_seconds=False) > 0.5).any()
        now_ms = time.time() * 1000
        out: list[str] = []

        if voiced:
            self.buf.extend((pcm * 32768).astype("<i2").tobytes())
            self.last_voiced_ms = now_ms
        elif self.buf and self.last_voiced_ms and now_ms - self.last_voiced_ms >= SILENCE_GAP_MS:
            segs, _ = whisper.transcribe(
                np.frombuffer(self.buf, dtype="<i2"), beam_size=1, vad_filter=False
            )
            out.append(" ".join(s.text for s in segs).lower())
            self.buf.clear()
            self.last_voiced_ms = None
        return out

# ───────────────────────────── WebSocket handler ────────────────────────
async def handler(ws):
    logging.info("Twilio stream connected")
    asr = GapASR()

    try:
        async for msg in ws:
            d = json.loads(msg)
            if d.get("event") != "media":
                continue
            for txt in await asr.feed(decode_pcm(d["media"]["payload"])):
                logging.info(f"transcript: {txt}")
                for kw, th in KEYWORDS.items():
                    if kw in txt:
                        logging.info(f"AGENT DETECTED – '{kw}' in '{txt[:80]}…'")
                        if platform.system() == "Darwin":
                            os.system('say "Agent on the line"')
                        return
    except websockets.exceptions.ConnectionClosedError:
        logging.info("Stream closed")

# ─────────────────────────────── main loop ──────────────────────────────
async def main(host: str, port: int):
    async with websockets.serve(handler, host, port, max_size=2 ** 18):
        logging.info(f"HoldWatcher listening on ws://{host}:{port}")
        await asyncio.Future()          # run forever

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ingest/holdwatcher_server.py <host> <port>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
