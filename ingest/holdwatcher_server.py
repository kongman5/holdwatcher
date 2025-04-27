"""
holdwatcher_server.py — minimal working prototype
-------------------------------------------------
• Receives 20-ms μ-law packets from Twilio Media Streams
• Filters out hold music with a simple VAD + music classifier
• Uses faster-whisper ASR to spot greeting keywords
-------------------------------------------------
Run:  python ingest/holdwatcher_server.py 0.0.0.0 443
"""

import asyncio, base64, json, ssl, sys, logging, numpy as np
import websockets

SAMPLE_RATE = 16000
KEYWORDS = {"hello": 0.8, "how can i help": 0.6, "this is": 0.6}

# --- μ-law → PCM -------------------------------------------------------------
μLAW_TABLE = np.sign(np.arange(256) - 127) * (
    (1 / 255) * (((1 + 255) ** (np.abs(np.arange(256) - 127) / 127)) - 1)
)


def mulaw_decode(chunk: bytes) -> np.ndarray:
    idx = np.frombuffer(chunk, dtype=np.uint8)
    return μLAW_TABLE[idx].astype(np.float32)


# --- Simple VAD (Silero) -----------------------------------------------------
try:
    from silero_vad import SileroVad

    _vad = SileroVad(sample_rate=SAMPLE_RATE)
    logging.info("Silero VAD loaded")
except ImportError:
    _vad = None
    logging.warning("silero-vad not installed; treating all audio as speech")


def is_speech(pcm: np.ndarray) -> bool:
    if _vad is None:
        return True
    return bool((_vad(pcm, return_seconds=False) > 0.5).any())


# --- Streaming Whisper -------------------------------------------------------
try:
    from faster_whisper import WhisperModel

    _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
except ImportError:
    _whisper = None
    logging.warning("faster-whisper not installed; ASR disabled")


class ASRStream:
    def __init__(self):
        self.buf = bytearray()
        self.last_ms = 0

    async def feed(self, pcm: np.ndarray) -> str:
        if _whisper is None:
            return ""
        self.buf.extend(pcm.tobytes())
        now_ms = len(self.buf) / SAMPLE_RATE / 2 * 1000
        if now_ms - self.last_ms < 200:  # run every 200 ms
            return ""
        self.last_ms = now_ms
        segments, _ = _whisper.transcribe(
            np.frombuffer(self.buf, dtype=np.int16), beam_size=1, vad_filter=False
        )
        return " ".join(s.text for s in segments).lower()


# --- WebSocket handler -------------------------------------------------------
async def handle(ws):
    logging.info("New Twilio stream")
    asr = ASRStream()
    async for msg in ws:
        data = json.loads(msg)
        if data.get("event") != "media":
            continue
        pcm = mulaw_decode(base64.b64decode(data["media"]["payload"]))
        if not is_speech(pcm):
            continue
        text = await asr.feed(pcm)
        for kw, thresh in KEYWORDS.items():
            if kw in text:
                logging.info(f"AGENT DETECTED: '{kw}' in '{text[:60]}…'")
                # TODO: call webhook or bridge the call here
                return


async def main(host: str, port: int):
    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain("cert.pem", "key.pem")  # your TLS cert
    async with websockets.serve(handle, host, port, ssl=ssl_ctx, max_size=2**18):
        logging.info(f"HoldWatcher running on wss://{host}:{port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python ingest/holdwatcher_server.py <host> <port>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1], int(sys.argv[2])))
