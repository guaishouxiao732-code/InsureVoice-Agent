"""Local-only streaming CosyVoice worker running in its isolated Conda env."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import sys
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np

from project_config import PROJECT_ROOT, project_path


COSYVOICE_ROOT = project_path("COSYVOICE_CODE_PATH", "third_party/CosyVoice")
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"
for path in (str(MATCHA_ROOT), str(COSYVOICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# The fallback frontend still performs local punctuation splitting. Disabling
# ModelScope network access prevents a service restart from reaching the cloud.
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

import torch  # noqa: E402
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


DEFAULT_MODEL_PATH = project_path("TTS_MODEL_PATH", "models/speech/CosyVoice-300M-SFT")
MAX_REQUEST_BYTES = 256 * 1024


def _frame(meta: dict[str, Any], pcm16: bytes = b"") -> bytes:
    encoded = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return struct.pack("!I", len(encoded)) + encoded + struct.pack("!I", len(pcm16)) + pcm16


class CosyVoiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], model_path: Path, physical_device: str,
    ) -> None:
        super().__init__(address, CosyVoiceHandler)
        self.physical_device = physical_device
        self.model = AutoModel(
            model_dir=str(model_path), load_jit=False, load_trt=False, fp16=True,
        )
        self.model_lock = Lock()
        self.speakers = self.model.list_available_spks()


class CosyVoiceHandler(BaseHTTPRequestHandler):
    server: CosyVoiceHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(200, {
            "status": "ready",
            "engine": "CosyVoice-300M-SFT",
            "device": self.server.physical_device,
            "sample_rate": self.server.model.sample_rate,
            "speakers": self.server.speakers,
        })

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            text = payload.get("text", "")
            speaker = payload.get("speaker", "中文女")
            speed = float(payload.get("speed", 1.0))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must not be empty")
            if speaker not in self.server.speakers:
                raise ValueError(f"unknown speaker: {speaker}")
            if not 0.5 <= speed <= 2.0:
                raise ValueError("speed must be between 0.5 and 2.0")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-cosyvoice-pcm-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        started = perf_counter()
        previous = started
        try:
            with self.server.model_lock, torch.inference_mode():
                for index, item in enumerate(self.server.model.inference_sft(
                    text.strip(), speaker, stream=True, speed=speed,
                )):
                    now = perf_counter()
                    samples = item["tts_speech"].detach().cpu().numpy().reshape(-1)
                    pcm16 = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                    duration_ms = len(samples) / self.server.model.sample_rate * 1000
                    self.wfile.write(_frame({
                        "index": index,
                        "text": text.strip(),
                        "sample_rate": self.server.model.sample_rate,
                        "duration_ms": duration_ms,
                        "synthesis_ms": (now - previous) * 1000,
                        "elapsed_ms": (now - started) * 1000,
                    }, pcm16))
                    self.wfile.flush()
                    previous = now
            self.wfile.write(_frame({"done": True}))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except BaseException as exc:
            try:
                self.wfile.write(_frame({"error": f"{type(exc).__name__}: {exc}"}))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CosyVoice streaming worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--physical-device", default="cuda:0")
    args = parser.parse_args()
    server = CosyVoiceHTTPServer(
        (args.host, args.port), args.model.expanduser().resolve(), args.physical_device,
    )
    print(json.dumps({
        "status": "ready", "host": args.host, "port": args.port,
        "sample_rate": server.model.sample_rate, "speakers": server.speakers,
        "device": server.physical_device,
    }, ensure_ascii=False), flush=True)
    server.serve_forever(poll_interval=0.25)


if __name__ == "__main__":
    main()
