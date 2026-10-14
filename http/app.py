#!/usr/bin/env python3
"""Responsive web gateway for the local Qwen/RAG/ASR/TTS assistant."""

from __future__ import annotations

import asyncio
import base64
import json
from contextvars import ContextVar
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import subprocess
import sys
from threading import Event, Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import project_path

DB_PATH = project_path("WEB_DB_PATH", "data/web_conversations.sqlite3")

MAX_AUDIO_BYTES = 12 * 1024 * 1024
COOKIE_NAME = "assistant_access"


class AssistantRuntime:
    """Own the expensive models and expose one single-user service."""

    def __init__(self) -> None:
        self._load_lock = Lock()
        self.state = "idle"
        self.error: str | None = None
        self.context: Any = None
        self.voice: Any = None
        self.tts: Any = None
        self.asr: Any = None
        self.rag: Any = None

    def load(self) -> None:
        if self.state == "ready":
            return
        with self._load_lock:
            if self.state == "ready":
                return
            self.state = "loading"
            self.error = None
            try:
                from api import Qwen3Client
                from context_manager import ContextManager
                from cooperative import CooperativeClient
                from rag import RAGClient, RAGRetriever
                from speech import ASRClient, TTSClient
                from voice_assistant import VoiceAssistant

                asr = ASRClient()
                self.asr = asr
                self.tts = TTSClient(
                    speaker=os.environ.get("TTS_SPEAKER", "中文女"),
                    speed=float(os.environ.get("TTS_SPEED", "1.0")),
                    device=os.environ.get("TTS_DEVICE", "cuda:0"),
                    server_url=os.environ.get("TTS_SERVER_URL") or None,
                )
                retriever = RAGRetriever(embedding_device="cpu")
                qwen = Qwen3Client(device=os.environ.get("QWEN_DEVICE", "cuda:1"))
                qwen.warmup()
                self.tts.warmup()
                rag = RAGClient(qwen, retriever)
                self.rag = CooperativeClient(rag)
                self.context = ContextManager(self.rag, db_path=DB_PATH)
                self.voice = VoiceAssistant(self.context, asr, self.tts)
                self.state = "ready"
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "failed"
                raise

    def close(self) -> None:
        if self.context is not None:
            self.context.close()
        self.state = "closed"

    def interrupt(self, session_id: str | None) -> bool:
        changed = bool(self.voice and self.voice.interrupt())
        if session_id and self.context is not None:
            try:
                changed = self.context.interrupt(session_id) or changed
            except KeyError:
                pass
        return changed

    def request_interrupt(self, session_id: str | None) -> None:
        if self.voice is not None:
            self.voice.request_interrupt()
        if session_id and self.context is not None:
            try:
                self.context.request_interrupt(session_id)
            except KeyError:
                pass


runtime = AssistantRuntime()
ACCESS_PASSWORD = os.environ.get("WEB_PASSWORD", "").strip()
_login_tokens: set[str] = set()
_load_task: asyncio.Task[None] | None = None


def _authorized_token(token: str | None) -> bool:
    return not ACCESS_PASSWORD or bool(token and token in _login_tokens)


def _request_authorized(request: Request) -> bool:
    return _authorized_token(request.cookies.get(COOKIE_NAME))


def _session_summaries() -> list[dict[str, Any]]:
    result = []
    for item in reversed(runtime.context.list_sessions()):
        history = runtime.context.get_history(item["session_id"])
        title = history[0]["user_text"][:24] if history else "新对话"
        result.append({
            "session_id": item["session_id"],
            "created_at": item["created_at"],
            "title": title,
            "turn_count": len(history),
        })
    return result


def _public_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        key: turn.get(key) for key in (
            "turn_id", "user_text", "assistant_text", "status", "created_at",
            "finished_at", "first_chunk_ms", "metadata",
        )
    }


def _mode_options(mode: str) -> dict[str, Any]:
    if mode == "slow":
        return {"thinking_mode": "slow", "max_new_tokens": 512, "temperature": 0.0, "top_p": 0.8}
    if mode == "fast":
        return {"thinking_mode": "fast", "max_new_tokens": 192, "temperature": 0.0, "top_p": 0.8}
    return {"thinking_mode": "auto"}


def _thinking_mode(value: Any) -> str:
    return value if isinstance(value, str) and value in {"auto", "fast", "slow"} else "auto"


def _decode_browser_audio(encoded: str) -> bytes:
    try:
        source = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("录音数据格式不正确") from exc
    if not source or len(source) > MAX_AUDIO_BYTES:
        raise ValueError("录音为空或超过 12 MB")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
            ],
            input=source,
            capture_output=True,
            check=True,
            timeout=45,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 FFmpeg，无法解码浏览器录音") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", "replace").strip()
        raise ValueError(f"录音解码失败：{detail or '不支持的音频'}") from exc
    return completed.stdout


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _load_task
    if os.environ.get("WEB_SKIP_MODEL_LOAD") != "1":
        _load_task = asyncio.create_task(asyncio.to_thread(runtime.load))
    yield
    if _load_task and not _load_task.done():
        _load_task.cancel()
    await asyncio.to_thread(runtime.close)


app = FastAPI(title="智能业务助手", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/healthz")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": runtime.state, "error": runtime.error}


@app.get("/api/auth")
async def auth_status(request: Request) -> dict[str, bool]:
    return {"required": bool(ACCESS_PASSWORD), "authenticated": _request_authorized(request)}


@app.post("/api/login")
async def login(request: Request) -> JSONResponse:
    try:
        supplied = str((await request.json()).get("password", ""))
    except Exception as exc:
        raise HTTPException(400, "请求格式不正确") from exc
    if ACCESS_PASSWORD and not secrets.compare_digest(supplied, ACCESS_PASSWORD):
        raise HTTPException(401, "访问密码错误")
    token = secrets.token_urlsafe(32)
    _login_tokens.add(token)
    response = JSONResponse({"ok": True})
    forwarded_https = request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, secure=forwarded_https,
        samesite="lax", max_age=12 * 60 * 60,
    )
    return response


async def _ensure_ready() -> None:
    global _load_task
    if runtime.state == "ready":
        return
    if _load_task is None or (_load_task.done() and runtime.state != "ready"):
        _load_task = asyncio.create_task(asyncio.to_thread(runtime.load))
    await _load_task


async def _next(iterator: Any) -> tuple[bool, Any]:
    def get_next() -> tuple[bool, Any]:
        try:
            return True, next(iterator)
        except StopIteration:
            return False, None
    return await asyncio.to_thread(get_next)


@app.websocket("/ws")
async def assistant_socket(websocket: WebSocket) -> None:
    if not _authorized_token(websocket.cookies.get(COOKIE_NAME)):
        await websocket.close(code=4401, reason="unauthorized")
        return
    await websocket.accept()
    send_lock = asyncio.Lock()
    current_session: str | None = None
    active_task: asyncio.Task[None] | None = None
    active_cancel: Event | None = None
    response_serial = 0
    response_context = ContextVar("response_id", default=None)
    stop_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task] = set()

    def spawn(coroutine):
        serial = response_serial
        async def run():
            token = response_context.set(serial)
            try:
                await coroutine
            finally:
                response_context.reset(token)
        task = asyncio.create_task(run())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return task

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            serial = response_context.get()
            if serial is not None:
                if serial != response_serial:
                    return
                payload = {**payload, "response_id": serial}
            await websocket.send_json(payload)

    async def send_error(exc: BaseException) -> None:
        try:
            await send({"type": "error", "message": str(exc) or type(exc).__name__})
        except BaseException:
            # The browser may already be gone while a model worker is stopping.
            pass

    async def refresh_sessions() -> None:
        items = await asyncio.to_thread(_session_summaries)
        await send({"type": "sessions", "items": items})

    async def stop_active(expected=None) -> None:
        nonlocal active_task
        async with stop_lock:
            task = expected or active_task
            if task is not active_task:
                return
            if task and not task.done():
                if active_cancel is not None:
                    active_cancel.set()
                await asyncio.to_thread(runtime.interrupt, current_session)
                # Keep the receive loop live while workers finish cooperative cancellation.
                # Do not abandon an in-flight model construction in a background thread.
                await asyncio.gather(task, return_exceptions=True)
            if active_task is task:
                active_task = None

    async def finish_turn(response: Any, mode: str) -> None:
        await send({
            "type": "turn_done", "status": response.status, "text": response.text,
            "metadata": response.metadata, "mode": mode,
            "first_chunk_ms": getattr(response, "first_chunk_ms", None),
        })
        await refresh_sessions()

    async def voice_response(response: Any, mode: str) -> None:
        sent_chars = 0
        sent_plan = False
        sent_text_done = False

        async def send_text():
            nonlocal sent_chars, sent_plan, sent_text_done
            plan = response.metadata.get("cooperation")
            if plan and not sent_plan:
                await send({"type": "cooperation", "plan": plan})
                sent_plan = True
            text = response.text
            if len(text) > sent_chars:
                await send({"type": "text_delta", "text": text[sent_chars:]})
                sent_chars = len(text)
            done = getattr(response, "text_done", None)
            if done is not None and done.is_set() and not sent_text_done:
                await send({"type": "text_done", "text": text, "metadata": response.metadata})
                sent_text_done = True
        try:
            while True:
                pending = asyncio.create_task(_next(response))
                while not pending.done():
                    await send_text()
                    await asyncio.sleep(0.035)
                available, chunk = await pending
                await send_text()
                if not available:
                    break
                await send({
                    "type": "audio_chunk", "sample_rate": chunk.sample_rate,
                    "pcm": base64.b64encode(chunk.pcm16).decode("ascii"),
                    "duration_ms": chunk.duration_ms,
                })
            await send({"type": "audio_end"})
            await finish_turn(response, mode)
        finally:
            await asyncio.to_thread(response.close)

    async def text_turn(text: str, mode: str, autoplay: bool, speed: float, stop: Event) -> None:
        try:
            await _ensure_ready()
            if stop.is_set():
                return
            await send({"type": "turn_start", "question": text, "mode": mode})
            options = _mode_options(mode)
            if autoplay:
                response = await asyncio.to_thread(
                    runtime.voice.stream_text_chat, current_session, text,
                    tts_speed=speed, cancel_event=stop, **options,
                )
                if stop.is_set():
                    response.request_cancel()
                    await asyncio.to_thread(response.close)
                    return
                response.response.update_metadata({"web": {"mode": mode}})
                await voice_response(response, mode)
            else:
                response = await asyncio.to_thread(
                    runtime.context.stream_chat, current_session, text, **options,
                )
                if stop.is_set():
                    response.request_cancel()
                response.update_metadata({"web": {"mode": mode}})
                try:
                    sent_plan = False
                    while True:
                        available, chunk = await _next(response)
                        plan = response.metadata.get("cooperation")
                        if plan and not sent_plan:
                            await send({"type": "cooperation", "plan": plan})
                            sent_plan = True
                        if not available:
                            break
                        await send({"type": "text_delta", "text": chunk})
                    await finish_turn(response, mode)
                finally:
                    await asyncio.to_thread(response.close)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if not stop.is_set():
                await send_error(exc)

    async def recorded_turn(encoded: str, mode: str, speed: float, stop: Event) -> None:
        try:
            await _ensure_ready()
            await send({"type": "stage", "stage": "asr", "label": "正在识别语音"})
            wav = await asyncio.to_thread(_decode_browser_audio, encoded)
            if stop.is_set():
                return
            response = await asyncio.to_thread(
                runtime.voice.stream_chat, current_session, wav,
                tts_speed=speed, cancel_event=stop, **_mode_options(mode),
            )
            if stop.is_set():
                response.request_cancel()
                await asyncio.to_thread(response.close)
                return
            response.response.update_metadata({"web": {"mode": mode}})
            await send({
                "type": "turn_start", "question": response.transcript,
                "mode": mode, "from_voice": True,
            })
            await voice_response(response, mode)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if not stop.is_set():
                await send_error(exc)

    async def speak(text: str, speed: float, stop: Event) -> None:
        iterator = None
        try:
            await _ensure_ready()
            await send({"type": "audio_reset"})
            # Replay can wait for full-utterance synthesis; unlike first-response
            # autoplay it does not need to meet the low-latency first-audio gate.
            iterator = runtime.tts.synthesize_stream(
                text, speed=speed, cancel_event=stop, stream=False,
            )
            while True:
                available, chunk = await _next(iterator)
                if response_context.get() != response_serial:
                    break
                if not available:
                    break
                await send({
                    "type": "audio_chunk", "sample_rate": chunk.sample_rate,
                    "pcm": base64.b64encode(chunk.pcm16).decode("ascii"),
                    "duration_ms": chunk.duration_ms,
                })
            await send({"type": "audio_end"})
        except BaseException as exc:
            if not stop.is_set():
                await send_error(exc)
        finally:
            if iterator is not None:
                await asyncio.to_thread(iterator.close)

    try:
        await send({"type": "service", "state": runtime.state})
        await _ensure_ready()
        sessions = await asyncio.to_thread(_session_summaries)
        if sessions:
            current_session = sessions[0]["session_id"]
        await send({"type": "ready", "session_id": current_session})
        await refresh_sessions()
        history = (
            await asyncio.to_thread(runtime.context.get_history, current_session)
            if current_session else []
        )
        await send({
            "type": "history", "session_id": current_session,
            "turns": [_public_turn(turn) for turn in history],
        })

        while True:
            packet = await websocket.receive()
            if packet["type"] == "websocket.disconnect":
                break
            if packet.get("bytes") is not None:
                await send({"type": "error", "message": "连续语音已移除，请录音结束后再发送"})
                continue
            data = json.loads(packet.get("text") or "{}")
            message_type = data.get("type")
            if message_type in {"listen_start", "listen_stop"}:
                await send({"type": "error", "message": "连续语音已移除，请刷新页面使用录完再发送"})
                continue
            response_serial += 1
            await send({"type": "response_epoch", "response_id": response_serial})
            if message_type in {"chat", "voice"}:
                await stop_active()
                if current_session is None:
                    current_session = await asyncio.to_thread(runtime.context.create_session)
                mode = _thinking_mode(data.get("mode"))
                speed = min(1.4, max(0.8, float(data.get("speed", 1.0))))
                active_cancel = Event()
                if message_type == "chat":
                    text = str(data.get("text", "")).strip()
                    if not text:
                        raise ValueError("请输入问题")
                    active_task = spawn(
                        text_turn(text, mode, bool(data.get("autoplay", True)), speed, active_cancel)
                    )
                else:
                    active_task = spawn(
                        recorded_turn(str(data.get("audio", "")), mode, speed, active_cancel)
                    )
            elif message_type == "interrupt":
                if active_cancel is not None:
                    active_cancel.set()
                await send({"type": "interrupted"})
                request = getattr(runtime, "request_interrupt", None)
                if request is not None:
                    await asyncio.to_thread(request, current_session)
                if active_task is not None and not active_task.done():
                    cleanup = asyncio.create_task(stop_active(active_task))
                    background_tasks.add(cleanup)
                    cleanup.add_done_callback(background_tasks.discard)
            elif message_type == "new_session":
                await stop_active()
                current_session = await asyncio.to_thread(runtime.context.create_session)
                await send({"type": "history", "session_id": current_session, "turns": []})
                await refresh_sessions()
            elif message_type == "load_session":
                await stop_active()
                requested = str(data.get("session_id", ""))
                history = await asyncio.to_thread(runtime.context.get_history, requested)
                current_session = requested
                await send({
                    "type": "history", "session_id": current_session,
                    "turns": [_public_turn(turn) for turn in history],
                })
            elif message_type == "clear_session":
                await stop_active()
                if current_session is None:
                    await send({"type": "error", "message": "当前没有可清空的会话"})
                    continue
                await asyncio.to_thread(runtime.context.clear_history, current_session)
                await send({"type": "history", "session_id": current_session, "turns": []})
                await refresh_sessions()
            elif message_type == "delete_session":
                requested = str(data.get("session_id", ""))
                if current_session == requested:
                    await stop_active()
                await asyncio.to_thread(runtime.context.delete_session, requested)
                sessions = await asyncio.to_thread(_session_summaries)
                if current_session == requested:
                    current_session = sessions[0]["session_id"] if sessions else None
                    history = (
                        await asyncio.to_thread(runtime.context.get_history, current_session)
                        if current_session else []
                    )
                    await send({
                        "type": "history", "session_id": current_session,
                        "turns": [_public_turn(turn) for turn in history],
                    })
                await send({"type": "sessions", "items": sessions})
            elif message_type == "speak":
                if active_task and not active_task.done():
                    await stop_active()
                text = str(data.get("text", "")).strip()
                speed = min(1.5, max(0.7, float(data.get("speed", 1.0))))
                active_cancel = Event()
                active_task = spawn(speak(text, speed, active_cancel))
            else:
                await send({"type": "error", "message": "不支持的操作"})
    except WebSocketDisconnect:
        pass
    except BaseException as exc:
        try:
            await send_error(exc)
        except BaseException:
            pass
    finally:
        response_serial += 1
        if active_cancel is not None:
            active_cancel.set()
        await stop_active()
        for task in list(background_tasks):
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
