"""Persistent text conversations and cooperative interruption for Qwen3.

One ContextManager owns a SQLite file; share it between threads/sessions.
Only text yielded to the consumer is persisted as an assistant reply.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
import fcntl
import json
from pathlib import Path
from queue import Empty, Queue
import sqlite3
from threading import Event, Lock, RLock, Thread
from time import perf_counter, time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from project_config import project_path

if TYPE_CHECKING:
    from api import Qwen3Client

DEFAULT_DB_PATH = project_path("CONTEXT_DB_PATH", "data/conversations.sqlite3")
DEFAULT_SYSTEM_PROMPT = (
    "你是一名简洁、准确的中文助手。结合当前对话理解用户的指代和追问，"
    "优先回应用户最新的问题或更正。不确定时明确说明并询问必要信息。"
    "若前一条回答被打断，只能认为用户看过记录中的部分，不要假设未输出的内容已告知用户。"
)


class _SQLiteHistory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A single owner makes startup recovery safe. SQLite still handles all
        # transactions; flock prevents competing model services sharing history.
        self._owner = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._owner.close()
            raise RuntimeError(f"This database already has an active ContextManager: {self.path}") from exc
        self._lock = RLock()
        try:
            self._db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS context_sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS context_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES context_sessions(session_id),
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN
                        ('generating', 'completed', 'interrupted', 'failed')),
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    first_chunk_ms REAL,
                    error TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS context_turns_session
                    ON context_turns(session_id, turn_id);
            """)
            columns = {
                row[1] for row in self._db.execute("PRAGMA table_info(context_turns)")
            }
            if "metadata_json" not in columns:
                self._db.execute("ALTER TABLE context_turns ADD COLUMN metadata_json TEXT")
            with self._db:
                self._db.execute(
                    "UPDATE context_turns SET status='interrupted', finished_at=? "
                    "WHERE status='generating'", (time(),),
                )
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._owner.close()
            raise

    def create_session(self, session_id: str, system_prompt: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO context_sessions VALUES (?, ?, ?)",
                (session_id, system_prompt, time()),
            )

    def session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM context_sessions WHERE session_id=?", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown session_id: {session_id}")
            return dict(row)

    def sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._db.execute(
                "SELECT * FROM context_sessions ORDER BY created_at, session_id"
            )]

    def turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self.session(session_id)
            turns = []
            for row in self._db.execute(
                "SELECT * FROM context_turns WHERE session_id=? ORDER BY turn_id", (session_id,)
            ):
                turn = dict(row)
                raw_metadata = turn.pop("metadata_json", None)
                turn["metadata"] = json.loads(raw_metadata) if raw_metadata else {}
                turns.append(turn)
            return turns

    def begin_turn(self, session_id: str, prompt: str) -> int:
        with self._lock, self._db:
            result = self._db.execute(
                "INSERT INTO context_turns(session_id, user_text, status, created_at) "
                "VALUES (?, ?, 'generating', ?)", (session_id, prompt, time()),
            )
            return int(result.lastrowid)

    def append(self, turn_id: int, chunk: str, latency_ms: float) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE context_turns SET assistant_text=assistant_text || ?, "
                "first_chunk_ms=COALESCE(first_chunk_ms, ?) "
                "WHERE turn_id=? AND status='generating'", (chunk, latency_ms, turn_id),
            )

    def set_metadata(self, turn_id: int, metadata: dict[str, Any]) -> None:
        serialized = json.dumps(metadata, ensure_ascii=False)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE context_turns SET metadata_json=? WHERE turn_id=?",
                (serialized, turn_id),
            )

    def merge_metadata(self, turn_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
        """Merge top-level request metadata without discarding other wrappers."""
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT metadata_json FROM context_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown turn_id: {turn_id}")
            current = json.loads(row[0]) if row[0] else {}
            current.update(metadata)
            self._db.execute(
                "UPDATE context_turns SET metadata_json=? WHERE turn_id=?",
                (json.dumps(current, ensure_ascii=False), turn_id),
            )
            return current

    def finish(self, turn_id: int, status: str, error: str | None = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE context_turns SET status=?, finished_at=?, error=? "
                "WHERE turn_id=? AND status='generating'", (status, time(), error, turn_id),
            )

    def clear(self, session_id: str) -> None:
        with self._lock, self._db:
            self.session(session_id)
            self._db.execute("DELETE FROM context_turns WHERE session_id=?", (session_id,))

    def delete_session(self, session_id: str) -> None:
        """Permanently remove one session and all of its turns."""
        with self._lock, self._db:
            self.session(session_id)
            self._db.execute("DELETE FROM context_turns WHERE session_id=?", (session_id,))
            self._db.execute("DELETE FROM context_sessions WHERE session_id=?", (session_id,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
            self._owner.close()


class ResponseStream(Iterator[str]):
    """A single-consumer response with explicit cancellation and delivery tracking.

    Use as a context manager if you might break out of iteration early. Starting
    another request for the same session also closes this stream automatically.
    """

    def __init__(
        self, client: Qwen3Client, store: _SQLiteHistory, turn_id: int,
        messages: list[dict[str, str]], options: dict[str, Any], started: float,
    ) -> None:
        self.turn_id = turn_id
        self._client, self._store = client, store
        self._messages, self._options = messages, options
        self._started = started
        self._stop = Event()
        self._queue: Queue[str | None] = Queue()
        self._lock = Lock()
        self._consumer_lock = Lock()
        self._status = "generating"
        self._error: BaseException | None = None
        self._text = ""
        self._first_chunk_ms: float | None = None
        self._metadata: dict[str, Any] = {}
        self._worker = Thread(target=self._produce, name=f"context-turn-{turn_id}", daemon=True)

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def text(self) -> str:
        """The prefix already yielded to the caller, not buffered future text."""
        with self._lock:
            return self._text

    @property
    def first_chunk_ms(self) -> float | None:
        with self._lock:
            return self._first_chunk_ms

    @property
    def metadata(self) -> dict[str, Any]:
        """Request metadata such as RAG sources, available after retrieval starts."""
        with self._lock:
            return dict(self._metadata)

    def _finish(self, status: str, error: str | None = None) -> None:
        # Caller holds _lock, serializing persistence and delivery with cancel.
        if self._status == "generating":
            self._store.finish(self.turn_id, status, error)
            self._status = status

    def _produce(self) -> None:
        try:
            source_iterator = self._client.stream_chat(
                self._messages, cancel_event=self._stop, **self._options
            )
            metadata = getattr(source_iterator, "metadata", None)
            if isinstance(metadata, dict):
                with self._lock:
                    self._metadata.update(metadata)
                    self._store.merge_metadata(self.turn_id, metadata)
            with closing(source_iterator) as source:
                for chunk in source:
                    if self._stop.is_set():
                        break
                    if chunk:
                        self._queue.put(chunk)
        except BaseException as exc:
            with self._lock:
                if self._status == "generating":
                    self._error = exc
                    self._finish("failed", f"{type(exc).__name__}: {exc}")
        finally:
            self._messages = []
            self._options = {}
            self._queue.put(None)

    def __iter__(self) -> ResponseStream:
        return self

    def __next__(self) -> str:
        if not self._consumer_lock.acquire(blocking=False):
            raise RuntimeError("A ResponseStream supports only one consumer")
        try:
            while True:
                with self._lock:
                    if self._status == "failed":
                        raise RuntimeError("Conversation generation failed") from self._error
                    if self._status != "generating":
                        raise StopIteration
                try:
                    chunk = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                with self._lock:
                    # A new message may have interrupted us during queue.get().
                    if self._status != "generating":
                        continue
                    if chunk is None:
                        self._finish("completed")
                        raise StopIteration
                    latency = (perf_counter() - self._started) * 1000
                    self._store.append(self.turn_id, chunk, latency)
                    self._text += chunk
                    if self._first_chunk_ms is None:
                        self._first_chunk_ms = latency
                    return chunk
        except StopIteration:
            raise
        except BaseException:
            self.cancel()
            raise
        finally:
            self._consumer_lock.release()

    def request_cancel(self) -> bool:
        """Signal cancellation immediately, without waiting for a GPU kernel."""
        with self._lock:
            changed = self._status == "generating"
            self._stop.set()
            self._finish("interrupted")
            self._queue.put(None)
        return changed

    def cancel(self) -> bool:
        """Stop generation, discard undelivered text, and wait for GPU work to end."""
        changed = self.request_cancel()
        self._worker.join()
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        return changed

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Persist wrapper metadata such as ASR/TTS timing for this turn."""
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        with self._lock:
            self._metadata.update(metadata)
            self._store.merge_metadata(self.turn_id, metadata)

    def close(self) -> None:
        self.cancel()

    def __enter__(self) -> ResponseStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@dataclass
class _SessionState:
    gate: Any = field(default_factory=Lock)
    active: ResponseStream | None = None


class ContextManager:
    """SQLite-backed fast text conversation layer; reuse one loaded model.

    stream_chat starts immediately (not at the first next()). A new request for
    a session cancels its previous request, then reads the persisted history.
    Different sessions are isolated; Qwen3Client serializes GPU generation.
    """

    def __init__(
        self, client: Qwen3Client, *, db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self.client = client
        self._store = _SQLiteHistory(db_path)
        self.db_path = self._store.path
        self._registry = RLock()
        self._states: dict[str, _SessionState] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ContextManager is closed")

    def _state(self, session_id: str) -> _SessionState:
        with self._registry:
            self._ensure_open()
            self._store.session(session_id)
            return self._states.setdefault(session_id, _SessionState())

    def create_session(
        self, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT, session_id: str | None = None,
    ) -> str:
        """Create a session. Reuse its returned ID to resume it after restart."""
        session_id = uuid4().hex if session_id is None else session_id
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be text")
        with self._registry:
            self._ensure_open()
            try:
                self._store.create_session(session_id, system_prompt)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Session already exists: {session_id}") from exc
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._registry:
            self._ensure_open()
            return self._store.sessions()

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return persisted turns including partial text, status, and latency."""
        with self._registry:
            self._ensure_open()
            return self._store.turns(session_id)

    def _messages(self, session_id: str) -> list[dict[str, str]]:
        session = self._store.session(session_id)
        messages = [{"role": "system", "content": session["system_prompt"]}]
        for turn in self._store.turns(session_id):
            messages.append({"role": "user", "content": turn["user_text"]})
            reply = turn["assistant_text"]
            if reply:
                if turn["status"] == "interrupted":
                    reply += "\n[该回答在此处被打断，尚未完成。]"
                elif turn["status"] == "failed":
                    reply += "\n[该回答在此处因生成错误而停止。]"
                messages.append({"role": "assistant", "content": reply})
        return messages

    def stream_chat(self, session_id: str, prompt: str, *, thinking_mode: str | None = None,
                    **generation_options: Any) -> ResponseStream:
        """Submit text, interrupt this session's previous response, and stream a reply.

        Qwen3Client options (tools, max_input_tokens, max_new_tokens, temperature,
        top_p, top_k, repetition_penalty) pass through unchanged.
        """
        started = perf_counter()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        allowed = {"tools", "max_input_tokens", "max_new_tokens", "temperature", "top_p",
                   "top_k", "repetition_penalty"}
        unknown = generation_options.keys() - allowed
        if unknown:
            raise TypeError(f"Unsupported generation options: {sorted(unknown)}")
        if thinking_mode is not None:
            if not getattr(self.client, "supports_thinking_mode", False):
                raise TypeError("thinking_mode requires CooperativeClient")
            if thinking_mode not in {"auto", "fast", "slow"}:
                raise ValueError("thinking_mode must be auto, fast or slow")
            generation_options["thinking_mode"] = thinking_mode
        state = self._state(session_id)
        with state.gate:
            if state.active is not None:
                state.active.cancel()
            with self._registry:
                self._ensure_open()
                messages = self._messages(session_id)
                messages.append({"role": "user", "content": prompt})
                turn_id = self._store.begin_turn(session_id, prompt)
                response = ResponseStream(
                    self.client, self._store, turn_id, messages, generation_options, started,
                )
                state.active = response
                try:
                    response._worker.start()
                except BaseException as exc:
                    state.active = None
                    self._store.finish(turn_id, "failed", str(exc))
                    raise
                return response

    def chat(self, session_id: str, prompt: str, **generation_options: Any) -> str:
        """Collect a stream into a string; prefer streaming for visible low latency."""
        with self.stream_chat(session_id, prompt, **generation_options) as response:
            return "".join(response)

    def interrupt(self, session_id: str) -> bool:
        """Cancel an active response without submitting a new user message."""
        state = self._state(session_id)
        with state.gate:
            return state.active.cancel() if state.active is not None else False

    def request_interrupt(self, session_id: str) -> bool:
        state = self._state(session_id)
        with state.gate:
            return state.active.request_cancel() if state.active is not None else False

    def clear_history(self, session_id: str) -> None:
        """Explicitly delete this session's turns, preserving its system prompt."""
        state = self._state(session_id)
        with state.gate:
            if state.active is not None:
                state.active.cancel()
                state.active = None
            with self._registry:
                self._ensure_open()
                self._store.clear(session_id)

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a conversation, including its system prompt."""
        state = self._state(session_id)
        with state.gate:
            if state.active is not None:
                state.active.cancel()
                state.active = None
            with self._registry:
                self._ensure_open()
                self._store.delete_session(session_id)
                self._states.pop(session_id, None)

    def close(self) -> None:
        with self._registry:
            if self._closed:
                return
            self._closed = True
            states = list(self._states.values())
        for state in states:
            with state.gate:
                if state.active is not None:
                    state.active.cancel()
        self._store.close()

    def __enter__(self) -> ContextManager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["ContextManager", "ResponseStream", "DEFAULT_DB_PATH", "DEFAULT_SYSTEM_PROMPT"]
