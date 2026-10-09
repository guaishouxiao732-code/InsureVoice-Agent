"""Stateless local Python interface for Qwen3-4B-Instruct-2507."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from queue import Empty
from threading import Event, Lock, Thread
from typing import Any

from project_config import PROJECT_ROOT, project_path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


DEFAULT_MODEL_PATH = project_path("QWEN_MODEL_PATH", "models/Qwen3-4B-Instruct-2507")
DEFAULT_MAX_INPUT_TOKENS = 32_768

Message = dict[str, Any]
Tool = dict[str, Any]


class _CancelGeneration(StoppingCriteria):
    def __init__(self, event: Event) -> None:
        self.event = event

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        return self.event.is_set()


class Qwen3Client:
    """Reusable local inference client for Qwen3-4B-Instruct-2507.

    The client is stateless: every call must receive the complete ``messages``
    history. If a rendered prompt exceeds ``max_input_tokens``, the oldest
    non-system messages before the latest user turn are removed first.
    """

    _DTYPES: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = attn_implementation
        self.max_input_tokens = self._validate_positive_int(
            "max_input_tokens", max_input_tokens
        )
        self.local_files_only = local_files_only
        self._generation_lock = Lock()

        self._validate_runtime()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": str(self.device)},
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model_max_context = int(self.model.config.max_position_embeddings)

    @classmethod
    def _resolve_dtype(cls, dtype: str) -> torch.dtype | str:
        try:
            return cls._DTYPES[dtype]
        except KeyError as exc:
            choices = ", ".join(cls._DTYPES)
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from: {choices}") from exc

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _validate_runtime(self) -> None:
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_path}")

        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")

        index = self.device.index if self.device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {self.device}, but only {torch.cuda.device_count()} CUDA device(s) exist"
            )
        with torch.cuda.device(index):
            if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support bfloat16")

    @staticmethod
    def build_messages(
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> list[Message]:
        """Build a single-turn request in Hugging Face chat format."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")

        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _render_and_tokenize(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
    ) -> Any:
        template_kwargs: dict[str, Any] = {}
        if tools is not None:
            template_kwargs["tools"] = list(tools)

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    @staticmethod
    def _oldest_removable_span(
        messages: Sequence[Message],
    ) -> tuple[int, int] | None:
        """Find the oldest complete turn while preserving system prompts and latest turn."""
        latest_user = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            len(messages) - 1,
        )
        for start, message in enumerate(messages):
            if start >= latest_user or message.get("role") == "system":
                continue
            end = next(
                (
                    i
                    for i in range(start + 1, latest_user + 1)
                    if messages[i].get("role") == "user"
                ),
                latest_user,
            )
            return start, end
        return None

    def _prepare_inputs(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None,
        max_input_tokens: int | None,
        max_new_tokens: int,
    ) -> Any:
        if not messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, dict) for message in messages):
            raise TypeError("every message must be a dictionary")

        limit = self.max_input_tokens if max_input_tokens is None else self._validate_positive_int(
            "max_input_tokens", max_input_tokens
        )
        available_context = self.model_max_context - max_new_tokens
        if available_context <= 0:
            raise ValueError(
                f"max_new_tokens must be smaller than the model context ({self.model_max_context})"
            )
        if limit > available_context:
            raise ValueError(
                f"max_input_tokens ({limit}) plus max_new_tokens ({max_new_tokens}) exceeds "
                f"the model context ({self.model_max_context})"
            )

        prepared_messages = list(messages)
        while True:
            inputs = self._render_and_tokenize(prepared_messages, tools)
            token_count = int(inputs["input_ids"].shape[1])
            if token_count <= limit:
                return inputs.to(self.device)

            removable_span = self._oldest_removable_span(prepared_messages)
            if removable_span is None:
                raise ValueError(
                    f"The protected system prompt, latest user turn, and tool definitions use "
                    f"{token_count} tokens, exceeding max_input_tokens={limit}. Shorten them or "
                    "increase max_input_tokens."
                )
            start, end = removable_span
            prepared_messages[start:end] = [
                message for message in prepared_messages[start:end]
                if message.get("role") == "system"
            ]

    @staticmethod
    def _generation_kwargs(
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> dict[str, Any]:
        Qwen3Client._validate_positive_int("max_new_tokens", max_new_tokens)
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
        }
        if temperature > 0:
            kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            kwargs.update(
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        return kwargs

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        max_input_tokens: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate and return one complete response."""
        generation_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        inputs = self._prepare_inputs(
            messages,
            tools=tools,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def warmup(self) -> None:
        """Run one unpersisted token so the first real request avoids cold kernels."""
        messages = self.build_messages("你好")
        with closing(self.stream_chat(messages, max_new_tokens=1, temperature=0)) as stream:
            for _ in stream:
                pass

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Tool] | None = None,
        max_input_tokens: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        cancel_event: Event | None = None,
    ) -> Iterator[str]:
        """Yield text chunks; setting cancel_event or closing the iterator stops generation.

        Cancellation is cooperative between decoding steps. It cannot interrupt
        an in-flight GPU kernel or prompt prefill. Closing an unfinished iterator
        sets the event; a naturally completed generation leaves a caller-supplied
        event unset.
        """
        generation_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        stop = cancel_event if cancel_event is not None else Event()
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            timeout=0.1,
        )
        errors: list[BaseException] = []

        def generate() -> None:
            acquired = False
            try:
                while not stop.is_set():
                    acquired = self._generation_lock.acquire(timeout=0.05)
                    if acquired:
                        break
                if not acquired or stop.is_set():
                    return
                inputs = self._prepare_inputs(
                    messages,
                    tools=tools,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                if stop.is_set():
                    return
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        **generation_kwargs,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_CancelGeneration(stop)]),
                    )
            except BaseException as exc:
                errors.append(exc)
            finally:
                if acquired:
                    self._generation_lock.release()
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=generate, name="qwen3-generation", daemon=True)
        worker.start()
        reached_stream_end = False
        try:
            while not stop.is_set():
                try:
                    chunk = next(streamer)
                except Empty:
                    continue
                except StopIteration:
                    reached_stream_end = True
                    break
                if stop.is_set():
                    break
                if chunk:
                    yield chunk
        finally:
            # Do not turn natural completion into cancellation. Wrappers such
            # as RAG may still need to yield deterministic source metadata
            # after the model's text stream has ended.
            if not reached_stream_end:
                stop.set()
            worker.join()

        if errors:
            raise RuntimeError("Qwen3 streaming generation failed") from errors[0]


__all__ = ["DEFAULT_MAX_INPUT_TOKENS", "DEFAULT_MODEL_PATH", "Qwen3Client"]
