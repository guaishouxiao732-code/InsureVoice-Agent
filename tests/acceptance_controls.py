"""Repeatable backend cancellation check without browser or GPU dependencies.

Run with: python tests/acceptance_controls.py
This measures server-side text-stream cancellation, not physical browser audio stop.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from context_manager import ContextManager


class CancellableClient:
    def stream_chat(self, messages, *, cancel_event=None, **_options):
        def chunks():
            yield "已输出的回答前缀。"
            while cancel_event is not None and not cancel_event.wait(0.01):
                pass
            if cancel_event is None or not cancel_event.is_set():
                yield "这段内容不应在打断后继续输出。"

        return chunks()


def run(repetitions: int = 100) -> tuple[int, list[float]]:
    successes = 0
    latencies_ms: list[float] = []
    with TemporaryDirectory(prefix="fyx-interrupt-acceptance-") as temp_dir:
        db_path = Path(temp_dir) / "history.sqlite3"
        with ContextManager(CancellableClient(), db_path=db_path) as context:
            for _ in range(repetitions):
                session = context.create_session()
                response = context.stream_chat(session, "请给出一段较长的解释。")
                first = next(response)
                started = perf_counter()
                accepted = context.request_interrupt(session)
                latencies_ms.append((perf_counter() - started) * 1000)
                tail = "".join(response)
                if accepted and first and response.status == "interrupted" and not tail:
                    successes += 1
                response.close()
    return successes, latencies_ms


if __name__ == "__main__":
    import statistics

    passed, samples = run()
    ordered = sorted(samples)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    rate = passed / len(samples) * 100 if samples else 0
    print(f"backend_interrupt={passed}/{len(samples)} ({rate:.1f}%)")
    print(f"request_interrupt latency ms: median={statistics.median(samples):.3f}, p95={p95:.3f}, max={max(samples):.3f}")
