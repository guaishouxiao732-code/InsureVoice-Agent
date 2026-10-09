"""Terminal RAG conversation over the local insurance knowledge base."""

from __future__ import annotations

import argparse
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from api import Qwen3Client
from context_manager import ContextManager, ResponseStream
from cooperative import CooperativeClient
from rag import DEFAULT_INDEX_DIR, RAGClient, RAGRetriever


DEFAULT_RAG_DB_PATH = Path(__file__).resolve().parent / "data" / "rag_conversations.sqlite3"


def _print_sources(metadata: dict[str, Any]) -> None:
    if not metadata:
        print("当前还没有 RAG 检索记录。")
        return
    mode = metadata.get("knowledge_mode", "unknown")
    print(
        f"知识模式：{mode}；检索耗时："
        f"{metadata.get('retrieval_ms', 0):.1f} ms；"
        f"最高相似度：{metadata.get('top_score')}；"
        f"最大词面覆盖：{metadata.get('max_lexical_overlap')}"
    )
    sources = metadata.get("sources", [])
    if not sources:
        print("无候选片段。")
        return
    label = "引用来源" if mode == "rag" else "未采用的候选片段"
    print(f"{label}：")
    for number, source in enumerate(sources, start=1):
        preview = " ".join(source.get("text", "").split())[:100]
        source_label = source.get("source_name") or f"PDF第{source['pdf_page']}页"
        location = source.get("location", "")
        if location:
            source_label = f"{source_label}，{location}"
        print(
            f"  {number}. {source_label} "
            f"(相似度={source['score']:.3f}, "
            f"词面覆盖={source['lexical_overlap']:.3f}) {preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 + 保险资料 RAG + SQLite 上下文终端")
    parser.add_argument("--session", help="恢复已有 session_id；省略时新建会话")
    parser.add_argument("--db", default=str(DEFAULT_RAG_DB_PATH), help="SQLite 文件路径")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="FAISS 索引目录")
    parser.add_argument("--device", default="cuda:1", help="Qwen 设备")
    parser.add_argument("--embedding-device", default="cpu", help="BGE 设备")
    parser.add_argument("--mode", choices=("auto", "fast", "slow"), default="auto", help="自动协同或手动调试模式")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.47)
    parser.add_argument("--min-lexical-overlap", type=float, default=0.08)
    args = parser.parse_args()

    print("正在加载 BGE 检索器和 Qwen 模型……", flush=True)
    retriever = RAGRetriever(
        index_dir=args.index,
        embedding_device=args.embedding_device,
        top_k=args.top_k,
        min_score=args.min_score,
        min_lexical_overlap=args.min_lexical_overlap,
    )
    qwen = Qwen3Client(device=args.device)
    client = CooperativeClient(RAGClient(qwen, retriever))
    output_lock = Lock()
    consumers: list[Thread] = []

    def display(response: ResponseStream) -> None:
        try:
            with response:
                for chunk in response:
                    with output_lock:
                        print(chunk, end="", flush=True)
            with output_lock:
                print(f"\n[{response.status}]", flush=True)
        except Exception as exc:
            with output_lock:
                print(f"\n生成失败：{exc}", flush=True)

    with ContextManager(client, db_path=args.db) as context:
        session = args.session or context.create_session()
        context.get_history(session)
        print(f"session_id: {session}")
        print("输入保险咨询问题；系统先检索，证据充分时调用 Qwen，证据不足时提示待人工处理。")
        print("/sources 查看本轮证据，/history 查看历史，/new 新会话，/stop 停止，/exit 退出。")
        try:
            while True:
                try:
                    prompt = input().strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt:
                    continue
                if prompt == "/exit":
                    break
                if prompt == "/stop":
                    context.interrupt(session)
                elif prompt == "/new":
                    context.interrupt(session)
                    session = context.create_session()
                    print(f"session_id: {session}", flush=True)
                elif prompt == "/history":
                    with output_lock:
                        for turn in context.get_history(session):
                            print(f"用户：{turn['user_text']}")
                            print(f"助手 [{turn['status']}]：{turn['assistant_text']}")
                elif prompt == "/sources":
                    history = context.get_history(session)
                    with output_lock:
                        _print_sources(history[-1]["metadata"] if history else {})
                else:
                    response = context.stream_chat(session, prompt, thinking_mode=args.mode)
                    consumer = Thread(target=display, args=(response,), daemon=True)
                    consumers = [thread for thread in consumers if thread.is_alive()]
                    consumers.append(consumer)
                    consumer.start()
        finally:
            context.interrupt(session)
            for consumer in consumers:
                consumer.join()


if __name__ == "__main__":
    main()
