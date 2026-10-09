"""Terminal text conversation: keep typing to interrupt a streamed reply."""

from __future__ import annotations

import argparse
from threading import Lock, Thread

from api import Qwen3Client
from context_manager import ContextManager, DEFAULT_DB_PATH, ResponseStream


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 SQLite 多轮对话与文本打断演示")
    parser.add_argument("--session", help="恢复已有 session_id；省略时新建会话")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite 文件路径")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("正在加载模型……", flush=True)
    client = Qwen3Client(device=args.device)
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
        context.get_history(session)  # Validate an explicitly supplied ID.
        print(f"session_id: {session}")
        print("输入消息并回车；回答过程中可继续输入新消息打断。")
        print("/stop 停止，/history 查看历史，/new 新会话，/exit 退出。")
        print("终端的键入文字可能与流式输出交错；每次回车才会提交新消息。")
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
                else:
                    response = context.stream_chat(session, prompt)
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
