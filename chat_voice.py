"""One-request local voice demo: audio -> ASR -> RAG/Qwen -> streamed TTS."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from api import Qwen3Client
from context_manager import ContextManager
from cooperative import CooperativeClient
from rag import RAGClient, RAGRetriever
from speech import ASRClient, TTSClient, save_audio_chunks
from voice_assistant import VoiceAssistant


DEFAULT_VOICE_DB = Path(__file__).resolve().parent / "data" / "voice_conversations.sqlite3"


def _corrections(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"correction must use WRONG=RIGHT: {value}")
        source, replacement = value.split("=", 1)
        if not source:
            raise ValueError("correction source cannot be empty")
        result[source] = replacement
    return result


def _start_player(sample_rate: int) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            [
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
                "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("--play requires ffplay on PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 ASR + RAG/Qwen + TTS 语音问答")
    parser.add_argument("audio", help="不超过30秒的录音文件")
    parser.add_argument("--output", help="显式保存完整 TTS WAV；默认不保存")
    parser.add_argument("--play", action="store_true", help="通过 ffplay 边合成边播放")
    parser.add_argument("--session", help="恢复已有 session_id")
    parser.add_argument("--db", default=str(DEFAULT_VOICE_DB))
    parser.add_argument("--device", default="cuda:1", help="Qwen 设备")
    parser.add_argument("--tts-device", default="cuda:0", help="CosyVoice 设备")
    parser.add_argument("--speaker", default="中文女", help="CosyVoice 预置说话人")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, help="显式覆盖协同路径的默认生成预算")
    parser.add_argument("--mode", choices=("auto", "fast", "slow"), default="auto")
    parser.add_argument(
        "--correction", action="append", default=[], metavar="WRONG=RIGHT",
        help="ASR 术语纠错，可重复传入",
    )
    args = parser.parse_args()

    print("正在预加载 ASR、TTS、RAG 和 Qwen……", flush=True)
    asr = ASRClient()
    tts = TTSClient(speaker=args.speaker, speed=args.speed, device=args.tts_device)
    retriever = RAGRetriever(embedding_device="cpu")
    qwen = Qwen3Client(device=args.device)
    qwen.warmup()
    tts.warmup()
    rag = CooperativeClient(RAGClient(qwen, retriever))
    player: subprocess.Popen[bytes] | None = None
    chunks = []

    with ContextManager(rag, db_path=args.db) as context:
        session = args.session or context.create_session()
        context.get_history(session)
        voice = VoiceAssistant(context, asr, tts)
        response = voice.stream_chat(
            session,
            args.audio,
            corrections=_corrections(args.correction),
            thinking_mode=args.mode,
            **({"max_new_tokens": args.max_new_tokens} if args.max_new_tokens is not None else {}),
        )
        print(f"session_id: {session}")
        print(f"ASR：{response.transcript}")
        try:
            with response:
                for chunk in response:
                    if args.output:
                        chunks.append(chunk)
                    if args.play:
                        if player is None:
                            player = _start_player(chunk.sample_rate)
                        assert player.stdin is not None
                        player.stdin.write(chunk.pcm16)
                        player.stdin.flush()
            print(f"\n助手文字：\n{response.text}")
            print(f"\n语音指标：{response.metadata.get('voice', {})}")
            if args.output:
                saved = save_audio_chunks(chunks, args.output)
                print(f"音频已保存：{saved}")
        except KeyboardInterrupt:
            voice.interrupt()
            print("\n已打断。")
        finally:
            if player is not None:
                if player.stdin is not None:
                    player.stdin.close()
                player.wait(timeout=10)


if __name__ == "__main__":
    main()
