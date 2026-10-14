"""Real CosyVoice waveform screening for 20 spoken insurance responses.

The text producer is fixed to isolate speech quality from LLM accuracy. Output
includes WAVs and numerical screening results; human listening remains separate.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from context_manager import ContextManager
from speech import TTSClient, save_audio_chunks
from voice_assistant import VoiceAssistant

CORPUS = [
    "一般医疗年免赔额为10000元。正常赔付比例为100%，年度限额为300万元。具体赔付仍以保险合同和审核结果为准。",
    "有医保，但没有按医保身份结算，赔付比例会调整。请先核对就诊情况，再确认适用的保险责任。",
    "重大疾病医疗责任没有免赔额。一般医疗责任的年免赔额为1万元，两类责任需要分别核对。",
    "康复医疗在不同机构的比例不同：指定康复机构为80%，公立医院普通部为60%。未使用医保结算时，还要核对比例调整。",
    "护工日标准为500元（每天）。单次住院最多给付30天，同一保险年度累计最多给付30天。",
    "年缴保费为318元，月缴保费为31元。请注意，基础费率与最终核保后的实际保费可能不同。",
    "费用计算为1000÷2，结果是500元。这里的小括号只保留其中的正文内容，不朗读括号名称。",
    "请先说明被保险人的年龄；再说明是否参加社保；最后选择年缴或月缴。这样才能匹配正确的费率档。",
    "不确定时，系统会提示转人工。当前是待人工处理状态，尚未接入人工客服，请不要将提示理解为已经接通人工。",
    "投保人和被保险人不是同一个概念。投保人需要满足资格要求，被保险人则需要满足产品规定的年龄范围。",
]


class TextClient:
    def __init__(self, answer):
        self.answer = answer

    def stream_chat(self, messages, **options):
        yield self.answer


def pitch_screen(samples, rate):
    frame, hop = round(rate * 0.04), round(rate * 0.01)
    low, high = int(rate / 400), int(rate / 80)
    track = []
    for start in range(0, len(samples) - frame, hop):
        y = samples[start:start + frame]
        if np.sqrt(np.mean(y * y)) < 250:
            continue
        y = (y - y.mean()) * np.hanning(len(y))
        ac = np.correlate(y, y, mode="full")[len(y) - 1:]
        ac /= ac[0] + 1e-9
        lag = low + int(ac[low:high].argmax())
        if ac[lag] > 0.35:
            track.append((start / rate, rate / lag))
    jumps = [abs(12 * np.log2(b[1] / a[1])) for a, b in zip(track, track[1:]) if b[0] - a[0] < 0.03]
    return {
        "f0_median_hz": round(float(np.median([item[1] for item in track])), 2) if track else None,
        "f0_jump_p95_semitones": round(float(np.percentile(jumps, 95)), 2) if jumps else None,
        "f0_jump_max_semitones": round(max(jumps), 2) if jumps else None,
        "pitch_method": "40ms autocorrelation, 10ms hop, 80-400Hz; octave errors may affect maxima",
    }


def main(
    destination: Path, source: Path, tts_url: str | None = None,
    sample_ids: set[int] | None = None,
):
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    texts = CORPUS.copy()
    if source.exists():
        actual = [json.loads(line) for line in source.read_text().splitlines()]
        texts.extend(item["answer"] for item in actual if item["mode"] == "slow" and item["pass"] and len(item["answer"]) > 40)
    if len(texts) < 20:
        texts.extend(CORPUS)
    tts = TTSClient(auto_start=False, device="cuda:0", server_url=tts_url)
    for i, text in enumerate(texts[:20], 1):
        if sample_ids is not None and i not in sample_ids:
            continue
        with TemporaryDirectory() as temp, ContextManager(TextClient(text), db_path=Path(temp) / "speech.sqlite3") as context:
            assistant = VoiceAssistant(context, None, tts)
            response = assistant.stream_text_chat(context.create_session(), "语音质量验收")
            chunks = list(response)
            response.close()
            audio = save_audio_chunks(chunks, destination / f"sample-{i:02d}.wav")
            samples = np.frombuffer(b"".join(c.pcm16 for c in chunks), dtype="<i2").astype(float)
            rate = chunks[0].sample_rate
            frame = round(rate * 0.02)
            rms = np.array([np.sqrt(np.mean(samples[p:p+frame] ** 2)) for p in range(0, len(samples)-frame, frame)])
            active = rms > 100
            split = max(1, int(len(rms) * 0.8))
            head = rms[:split][active[:split]]
            tail = rms[split:][active[split:]]
            tail_delta = 20 * np.log10(np.median(tail) / np.median(head)) if len(head) and len(tail) else None
            pauses = [c.duration_ms for c in chunks if not c.text and not any(c.pcm16)]
            segments = list(dict.fromkeys(c.text for c in chunks if c.text))
            rates = []
            for segment in segments:
                seconds = sum(c.duration_ms for c in chunks if c.text == segment) / 1000
                rates.append(round(len(re.findall(r"[\u4e00-\u9fffA-Za-z]", segment)) / seconds, 2))
            row = {
                "id": i, "text": text, "wav": str(audio.relative_to(ROOT)),
                "first_audio_ms": response.first_audio_ms,
                "voice_metadata": response.metadata.get("voice", {}),
                "explicit_intersegment_pauses_ms": pauses,
                "tail_vs_head_active_median_db": round(float(tail_delta), 2) if tail_delta is not None else None,
                "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 32767)),
                "speed_setting": tts.speed,
                "segment_characters_per_second": rates,
                "max_adjacent_segment_rate_ratio": round(max((max(a,b) / min(a,b) for a,b in zip(rates,rates[1:]) if min(a,b)>0),default=1), 2),
                **pitch_screen(samples, rate),
            }
            rows.append(row)
            (destination.parent / "speech-results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"speech {i}/20: tail delta={row['tail_vs_head_active_median_db']}dB; pitch p95={row['f0_jump_p95_semitones']}; pauses={pauses}", flush=True)
    print(f"RESULT: {len(rows)} real waveform samples saved; listening judgment is not inferred from numerical screening.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exports/acceptance/final/audio")
    parser.add_argument("--answers", type=Path, default=ROOT / "exports/acceptance-final-200.jsonl")
    parser.add_argument("--tts-url", default=None, help="Local CosyVoice test service URL")
    parser.add_argument("--only", type=int, action="append", help="Only synthesize selected 1-based sample ID; repeatable")
    args = parser.parse_args()
    main(args.output_dir, args.answers, args.tts_url, set(args.only) if args.only else None)
