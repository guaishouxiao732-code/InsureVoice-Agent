"""Bounded 16 kHz PCM stream segmentation using local Silero VAD."""
from pathlib import Path

import numpy as np
import sherpa_onnx


VAD_MODEL = Path(__file__).parent / "models/speech/vad/silero_vad.onnx"


class LiveVAD:
    def __init__(self, model: Path = VAD_MODEL):
        if not model.is_file():
            raise FileNotFoundError(f"缺少本地 VAD 模型：{model}")
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model)
        config.sample_rate = 16000
        config.num_threads = 1
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.min_silence_duration = 0.6
        # Silero can include trailing silence beyond this duration. Leave headroom
        # below SenseVoice's hard 30-second input limit rather than failing ASR.
        config.silero_vad.max_speech_duration = 25
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=40)
        self.speaking = False

    def feed(self, pcm: bytes):
        if not pcm or len(pcm) % 2 or len(pcm) > 6400:
            raise ValueError("音频帧必须是至多 200 ms 的单声道 16 kHz PCM16")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.vad.accept_waveform(samples)
        if len(self.vad.current_segment.samples) >= 28 * 16000:
            self.vad.flush()
        detected = self.vad.is_speech_detected()
        started = detected and not self.speaking
        segments = []
        while not self.vad.empty():
            segments.append(np.array(self.vad.front.samples, dtype=np.float32, copy=True))
            self.vad.pop()
        self.speaking = detected
        return started, segments

    def snapshot(self):
        return np.array(self.vad.current_segment.samples, dtype=np.float32, copy=True)

    def reset(self):
        self.vad.reset()
        self.speaking = False
