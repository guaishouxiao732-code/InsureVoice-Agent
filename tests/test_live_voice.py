import subprocess
import shutil
import unittest
from pathlib import Path

import numpy as np

from live_voice import LiveVAD, VAD_MODEL


SAMPLE_AUDIO = Path(__file__).resolve().parents[1] / "data/speech_samples/input_question.wav"
HAS_AUDIO_FIXTURE = SAMPLE_AUDIO.is_file() and shutil.which("ffmpeg") is not None


@unittest.skipUnless(VAD_MODEL.is_file(), "可选旧 VAD 实测需要本机 Silero 权重；源码包不含权重")
class LiveVADTests(unittest.TestCase):
    def test_silence_and_malformed_frames(self):
        vad = LiveVAD()
        for _ in range(30):
            self.assertEqual(vad.feed(bytes(3200)), (False, []))
        for pcm in (b"", b"x", bytes(6402)):
            with self.assertRaises(ValueError):
                vad.feed(pcm)

    @unittest.skipUnless(HAS_AUDIO_FIXTURE, "可选旧 VAD 实测需要私有录音样例和 FFmpeg")
    def test_real_speech_is_segmented_and_reset_discards_audio(self):
        source = Path(__file__).resolve().parents[1] / "data/speech_samples/input_question.wav"
        audio = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
             "-f", "s16le", "pipe:1"], check=True, capture_output=True,
        ).stdout
        vad = LiveVAD()
        starts, segments = 0, []
        pcm = bytes(16000) + audio + bytes(32000)
        for offset in range(0, len(pcm), 3200):
            started, completed = vad.feed(pcm[offset:offset + 3200])
            starts += int(started)
            segments.extend(completed)
        self.assertGreaterEqual(starts, 1)
        self.assertGreaterEqual(len(segments), 1)
        self.assertGreater(sum(len(s) for s in segments), 4000)
        self.assertTrue(all(s.dtype == np.float32 for s in segments))
        vad.reset()
        self.assertEqual(len(vad.snapshot()), 0)

    @unittest.skipUnless(HAS_AUDIO_FIXTURE, "可选旧 VAD 实测需要私有录音样例和 FFmpeg")
    def test_long_speech_stays_below_asr_limit(self):
        source = Path(__file__).resolve().parents[1] / "data/speech_samples/input_question.wav"
        audio = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
             "-f", "s16le", "pipe:1"], check=True, capture_output=True,
        ).stdout * 25 + bytes(32000)
        vad = LiveVAD()
        segments = []
        for offset in range(0, len(audio), 3200):
            _, completed = vad.feed(audio[offset:offset + 3200])
            segments.extend(completed)
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(len(s) <= 30 * 16000 for s in segments))


if __name__ == "__main__":
    unittest.main()
