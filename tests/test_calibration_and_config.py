from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import numpy as np

from protobridge.config import AppConfig
from protobridge.diarization.speaker_id import OnlineSpeakerTracker


class TestCalibrationAndConfig(unittest.TestCase):
    def test_pitch_estimation_accuracy(self) -> None:
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        target_f0 = 145.0
        tone = np.sin(2.0 * np.pi * target_f0 * t).astype(np.float32)

        estimated_f0 = OnlineSpeakerTracker._estimate_pitch_hz(tone)
        self.assertAlmostEqual(estimated_f0, target_f0, delta=2.5)

    def test_config_roundtrip_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_cfg_path = Path(tmp_dir) / "config.json"
            cfg = AppConfig.load(tmp_cfg_path)
            self.assertEqual(cfg.my_voice_profile, "am_adam")
            cfg.my_voice_profile = "my_cloned"
            cfg.tts_speed_out = 1.25
            cfg.monitor_own_voice = True
            cfg.save(tmp_cfg_path)

            reloaded = AppConfig.load(tmp_cfg_path)
            self.assertEqual(reloaded.my_voice_profile, "my_cloned")
            self.assertEqual(reloaded.tts_speed_out, 1.25)
            self.assertTrue(reloaded.monitor_own_voice)


if __name__ == "__main__":
    unittest.main()
