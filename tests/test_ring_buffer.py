from __future__ import annotations

import unittest
import numpy as np
from protobridge.audio.ring_buffer import SPSCRingBuffer


class TestSPSCRingBuffer(unittest.TestCase):
    def test_write_read_and_wraparound(self) -> None:
        rb = SPSCRingBuffer(
            name="test_rb_ci_1", capacity_seconds=0.1, sample_rate=16000, create=True
        )
        try:
            self.assertEqual(rb.capacity, 1600)
            chunk = np.arange(1000, dtype=np.float32)
            rb.write(chunk)
            self.assertEqual(rb.available_to_read, 1000)

            out = np.empty((600,), dtype=np.float32)
            read_n = rb.read_into(out)
            self.assertEqual(read_n, 600)
            np.testing.assert_array_equal(out, np.arange(600, dtype=np.float32))

            # Forzar wrap-around circular escribiendo 1000 muestras adicionales
            chunk2 = np.ones((1000,), dtype=np.float32) * 7.0
            rb.write(chunk2)
            self.assertEqual(rb.available_to_read, 1400)
            self.assertEqual(rb.stats().overruns, 0)
        finally:
            rb.close()

    def test_preroll_lookback_window(self) -> None:
        rb = SPSCRingBuffer(
            name="test_rb_ci_2", capacity_seconds=1.0, sample_rate=16000, create=True
        )
        try:
            data = np.linspace(-1.0, 1.0, 8000, dtype=np.float32)
            rb.write(data)
            # Extraer ventana [5000, 8000] con 4800 muestras (300 ms) de pre-roll -> [200, 8000]
            window = rb.slice_window(start_cursor=5000, end_cursor=8000, preroll_samples=4800)
            self.assertEqual(window.shape[0], 7800)
            np.testing.assert_allclose(window, data[200:8000])
        finally:
            rb.close()


if __name__ == "__main__":
    unittest.main()
