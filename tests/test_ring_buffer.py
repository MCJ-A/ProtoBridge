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

    def test_overflow_does_not_mutate_producer_read_cursor_and_read_all_clamps(self) -> None:
        rb = SPSCRingBuffer(
            name="test_rb_ci_overflow", capacity_seconds=0.1, sample_rate=1000, create=True
        )
        try:
            self.assertEqual(rb.capacity, 100)
            chunk1 = np.arange(80, dtype=np.float32)
            rb.write(chunk1)
            # Provocar overflow de 50 muestras sin que el consumidor haya avanzado _header[1]
            chunk2 = np.arange(80, 150, dtype=np.float32)
            rb.write(chunk2)

            # Invariante SPSC: el productor nunca debe mutar _header[1]
            self.assertEqual(int(rb._header[1]), 0)
            self.assertEqual(rb.write_cursor, 150)
            self.assertGreaterEqual(int(rb._header[2]), 1)

            # El consumidor debe clampear a los últimos `capacity` elementos (50..149)
            self.assertEqual(rb.read_cursor, 50)
            self.assertEqual(rb.available_to_read, 100)
            st = rb.stats()
            self.assertEqual(st.available_read, 100)
            self.assertEqual(st.read_cursor, 50)

            drained = rb.read_all()
            np.testing.assert_array_equal(drained, np.arange(50, 150, dtype=np.float32))
            self.assertEqual(rb.available_to_read, 0)
            self.assertEqual(int(rb._header[1]), 150)
            # read_all con buffer vacío devuelve arreglo vacío
            self.assertEqual(rb.read_all().shape, (0,))
        finally:
            rb.close()

    def test_close_releases_views_cleanly_for_reconnect(self) -> None:
        for _ in range(3):
            rb = SPSCRingBuffer(
                name="test_rb_ci_reconn", capacity_seconds=0.05, sample_rate=1000, create=True
            )
            rb.write(np.ones(16, dtype=np.float32))
            _ = rb.read_all()
            rb.close()
            # Segunda invocación debe ser idempotente sin BufferError
            rb.close()


if __name__ == "__main__":
    unittest.main()
