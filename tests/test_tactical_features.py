from __future__ import annotations

import time
import unittest
import numpy as np

from ignait.audio.streams import (
    TACTICAL_CHIRP_PCM,
    AsyncOutputWorker,
    PlaybackItem,
    generate_tactical_chirp,
)
from ignait.config import DEFAULT_GAMING_GLOSSARY, AppConfig
from ignait.control.hotkeys import GlobalHotkeyListener, parse_hotkey_spec
from ignait.mt.translator import apply_glossary, compile_glossary


class TestTacticalFeatures(unittest.TestCase):
    """Suite de validación operativa y unitaria para las características tácticas de Ignait."""

    def test_global_hotkey_state_transitions_and_callbacks(self) -> None:
        """Valida transiciones de estado de Mute TX/RX y PTT sin bloquear el hilo."""
        tx_events: list[bool] = []
        rx_events: list[bool] = []
        ptt_events: list[bool] = []

        listener = GlobalHotkeyListener(
            hotkey_mute_tx="ctrl+shift+m",
            hotkey_mute_rx="ctrl+shift+n",
            hotkey_ptt="caps_lock",
            ptt_enabled=True,
            initial_mute_tx=False,
            initial_mute_rx=False,
            on_tx_mute_changed=lambda s: tx_events.append(s),
            on_rx_mute_changed=lambda s: rx_events.append(s),
            on_ptt_changed=lambda s: ptt_events.append(s),
        )

        self.assertFalse(listener.is_tx_muted)
        self.assertFalse(listener.is_rx_muted)
        self.assertFalse(listener.is_ptt_active)

        # Transición Mute TX
        toggled_tx = listener.toggle_mute_tx()
        self.assertTrue(toggled_tx)
        self.assertTrue(listener.is_tx_muted)
        self.assertEqual(tx_events, [True])

        listener.set_mute_tx(False)
        self.assertFalse(listener.is_tx_muted)
        self.assertEqual(tx_events, [True, False])

        # Transición Mute RX
        toggled_rx = listener.toggle_mute_rx()
        self.assertTrue(toggled_rx)
        self.assertTrue(listener.is_rx_muted)
        self.assertEqual(rx_events, [True])

        # Transición PTT
        listener.set_ptt_active(True)
        self.assertTrue(listener.is_ptt_active)
        listener.set_ptt_active(False)
        self.assertFalse(listener.is_ptt_active)
        self.assertEqual(ptt_events, [True, False])

        # Ciclo de vida del hilo de escucha (inicio y parada no bloqueante)
        t_start = time.perf_counter()
        listener.start()
        time.sleep(0.05)
        listener.stop()
        t_stop = time.perf_counter()
        self.assertLess(t_stop - t_start, 0.5, "El arranque y detención de hotkeys no debe bloquear.")

    def test_hotkey_spec_parsing(self) -> None:
        """Verifica el mapeo de especificaciones de teclado a Virtual Key codes."""
        vks = parse_hotkey_spec("ctrl+shift+m")
        self.assertEqual(len(vks), 3)
        self.assertIn(0x11, vks)  # Ctrl
        self.assertIn(0x10, vks)  # Shift
        self.assertIn(ord("M"), vks)

        ptt_vks = parse_hotkey_spec("caps_lock")
        self.assertEqual(ptt_vks, (0x14,))

        mouse_vks = parse_hotkey_spec("mouse4")
        self.assertEqual(mouse_vks, (0x05,))

    def test_async_output_worker_flush_and_barge_in(self) -> None:
        """Valida que flush() vacíe atómicamente la cola FIFO y descarte chunks pendientes."""
        # Se inicializa el worker sin hardware WASAPI
        worker = AsyncOutputWorker(player=None)

        try:
            dummy_pcm = np.zeros(512, dtype=np.float32)
            # Encolar 6 chunks
            for _ in range(6):
                worker.enqueue(dummy_pcm, sample_rate=16000)

            # Debe haber elementos pendientes en cola o procesándose
            discarded = worker.flush()
            self.assertGreaterEqual(discarded, 0)
            self.assertEqual(worker.pending_items, 0, "La cola debe quedar vacía tras flush().")

            # Comprobar que encolar tras flush() funcione normalmente
            worker.enqueue(dummy_pcm, sample_rate=16000)
            discarded_second = worker.flush()
            self.assertEqual(worker.pending_items, 0)
        finally:
            worker.close()

    def test_tactical_chirp_properties_and_normalization(self) -> None:
        """Valida que TACTICAL_CHIRP_PCM cumpla rigurosamente las restricciones acústicas y de forma."""
        # 16 kHz * 0.020 s = 320 muestras
        self.assertEqual(TACTICAL_CHIRP_PCM.shape, (320,))
        self.assertEqual(TACTICAL_CHIRP_PCM.dtype, np.float32)

        # Rango numérico estricto [-1.0, 1.0]
        self.assertTrue(np.all(TACTICAL_CHIRP_PCM >= -1.0))
        self.assertTrue(np.all(TACTICAL_CHIRP_PCM <= 1.0))

        # Normalización pico a ~1.0
        peak = float(np.max(np.abs(TACTICAL_CHIRP_PCM)))
        self.assertAlmostEqual(peak, 1.0, delta=1e-3)

        # Hanning window suaviza los extremos a cero
        self.assertAlmostEqual(float(TACTICAL_CHIRP_PCM[0]), 0.0, delta=0.05)
        self.assertAlmostEqual(float(TACTICAL_CHIRP_PCM[-1]), 0.0, delta=0.05)

        # Prueba parametrizada
        custom_chirp = generate_tactical_chirp(sample_rate=16000, duration_seconds=0.010)
        self.assertEqual(custom_chirp.shape, (160,))
        self.assertEqual(custom_chirp.dtype, np.float32)

    def test_glossary_compilation_and_term_replacement(self) -> None:
        """Valida la compilación regex ordenada y la sustitución determinista de términos tácticos."""
        custom_terms = {
            "bile titan": "titán de bilis",
            "titan": "gigante",
            "one shot": "a un toque",
            "rush": "rusheo",
            "retreat": "retirada táctica",
        }

        compiled = compile_glossary(custom_terms)
        self.assertEqual(len(compiled), 5)

        # "bile titan" (longitud 10) debe preceder a "titan" (longitud 5)
        terms_order = [p.pattern for p, _ in compiled]
        idx_bile = next(i for i, p in enumerate(terms_order) if "bile" in p)
        idx_titan = next(i for i, p in enumerate(terms_order) if "titan" in p and "bile" not in p)
        self.assertLess(idx_bile, idx_titan, "Términos más largos deben compilarse antes para evitar eclipsamiento.")

        # Reemplazos en oraciones complejas
        text1 = "Watch out, there is a bile titan on the hill and he is one shot!"
        out1 = apply_glossary(text1, compiled)
        self.assertIn("titán de bilis", out1)
        self.assertIn("a un toque", out1)
        self.assertNotIn("bile titan", out1)
        self.assertNotIn("one shot", out1)

        # Soporte para plurales automáticos: "titans" -> "titán de bilis"
        text2 = "Two bile titans incoming!"
        out2 = apply_glossary(text2, compiled)
        self.assertIn("titán de bilis", out2)

        # Mayúsculas y minúsculas
        text3 = "RUSH B now, do not RETREAT!"
        out3 = apply_glossary(text3, compiled)
        self.assertIn("rusheo", out3)
        self.assertIn("retirada táctica", out3)

    def test_app_config_tactical_defaults(self) -> None:
        """Verifica que AppConfig cargue con los campos tácticos por defecto."""
        cfg = AppConfig()
        self.assertEqual(cfg.hotkey_mute_tx, "ctrl+shift+m")
        self.assertEqual(cfg.hotkey_mute_rx, "ctrl+shift+n")
        self.assertEqual(cfg.hotkey_ptt, "caps_lock")
        self.assertFalse(cfg.ptt_enabled)
        self.assertTrue(cfg.tactical_chirp_enabled)
        self.assertTrue(cfg.barge_in_enabled)
        self.assertIn("rush", cfg.custom_glossary)
        self.assertEqual(cfg.custom_glossary["rush"], "rusheo")


if __name__ == "__main__":
    unittest.main()
