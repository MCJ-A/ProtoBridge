from __future__ import annotations

import time
import pyaudiowpatch as pyaudio

from ignait.audio.devices import WASAPIDeviceManager
from ignait.audio.ring_buffer import SPSCRingBuffer
from ignait.audio.streams import WASAPIInputStream, VAD_FRAME_SAMPLES


def run_diagnostics() -> None:
    manager = WASAPIDeviceManager()
    try:
        endpoints = manager.list_wasapi_endpoints()
        topology = manager.auto_detect_topology()

        print("=" * 78)
        print(" IGNAIT - FASE 1: DIAGNOSTICO DE BUSES WASAPI (WINDOWS)")
        print("=" * 78)
        for ep in endpoints:
            flags: list[str] = []
            if ep.is_default_input:
                flags.append("DEFAULT_IN")
            if ep.is_default_output:
                flags.append("DEFAULT_OUT")
            if ep.is_loopback:
                flags.append("LOOPBACK")
            flag_str = f" [{' | '.join(flags)}]" if flags else ""
            print(
                f" [{ep.index:02d}] {ep.name:<48} "
                f"In:{ep.max_input_channels} Out:{ep.max_output_channels} "
                f"Rate:{ep.default_sample_rate}Hz{flag_str}"
            )

        print("\n" + "-" * 78)
        print(" TOPOLOGIA AUTO-DETECTADA")
        print("-" * 78)
        print(f" 1. Microfono Fisico (Flujo 1 In)     : {topology.physical_mic.name if topology.physical_mic else 'NO DETECTADO'}")
        print(f" 2. Auriculares Fisicos (SFX + TTS)   : {topology.physical_headphones.name if topology.physical_headphones else 'NO DETECTADO'}")
        print(f" 3. Microfono Virtual Juego (Cable-A) : {topology.virtual_mic_sink_cable_a.name if topology.virtual_mic_sink_cable_a else 'FALTA VB-CABLE A'}")
        print(f" 4. Captura Chat Companeros (Cable-B) : {topology.teammate_voip_source_cable_b.name if topology.teammate_voip_source_cable_b else 'FALTA VB-CABLE B'}")
        print("-" * 78)

        if topology.physical_mic is None:
            print(" [!] No se encontro un microfono fisico activo para la prueba de RingBuffer.")
            return

        print("\n Iniciando prueba de captura WASAPI -> SPSC RingBuffer (16 kHz) durante 5 segundos...")
        print(" Habla por tu microfono para verificar el medidor RMS y el Pre-Roll de 300 ms:\n")

        rb = SPSCRingBuffer(name="ignait_mic_test", capacity_seconds=15.0, sample_rate=16000, create=True)
        pa = pyaudio.PyAudio()
        stream = WASAPIInputStream(pa=pa, endpoint=topology.physical_mic, ring_buffer=rb)

        try:
            stream.start()
            t0 = time.perf_counter()
            frames_32ms_processed = 0

            while (time.perf_counter() - t0) < 5.0:
                while rb.available_to_read >= VAD_FRAME_SAMPLES:
                    dummy = rb.slice_window(
                        rb.read_cursor, rb.read_cursor + VAD_FRAME_SAMPLES, preroll_samples=0
                    )
                    out_chunk = dummy.copy()
                    rb.read_into(out_chunk)
                    frames_32ms_processed += 1

                st = rb.stats()
                bar_len = max(0, min(40, int((stream.last_rms_dbfs + 60.0) * 0.8)))
                vu_bar = "#" * bar_len + "." * (40 - bar_len)
                print(
                    f"\r Nivel: [{vu_bar}] {stream.last_rms_dbfs:6.1f} dBFS | "
                    f"Frames 32ms: {frames_32ms_processed:03d} | Overruns: {st.overruns}",
                    end="",
                    flush=True,
                )
                time.sleep(0.02)

            preroll_test = rb.slice_window(rb.write_cursor, rb.write_cursor, preroll_samples=4800)
            print(
                f"\n\n [OK] Prueba completada: {frames_32ms_processed} tramas de 32 ms procesadas sin bloqueo."
                f"\n [OK] Ventana Pre-Roll extraida: {preroll_test.shape[0]} muestras ({preroll_test.shape[0] / 16.0:.0f} ms a 16 kHz)."
            )
        finally:
            stream.stop()
            pa.terminate()
            rb.close()
    finally:
        manager.close()


if __name__ == "__main__":
    run_diagnostics()
