from __future__ import annotations

import argparse
from pathlib import Path
import time
import pyaudiowpatch as pyaudio

from protobridge.audio.devices import WASAPIDeviceManager
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.audio.streams import WASAPIInputStream, WASAPIOutputPlayer
from protobridge.stt.whisper_engine import WhisperTranslatorEngine
from protobridge.tts.kokoro_engine import KokoroTTSEngine
from protobridge.vad.silero import SileroVADSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProtoBridge Fase 2: Flujo 1 (Tu Micrófono Español -> Inglés hacia el Juego)"
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Desactiva escuchar tu propia traducción en tus auriculares ROG Theta (modo partida real).",
    )
    parser.add_argument(
        "--voice",
        default="am_adam",
        help="Perfil de voz en inglés (ej. am_adam, am_michael, bf_emma, bm_george).",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    dev_manager = WASAPIDeviceManager()
    topology = dev_manager.auto_detect_topology()

    if topology.physical_mic is None:
        raise RuntimeError("No se encontró tu micrófono físico (ROG CARNYX).")
    if topology.virtual_mic_sink_cable_a is None:
        raise RuntimeError("No se encontró CABLE Input (VB-Audio Virtual Cable).")

    print("=" * 80)
    print(" PROTOBRIDGE - FASE 2: TRADUCTOR DE MICROFONO EN VIVO (ESPAÑOL -> INGLES)")
    print("=" * 80)
    print(f" [Entrada Micrófono] : {topology.physical_mic.name}")
    print(f" [Salida al Juego]   : {topology.virtual_mic_sink_cable_a.name}")
    if not args.no_monitor and topology.physical_headphones is not None:
        print(f" [Monitoreo Local]   : {topology.physical_headphones.name} (Activado para prueba)")
    else:
        print(" [Monitoreo Local]   : Desactivado (Solo escuchan tus compañeros en el juego)")
    print("-" * 80)

    vad = SileroVADSegmenter(models_dir=models_dir)
    stt = WhisperTranslatorEngine(model_size="small")
    tts = KokoroTTSEngine(models_dir=models_dir, default_voice=args.voice)

    rb_mic = SPSCRingBuffer(
        name="protobridge_mic_phase2", capacity_seconds=30.0, sample_rate=16000, create=True
    )
    pa = pyaudio.PyAudio()

    mic_stream = WASAPIInputStream(pa=pa, endpoint=topology.physical_mic, ring_buffer=rb_mic)
    cable_player = WASAPIOutputPlayer(pa=pa, endpoint=topology.virtual_mic_sink_cable_a)
    monitor_player: WASAPIOutputPlayer | None = None
    if not args.no_monitor and topology.physical_headphones is not None:
        monitor_player = WASAPIOutputPlayer(pa=pa, endpoint=topology.physical_headphones)

    print("\n [LISTO] Habla en ESPAÑOL por tu micrófono ROG CARNYX (Presiona Ctrl+C para salir):\n")
    mic_stream.start()

    try:
        while True:
            segments = vad.process_available(rb_mic)
            for seg_pcm in segments:
                dur_s = seg_pcm.shape[0] / 16000.0
                t_start = time.perf_counter()

                english_text = stt.translate_es_to_en(seg_pcm)
                t_stt = (time.perf_counter() - t_start) * 1000.0

                if not english_text:
                    continue

                t_tts_start = time.perf_counter()
                tts_pcm, tts_rate = tts.synthesize(english_text, lang="en-us", speed=1.10)
                t_tts = (time.perf_counter() - t_tts_start) * 1000.0
                t_total = t_stt + t_tts

                print(
                    f"\n >>> [Voz: {dur_s:.1f}s] Traducción EN: \"{english_text}\""
                    f"\n     [Latencia Cómputo] Whisper GPU: {t_stt:.0f} ms | Kokoro TTS: {t_tts:.0f} ms | TOTAL: {t_total:.0f} ms"
                )

                # 1. Inyectar al micrófono virtual del juego (VB-Cable Input)
                cable_player.play_pcm(tts_pcm, source_rate=tts_rate)

                # 2. Reproducir en tus auriculares ROG Theta si el modo prueba está activo
                if monitor_player is not None:
                    monitor_player.play_pcm(tts_pcm, source_rate=tts_rate)

            state_tag = "HABLANDO..." if vad.is_speaking else "Escuchando "
            print(
                f"\r [{state_tag}] Nivel Mic: {mic_stream.last_rms_dbfs:5.1f} dBFS | Prob Voz: {vad.last_prob:.2f} ",
                end="",
                flush=True,
            )
            time.sleep(0.015)

    except KeyboardInterrupt:
        print("\n\n [Cerrando] Deteniendo streams de audio...")
    finally:
        mic_stream.stop()
        cable_player.close()
        if monitor_player is not None:
            monitor_player.close()
        pa.terminate()
        rb_mic.close()
        dev_manager.close()


if __name__ == "__main__":
    main()
