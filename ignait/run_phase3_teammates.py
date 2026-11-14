from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import pyaudiowpatch as pyaudio

from ignait.audio.devices import AudioEndpoint, WASAPIDeviceManager
from ignait.audio.ring_buffer import SPSCRingBuffer
from ignait.audio.streams import WASAPIInputStream, WASAPIOutputPlayer
from ignait.diarization.speaker_id import OnlineSpeakerTracker
from ignait.mt.translator import TacticalTranslatorEngine
from ignait.stt.whisper_engine import WhisperTranslatorEngine
from ignait.tts.kokoro_engine import KokoroTTSEngine
from ignait.vad.silero import SileroVADSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ignait Fase 3: Flujo 2 (Compañeros en Inglés -> Español Multi-Voz 3D)"
    )
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Usa tu micrófono ROG CARNYX para probar hablando en inglés (o acercando el móvil con un vídeo en inglés).",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    dev_manager = WASAPIDeviceManager()
    topology = dev_manager.auto_detect_topology()

    if topology.physical_headphones is None:
        raise RuntimeError("No se encontraron tus auriculares ROG Theta Ultimate 7.1.")

    source_endpoint: AudioEndpoint | None = (
        topology.physical_mic if args.test_mic else topology.teammate_voip_source_cable_b
    )
    if source_endpoint is None:
        raise RuntimeError("No se encontró el dispositivo de entrada para el Flujo 2.")

    print("=" * 84)
    print(" IGNAIT - FASE 3: TRADUCTOR DE COMPAÑEROS MULTI-VOZ 3D (INGLES -> ESPAÑOL)")
    print("=" * 84)
    print(f" [Fuente Captura]     : {source_endpoint.name}")
    print(f" [Salida Auriculares] : {topology.physical_headphones.name}")
    print(" [Voces Espaciales]   : #1 em_alex (IZQ) | #2 em_santa (CENTRO) | #3 ef_dora (DER)")
    print("-" * 84)

    vad = SileroVADSegmenter(models_dir=models_dir, min_silence_ms=240)
    speaker_tracker = OnlineSpeakerTracker(models_dir=models_dir, max_speakers=4)
    stt = WhisperTranslatorEngine(model_size="small")
    mt = TacticalTranslatorEngine(models_dir=models_dir)
    tts = KokoroTTSEngine(models_dir=models_dir)

    rb_in = SPSCRingBuffer(
        name="ignait_team_phase3", capacity_seconds=30.0, sample_rate=16000, create=True
    )
    pa = pyaudio.PyAudio()

    in_stream = WASAPIInputStream(pa=pa, endpoint=source_endpoint, ring_buffer=rb_in)
    hp_player = WASAPIOutputPlayer(pa=pa, endpoint=topology.physical_headphones)
    executor = ThreadPoolExecutor(max_workers=2)

    mode_msg = (
        "Habla en INGLÉS por tu ROG CARNYX (cambia el tono de voz para simular distintos compañeros)"
        if args.test_mic
        else f"Escuchando el chat de compañeros en {source_endpoint.name}"
    )
    print(f"\n [LISTO] {mode_msg} (Ctrl+C para salir):\n")
    in_stream.start()

    try:
        while True:
            segments = vad.process_available(rb_in)
            for seg_pcm in segments:
                dur_s = seg_pcm.shape[0] / 16000.0
                t0 = time.perf_counter()

                # 1. Lanzar identificación de hablante en CPU EN PARALELO con Whisper en GPU
                future_spk = executor.submit(speaker_tracker.identify_speaker, seg_pcm)
                english_text = stt.transcribe_en(seg_pcm)
                spk_profile = future_spk.result()
                t_stt_spk = (time.perf_counter() - t0) * 1000.0

                if not english_text:
                    continue

                # 2. Traducción Táctica Inglés -> Español (NLLB-200 INT8 + Glosario Gaming)
                t_mt0 = time.perf_counter()
                spanish_text = mt.translate_en_to_es(english_text)
                t_mt = (time.perf_counter() - t_mt0) * 1000.0

                if not spanish_text:
                    continue

                # 3. Síntesis TTS con la identidad vocal asignada al compañero
                t_tts0 = time.perf_counter()
                tts_pcm, tts_rate = tts.synthesize(
                    spanish_text,
                    voice=spk_profile.voice_name,
                    lang="es",
                    speed=1.14,
                )
                t_tts = (time.perf_counter() - t_tts0) * 1000.0
                t_total = t_stt_spk + t_mt + t_tts

                print(
                    f"\n >>> [{spk_profile.label} | Voz: {spk_profile.voice_name} | "
                    f"Audio: {spk_profile.pos_label} | Pitch: {spk_profile.pitch_hz:.0f}Hz]"
                    f"\n     [EN] \"{english_text}\""
                    f"\n     [ES] \"{spanish_text}\""
                    f"\n     [Latencia] Whisper+SpeakerID: {t_stt_spk:.0f}ms | NLLB-MT: {t_mt:.0f}ms | "
                    f"Kokoro-ES: {t_tts:.0f}ms | TOTAL: {t_total:.0f}ms"
                )

                # 4. Reproducir con posicionamiento estéreo espacial (Izquierda / Centro / Derecha)
                hp_player.play_pcm(
                    tts_pcm,
                    source_rate=tts_rate,
                    pan_left=spk_profile.pan_left,
                    pan_right=spk_profile.pan_right,
                )

            state_tag = "VOZ DETECTADA" if vad.is_speaking else "Esperando... "
            print(
                f"\r [{state_tag}] Nivel Bus: {in_stream.last_rms_dbfs:5.1f} dBFS | Prob VAD: {vad.last_prob:.2f} ",
                end="",
                flush=True,
            )
            time.sleep(0.015)

    except KeyboardInterrupt:
        print("\n\n [Cerrando] Deteniendo Fase 3...")
    finally:
        executor.shutdown(wait=False)
        in_stream.stop()
        hp_player.close()
        pa.terminate()
        rb_in.close()
        dev_manager.close()


if __name__ == "__main__":
    main()
