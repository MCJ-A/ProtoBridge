from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
from numpy.typing import NDArray
import pyaudiowpatch as pyaudio

from protobridge.audio.devices import AudioEndpoint, WASAPIDeviceManager
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.audio.streams import WASAPIInputStream, WASAPIOutputPlayer
from protobridge.config import AppConfig
from protobridge.diarization.speaker_id import OnlineSpeakerTracker
from protobridge.tts.kokoro_engine import KokoroTTSEngine
from protobridge.vad.silero import SileroVADSegmenter


def calibrate_user_voice(duration_seconds: float = 6.0, config_path: Path = Path("config.json")) -> float:
    cfg = AppConfig.load(config_path)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    dev_manager = WASAPIDeviceManager()
    all_endpoints = dev_manager.list_wasapi_endpoints()
    topology = dev_manager.auto_detect_topology()

    def _resolve(configured_name: str, fallback: AudioEndpoint | None, is_input: bool) -> AudioEndpoint | None:
        if configured_name:
            clean = configured_name.strip().lower()
            for ep in all_endpoints:
                if ep.is_loopback:
                    continue
                if is_input and ep.max_input_channels <= 0:
                    continue
                if not is_input and ep.max_output_channels <= 0:
                    continue
                if clean in ep.name.lower() or ep.name.lower() in clean:
                    if dev_manager.is_endpoint_valid(ep):
                        return ep
        return fallback

    mic = _resolve(cfg.mic_device_name, topology.physical_mic, is_input=True)
    headphones = _resolve(cfg.headphones_device_name, topology.physical_headphones, is_input=False)

    if mic is None:
        raise RuntimeError("No se detectó el micrófono físico para la calibración.")
    if headphones is None:
        raise RuntimeError("No se detectaron los auriculares físicos para el feedback acústico.")

    print("=" * 76)
    print(" PROTOBRIDGE - CALIBRACIÓN DE VOZ SIGILOSA (STEALTH VOICE CLONER)")
    print("=" * 76)
    print(f" Micrófono : {mic.name}")
    print(f" Salida    : {headphones.name}")
    print(f" Duración  : {duration_seconds:.1f} segundos de lectura táctica continua.")
    print("\n [INSTRUCCIÓN] Cuando inicie la cuenta atrás, habla con tu tono natural de juego.")
    print(" Frase recomendada: 'Helldiver reportándose. Solicitando extracción y apoyo aéreo.'")
    print("=" * 76)

    for i in range(3, 0, -1):
        print(f" Iniciando en {i}...", end="\r", flush=True)
        time.sleep(1.0)
    print("\n [GRABANDO] ¡Habla ahora!...                                         ")

    rb = SPSCRingBuffer(name="protobridge_calib_rb", capacity_seconds=15.0, sample_rate=16000, create=True)
    pa = pyaudio.PyAudio()
    mic_stream = WASAPIInputStream(pa=pa, endpoint=mic, ring_buffer=rb)
    vad = SileroVADSegmenter(models_dir=models_dir, speech_threshold=0.45, min_silence_ms=200)

    mic_stream.start()
    t_end = time.perf_counter() + duration_seconds
    speech_segments: list[NDArray[np.float32]] = []

    try:
        while time.perf_counter() < t_end:
            segments = vad.process_available(rb)
            speech_segments.extend(segments)
            remaining = max(0.0, t_end - time.perf_counter())
            print(f"\r Tiempo restante: {remaining:4.1f}s | Segmentos vocales: {len(speech_segments)}", end="", flush=True)
            time.sleep(0.02)
    finally:
        mic_stream.stop()
        rb.close()

    if not speech_segments:
        raise RuntimeError("No se detectó voz suficiente durante la calibración. Sube la ganancia del micrófono.")

    combined_speech = np.concatenate(speech_segments, axis=0)
    print("\n\n [PROCESANDO] Estimando tono fundamental (F0) por autocorrelación...")
    f0 = OnlineSpeakerTracker._estimate_pitch_hz(combined_speech)

    clone_path = models_dir / "my_voice_clone.npz"
    np.savez_compressed(
        clone_path,
        f0=np.float32(f0),
        samples_analyzed=np.int64(combined_speech.shape[0]),
        calibrated_at=time.time(),
    )
    print(f" [ÉXITO] Frecuencia fundamental detectada: {f0:.1f} Hz")
    print(f" [GUARDADO] Perfil persistido en: {clone_path.resolve()}")

    # Prueba acústica de validación directa al ROG Theta
    print("\n [AUDICIÓN] Sintetizando muestra de verificación clonada...")
    tts = KokoroTTSEngine(models_dir=models_dir, default_voice="my_cloned")
    test_phrase = "Voice calibration complete. Tactical speech synthesizer is online and calibrated to your frequency."
    pcm, rate = tts.synthesize(test_phrase, voice="my_cloned", lang="en-us", speed=cfg.tts_speed_out)

    player = WASAPIOutputPlayer(pa=pa, endpoint=headphones)
    try:
        player.play_pcm(pcm, source_rate=rate, pan_left=0.5, pan_right=0.5)
    finally:
        player.close()
        pa.terminate()
        dev_manager.close()

    return f0


def main() -> None:
    parser = argparse.ArgumentParser(description="ProtoBridge Voice Calibration Tool")
    parser.add_argument("--duration", type=float, default=6.0, help="Duración de captura en segundos (default: 6.0)")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Ruta al config.json")
    args = parser.parse_args()
    calibrate_user_voice(duration_seconds=args.duration, config_path=args.config)


if __name__ == "__main__":
    main()
