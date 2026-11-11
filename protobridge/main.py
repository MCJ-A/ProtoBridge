from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
import numpy as np
from numpy.typing import NDArray
import pyaudiowpatch as pyaudio

from protobridge.audio.devices import AudioEndpoint, WASAPIDeviceManager
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.audio.streams import WASAPIInputStream, WASAPIOutputPlayer
from protobridge.diarization.speaker_id import OnlineSpeakerTracker
from protobridge.mt.translator import TacticalTranslatorEngine
from protobridge.stt.whisper_engine import WhisperTranslatorEngine
from protobridge.tts.kokoro_engine import KokoroTTSEngine
from protobridge.vad.silero import SileroVADSegmenter


@dataclass(frozen=True)
class PlaybackItem:
    pcm: NDArray[np.float32]
    sample_rate: int
    pan_left: float = 1.0
    pan_right: float = 1.0


class AsyncOutputWorker:
    """
    Hilo dedicado de reproducción WASAPI no bloqueante.
    Permite que la captura VAD y la inferencia GPU/CPU sigan procesando mientras suena el audio.
    """

    def __init__(self, pa: pyaudio.PyAudio, endpoint: AudioEndpoint) -> None:
        self._player = WASAPIOutputPlayer(pa=pa, endpoint=endpoint)
        self._queue: queue.Queue[PlaybackItem | None] = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def pending_items(self) -> int:
        return self._queue.qsize()

    def enqueue(
        self,
        pcm: NDArray[np.float32],
        sample_rate: int,
        pan_left: float = 1.0,
        pan_right: float = 1.0,
    ) -> None:
        try:
            self._queue.put_nowait(PlaybackItem(pcm, sample_rate, pan_left, pan_right))
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._player.play_pcm(
                    item.pcm,
                    source_rate=item.sample_rate,
                    pan_left=item.pan_left,
                    pan_right=item.pan_right,
                )
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        self._player.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProtoBridge Fase 4: Traductor Bidireccional Full-Duplex para Gaming"
    )
    parser.add_argument(
        "--monitor-own-voice",
        action="store_true",
        help="Escuchar también tu propia voz traducida al inglés en tus auriculares ROG Theta.",
    )
    parser.add_argument(
        "--my-voice",
        default="am_adam",
        help="Tu perfil de voz en inglés hacia el juego (am_adam, am_michael, bm_george, bf_emma).",
    )
    args = parser.parse_args()

    models_dir = Path("models")
    dev_manager = WASAPIDeviceManager()
    topology = dev_manager.auto_detect_topology()

    if topology.physical_mic is None:
        raise RuntimeError("No se encontró tu micrófono físico (ROG CARNYX).")
    if topology.physical_headphones is None:
        raise RuntimeError("No se encontraron tus auriculares físicos (ROG Theta).")
    if topology.virtual_mic_sink_cable_a is None:
        raise RuntimeError("No se encontró CABLE Input (VB-Audio Virtual Cable).")
    if topology.teammate_voip_source_cable_b is None:
        raise RuntimeError("No se encontró el bus Loopback para capturar el chat de compañeros.")

    team_bus: AudioEndpoint = topology.teammate_voip_source_cable_b

    print("=" * 88)
    print(" PROTOBRIDGE v1.0 - TRADUCTOR BIDIRECCIONAL EN TIEMPO REAL (FULL-DUPLEX)")
    print("=" * 88)
    print(f" [FLUJO 1 - TU VOZ (ES -> EN)]")
    print(f"   Micrófono Físico : {topology.physical_mic.name}")
    print(f"   Salida al Juego  : {topology.virtual_mic_sink_cable_a.name} (Voz: {args.my_voice})")
    print(f"\n [FLUJO 2 - COMPAÑEROS (EN -> ES MULTI-VOZ 3D)]")
    print(f"   Captura Aislada  : {team_bus.name}")
    print(f"   Tus Auriculares  : {topology.physical_headphones.name}")
    print(f"   Perfiles 3D      : #1 em_alex (IZQ) | #2 em_santa (CENTRO) | #3 ef_dora (DER)")
    print("=" * 88)

    # Inicializar motores de IA (1 sola instancia de Whisper en RTX 5060 para ambos flujos)
    vad_mic = SileroVADSegmenter(models_dir=models_dir, min_silence_ms=230)
    vad_team = SileroVADSegmenter(models_dir=models_dir, min_silence_ms=240)
    speaker_tracker = OnlineSpeakerTracker(models_dir=models_dir, max_speakers=4)
    stt = WhisperTranslatorEngine(model_size="small")
    mt = TacticalTranslatorEngine(models_dir=models_dir)
    tts = KokoroTTSEngine(models_dir=models_dir, default_voice=args.my_voice)

    rb_mic = SPSCRingBuffer(
        name="protobridge_mic_duplex", capacity_seconds=30.0, sample_rate=16000, create=True
    )
    rb_team = SPSCRingBuffer(
        name="protobridge_team_duplex", capacity_seconds=30.0, sample_rate=16000, create=True
    )

    pa = pyaudio.PyAudio()
    mic_stream = WASAPIInputStream(pa=pa, endpoint=topology.physical_mic, ring_buffer=rb_mic)
    team_stream = WASAPIInputStream(pa=pa, endpoint=team_bus, ring_buffer=rb_team)

    cable_out = AsyncOutputWorker(pa=pa, endpoint=topology.virtual_mic_sink_cable_a)
    headphones_out = AsyncOutputWorker(pa=pa, endpoint=topology.physical_headphones)
    executor = ThreadPoolExecutor(max_workers=2)

    print("\n [ONLINE] Sistema Bidireccional Activo. ¡Buena caza, Helldiver! (Ctrl+C para detener)\n")
    mic_stream.start()
    team_stream.start()

    try:
        while True:
            # ------------------------------------------------------------------
            # FLUJO 1: Tu Micrófono (Español -> Inglés directo hacia el juego)
            # ------------------------------------------------------------------
            mic_segments = vad_mic.process_available(rb_mic)
            for seg_pcm in mic_segments:
                t0 = time.perf_counter()
                en_text = stt.translate_es_to_en(seg_pcm)
                t_stt = (time.perf_counter() - t0) * 1000.0
                if not en_text:
                    continue

                t_tts0 = time.perf_counter()
                speed_out = 1.18 if cable_out.pending_items > 0 else 1.10
                tts_pcm, tts_rate = tts.synthesize(
                    en_text, voice=args.my_voice, lang="en-us", speed=speed_out
                )
                t_tts = (time.perf_counter() - t_tts0) * 1000.0

                print(
                    f"\n [TU -> JUEGO (EN)] \"{en_text}\" "
                    f"({t_stt + t_tts:.0f} ms: GPU {t_stt:.0f}ms + TTS {t_tts:.0f}ms)"
                )
                cable_out.enqueue(tts_pcm, sample_rate=tts_rate)
                if args.monitor_own_voice:
                    headphones_out.enqueue(tts_pcm, sample_rate=tts_rate, pan_left=0.5, pan_right=0.5)

            # ------------------------------------------------------------------
            # FLUJO 2: Compañeros (Inglés -> Español Multi-Voz 3D a tus ROG Theta)
            # ------------------------------------------------------------------
            team_segments = vad_team.process_available(rb_team)
            for seg_pcm in team_segments:
                t0 = time.perf_counter()
                fut_spk = executor.submit(speaker_tracker.identify_speaker, seg_pcm)
                en_team_text = stt.transcribe_en(seg_pcm)
                spk = fut_spk.result()
                t_stt = (time.perf_counter() - t0) * 1000.0
                if not en_team_text:
                    continue

                t_mt0 = time.perf_counter()
                es_text = mt.translate_en_to_es(en_team_text)
                t_mt = (time.perf_counter() - t_mt0) * 1000.0
                if not es_text:
                    continue

                t_tts0 = time.perf_counter()
                # Aceleración adaptativa si hay cola acumulada de varios compañeros hablando
                speed_es = 1.24 if headphones_out.pending_items >= 1 else 1.14
                tts_pcm, tts_rate = tts.synthesize(
                    es_text, voice=spk.voice_name, lang="es", speed=speed_es
                )
                t_tts = (time.perf_counter() - t_tts0) * 1000.0

                print(
                    f"\n [{spk.label} ({spk.pos_label} | {spk.voice_name})] "
                    f"\"{en_team_text}\" -> \"{es_text}\" "
                    f"({t_stt + t_mt + t_tts:.0f} ms)"
                )
                headphones_out.enqueue(
                    tts_pcm,
                    sample_rate=tts_rate,
                    pan_left=spk.pan_left,
                    pan_right=spk.pan_right,
                )

            mic_flag = "TX:HABLANDO" if vad_mic.is_speaking else "TX:Idle    "
            rx_flag = "RX:RECIBIENDO" if vad_team.is_speaking else "RX:Idle      "
            print(
                f"\r [{mic_flag} {mic_stream.last_rms_dbfs:5.1f}dB] | "
                f"[{rx_flag} {team_stream.last_rms_dbfs:5.1f}dB] ",
                end="",
                flush=True,
            )
            time.sleep(0.012)

    except KeyboardInterrupt:
        print("\n\n [Cerrando] Apagando ProtoBridge de forma segura...")
    finally:
        executor.shutdown(wait=False)
        mic_stream.stop()
        team_stream.stop()
        cable_out.close()
        headphones_out.close()
        pa.terminate()
        rb_mic.close()
        rb_team.close()
        dev_manager.close()


if __name__ == "__main__":
    main()
