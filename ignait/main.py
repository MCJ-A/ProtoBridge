from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Optional
import pyaudiowpatch as pyaudio

from ignait.audio.devices import AudioEndpoint, WASAPIDeviceManager
from ignait.audio.ring_buffer import SPSCRingBuffer
from ignait.audio.streams import (
    AsyncOutputWorker,
    MediaLookaheadBuffer,
    PlaybackItem,
    TACTICAL_CHIRP_PCM,
    WASAPIInputStream,
    WASAPIOutputPlayer,
)
from ignait.config import AppConfig
from ignait.control.hotkeys import GlobalHotkeyListener
from ignait.diarization.speaker_id import OnlineSpeakerTracker
from ignait.mt.translator import TacticalTranslatorEngine
from ignait.stt.whisper_engine import WhisperTranslatorEngine
from ignait.system.os_priority import set_high_process_priority
from ignait.tts.kokoro_engine import KokoroTTSEngine
from ignait.vad.silero import SileroVADSegmenter


def _resolve_device(
    endpoints: list[AudioEndpoint],
    configured_name: str,
    fallback: Optional[AudioEndpoint],
    is_input: bool,
    is_loopback: bool,
    dev_manager: Optional[WASAPIDeviceManager] = None,
) -> Optional[AudioEndpoint]:
    if configured_name:
        clean = configured_name.strip().lower()
        for ep in endpoints:
            if is_loopback != ep.is_loopback:
                continue
            if is_input and ep.max_input_channels <= 0 and not ep.is_loopback:
                continue
            if not is_input and ep.max_output_channels <= 0:
                continue
            if clean in ep.name.lower() or ep.name.lower() in clean:
                if dev_manager is None or dev_manager.is_endpoint_valid(ep):
                    return ep
    return fallback


def main() -> None:
    # Elevación de prioridad para latencia determinista en DSP
    set_high_process_priority()

    parser = argparse.ArgumentParser(
        description="Ignait v1.0: Real-Time Full-Duplex Tactical Translator"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Ruta al archivo de configuración JSON",
    )
    parser.add_argument(
        "--monitor-own-voice",
        action="store_true",
        help="Sobrescribir y activar monitoreo en auriculares",
    )
    parser.add_argument(
        "--my-voice",
        type=str,
        default="",
        help="Sobrescribir perfil de voz (my_cloned, am_adam, etc.)",
    )
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    if args.monitor_own_voice:
        cfg.monitor_own_voice = True
    if args.my_voice:
        cfg.my_voice_profile = args.my_voice

    models_dir = Path("models")
    dev_manager = WASAPIDeviceManager()
    all_endpoints = dev_manager.list_wasapi_endpoints()
    topology_default = dev_manager.auto_detect_topology()

    mic = _resolve_device(
        all_endpoints,
        cfg.mic_device_name,
        topology_default.physical_mic,
        is_input=True,
        is_loopback=False,
        dev_manager=dev_manager,
    )
    headphones = _resolve_device(
        all_endpoints,
        cfg.headphones_device_name,
        topology_default.physical_headphones,
        is_input=False,
        is_loopback=False,
        dev_manager=dev_manager,
    )
    cable_a = _resolve_device(
        all_endpoints,
        cfg.cable_a_device_name,
        topology_default.virtual_mic_sink_cable_a,
        is_input=False,
        is_loopback=False,
        dev_manager=dev_manager,
    )
    team_bus = _resolve_device(
        all_endpoints,
        cfg.team_loopback_device_name,
        topology_default.teammate_voip_source_cable_b,
        is_input=True,
        is_loopback=True,
        dev_manager=dev_manager,
    )

    if mic is None:
        raise RuntimeError("No se pudo resolver el Micrófono Físico.")
    if headphones is None:
        raise RuntimeError("No se pudieron resolver los Auriculares Físicos.")
    if cable_a is None:
        raise RuntimeError("No se pudo resolver CABLE Input (VB-Cable A).")
    if team_bus is None:
        raise RuntimeError("No se pudo resolver el Bus Loopback de compañeros.")

    print("=" * 88)
    print(" IGNAIT v1.0 - TRADUCTOR TÁCTICO FULL-DUPLEX (CONFIGURACIÓN CARGADA)")
    print("=" * 88)
    print(f" [FLUJO 1 - TX ({cfg.my_source_lang.upper()} -> {cfg.my_target_lang.upper()})]")
    print(f"   Micrófono Físico : {mic.name}")
    print(f"   Salida al Juego  : {cable_a.name} (Perfil: {cfg.my_voice_profile} @ {cfg.tts_speed_out}x)")
    print(f"   Monitoreo Local  : {'ACTIVO' if cfg.monitor_own_voice else 'DESACTIVADO'}")
    print(f"   PTT / Hotkey TX  : {'ACTIVO (' + cfg.hotkey_ptt + ')' if cfg.ptt_enabled else 'VAD Libre'} | Mute TX: {cfg.hotkey_mute_tx}")
    print(f"\n [FLUJO 2 - RX ({cfg.teammate_source_lang.upper()} -> {cfg.teammate_target_lang.upper()} 3D)]")
    print(f"   Captura Aislada  : {team_bus.name}")
    print(f"   Tus Auriculares  : {headphones.name} (Velocidad: {cfg.tts_speed_in}x) | Mute RX: {cfg.hotkey_mute_rx}")
    print(f"   VAD Dinámico     : Umbral {cfg.vad_threshold:.2f} | Silencio {cfg.vad_silence_ms}ms")
    print(f"   Barge-In / Chirp : {'Barge-In ON' if cfg.barge_in_enabled else 'Barge-In OFF'} | {'Chirp ON' if cfg.tactical_chirp_enabled else 'Chirp OFF'}")
    print("=" * 88)

    vad_mic = SileroVADSegmenter(
        models_dir=models_dir,
        speech_threshold=cfg.vad_threshold,
        min_silence_ms=cfg.vad_silence_ms,
    )
    vad_team = SileroVADSegmenter(
        models_dir=models_dir,
        speech_threshold=cfg.vad_threshold,
        min_silence_ms=cfg.vad_silence_ms,
    )
    speaker_tracker = OnlineSpeakerTracker(models_dir=models_dir, max_speakers=4)
    stt = WhisperTranslatorEngine(model_size=cfg.whisper_model_size)
    mt = TacticalTranslatorEngine(models_dir=models_dir, custom_glossary=cfg.custom_glossary)
    tts = KokoroTTSEngine(models_dir=models_dir, default_voice=cfg.my_voice_profile)

    rb_mic = SPSCRingBuffer(
        name="ignait_mic_duplex", capacity_seconds=30.0, sample_rate=16000, create=True
    )
    rb_team = SPSCRingBuffer(
        name="ignait_team_duplex", capacity_seconds=30.0, sample_rate=16000, create=True
    )

    media_buffer = MediaLookaheadBuffer(
        enabled=bool(cfg.media_mode_enabled and not cfg.mute_rx),
        delay_ms=float(cfg.media_delay_ms),
        duck_db=cfg.media_ducking_db,
    )

    pa = pyaudio.PyAudio()
    mic_stream = WASAPIInputStream(pa=pa, endpoint=mic, ring_buffer=rb_mic)
    team_stream = WASAPIInputStream(
        pa=pa, endpoint=team_bus, ring_buffer=rb_team, media_buffer=media_buffer
    )

    cable_out = AsyncOutputWorker(pa=pa, endpoint=cable_a)
    headphones_out = AsyncOutputWorker(
        pa=pa, endpoint=headphones, media_buffer=media_buffer
    )
    executor = ThreadPoolExecutor(max_workers=2)

    def _on_ptt_changed(active: bool) -> None:
        if active and cfg.barge_in_enabled:
            headphones_out.flush()
        print(f"\n [PTT] {'HABLANDO' if active else 'LIBERADO'}")

    def _on_reset_squad() -> None:
        speaker_tracker.reset_squad()
        print("\n [ESCUADRÓN] Centroides de voz reiniciados (Ctrl+Shift+R).")

    def _on_tx_mute_changed(muted: bool) -> None:
        print(f"\n [HOTKEY] Mute TX: {'SILENCIADO' if muted else 'ACTIVO'}")

    def _on_rx_mute_changed(muted: bool) -> None:
        media_buffer.configure(
            enabled=bool(cfg.media_mode_enabled and not muted),
            delay_ms=float(cfg.media_delay_ms),
            duck_db=cfg.media_ducking_db,
        )
        print(f"\n [HOTKEY] Mute RX: {'SILENCIADO' if muted else 'ACTIVO'}")

    # Iniciar escuchador global de hotkeys
    hotkeys = GlobalHotkeyListener(
        hotkey_mute_tx=cfg.hotkey_mute_tx,
        hotkey_mute_rx=cfg.hotkey_mute_rx,
        hotkey_ptt=cfg.hotkey_ptt,
        hotkey_reset_squad=cfg.hotkey_reset_squad,
        ptt_enabled=cfg.ptt_enabled,
        initial_mute_tx=cfg.mute_tx,
        initial_mute_rx=cfg.mute_rx,
        on_tx_mute_changed=_on_tx_mute_changed,
        on_rx_mute_changed=_on_rx_mute_changed,
        on_ptt_changed=_on_ptt_changed,
        on_reset_squad=_on_reset_squad,
    )
    hotkeys.start()

    print("\n [ONLINE] Enlace Full-Duplex establecido. Presiona Ctrl+C para finalizar.\n")
    mic_stream.start()
    team_stream.start()

    prev_tx_allowed = False
    prev_rx_allowed = False

    try:
        while True:
            # FLUJO 1: TX (Usuario -> Juego)
            tx_allowed = not hotkeys.is_tx_muted and (
                not cfg.ptt_enabled or hotkeys.is_ptt_active
            )
            if prev_tx_allowed and not tx_allowed:
                vad_mic.reset_state()
                cable_out.flush()
            prev_tx_allowed = tx_allowed

            if tx_allowed:
                mic_segments = vad_mic.process_available(rb_mic)
                if mic_segments and cfg.barge_in_enabled:
                    # Barge-in: interrupción y vaciado de TTS de compañeros al hablar el usuario
                    headphones_out.flush()

                for seg_pcm in mic_segments:
                    if hotkeys.is_tx_muted or (cfg.ptt_enabled and not hotkeys.is_ptt_active):
                        break
                    t0 = time.perf_counter()
                    en_text = stt.translate_es_to_en(seg_pcm, source_lang=cfg.my_source_lang)
                    t_stt = (time.perf_counter() - t0) * 1000.0
                    if not en_text:
                        continue

                    t_tts0 = time.perf_counter()
                    speed_out = (
                        cfg.tts_speed_out + 0.08
                        if cable_out.pending_items > 0
                        else cfg.tts_speed_out
                    )
                    tts_pcm, tts_rate = tts.synthesize(
                        en_text,
                        voice=cfg.my_voice_profile,
                        lang=cfg.my_target_lang,
                        speed=speed_out,
                        ref_audio_16k=seg_pcm,
                    )
                    t_tts = (time.perf_counter() - t_tts0) * 1000.0

                    if hotkeys.is_tx_muted or (cfg.ptt_enabled and not hotkeys.is_ptt_active):
                        break

                    print(
                        f"\n [TX -> {cfg.my_target_lang.upper()}] \"{en_text}\" ({t_stt + t_tts:.0f}ms: STT {t_stt:.0f}ms + TTS {t_tts:.0f}ms)"
                    )
                    cable_out.enqueue(tts_pcm, sample_rate=tts_rate)

                    # Tono acústico de confirmación al usuario
                    if cfg.tactical_chirp_enabled:
                        headphones_out.play_chirp(volume=0.35)
                        vad_mic.notify_output_audio(150.0)

                    if cfg.monitor_own_voice:
                        headphones_out.enqueue(
                            tts_pcm, sample_rate=tts_rate, pan_left=0.5, pan_right=0.5
                        )
                        vad_mic.notify_output_audio(
                            (len(tts_pcm) / max(1, tts_rate)) * 1000.0 + 150.0
                        )
            else:
                # Drenar ring buffer de micrófono si TX está silenciado
                _ = rb_mic.read_all()

            # FLUJO 2: RX (Compañeros -> Auriculares)
            rx_allowed = not hotkeys.is_rx_muted
            if prev_rx_allowed and not rx_allowed:
                vad_team.reset_state()
                headphones_out.flush()
            prev_rx_allowed = rx_allowed

            if rx_allowed:
                team_segments = vad_team.process_available(rb_team)
                for seg_pcm in team_segments:
                    if hotkeys.is_rx_muted:
                        break
                    t0 = time.perf_counter()
                    fut_spk = executor.submit(speaker_tracker.identify_speaker, seg_pcm)
                    team_text, det_lang, _ = stt.transcribe_any(
                        seg_pcm, source_lang=cfg.teammate_source_lang
                    )
                    spk = fut_spk.result()
                    t_stt = (time.perf_counter() - t0) * 1000.0
                    if not team_text or hotkeys.is_rx_muted:
                        continue

                    t_mt0 = time.perf_counter()
                    target_es = mt.translate(
                        team_text, src_lang=det_lang, tgt_lang=cfg.teammate_target_lang
                    )
                    t_mt = (time.perf_counter() - t_mt0) * 1000.0
                    if not target_es or hotkeys.is_rx_muted:
                        continue

                    t_tts0 = time.perf_counter()
                    speed_in = (
                        cfg.tts_speed_in + 0.10
                        if headphones_out.pending_items >= 1
                        else cfg.tts_speed_in
                    )
                    tts_pcm, tts_rate = tts.synthesize(
                        target_es,
                        voice=spk.voice_name,
                        lang=cfg.teammate_target_lang,
                        speed=speed_in,
                    )
                    t_tts = (time.perf_counter() - t_tts0) * 1000.0

                    if hotkeys.is_rx_muted:
                        break

                    print(
                        f"\n [{spk.label} ({spk.pos_label} | {spk.voice_name})] \"{team_text}\" -> \"{target_es}\" ({t_stt + t_mt + t_tts:.0f}ms)"
                    )
                    headphones_out.enqueue(
                        tts_pcm,
                        sample_rate=tts_rate,
                        pan_left=spk.pan_left,
                        pan_right=spk.pan_right,
                    )
                    vad_mic.notify_output_audio(
                        (len(tts_pcm) / max(1, tts_rate)) * 1000.0 + 150.0
                    )
            else:
                # Drenar ring buffer de equipo si RX está silenciado
                _ = rb_team.read_all()

            mic_flag = "TX:VOZ" if vad_mic.is_speaking else "TX:Idle"
            rx_flag = "RX:VOZ" if vad_team.is_speaking else "RX:Idle"
            print(
                f"\r [{mic_flag} {mic_stream.last_rms_dbfs:5.1f}dB] | [{rx_flag} {team_stream.last_rms_dbfs:5.1f}dB] ",
                end="",
                flush=True,
            )
            time.sleep(0.012)

    except KeyboardInterrupt:
        print("\n\n [Cerrando] Apagando enlace Ignait de forma segura...")
    finally:
        hotkeys.stop()
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
