from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
if sys.stdout is None: sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None: sys.stderr = open(os.devnull, 'w', encoding='utf-8')
import threading
import time
from urllib.parse import parse_qs, urlparse
import urllib.request
import zipfile
import pyaudiowpatch as pyaudio

from protobridge.audio.devices import AudioEndpoint, WASAPIDeviceManager
from protobridge.audio.ring_buffer import SPSCRingBuffer
from protobridge.audio.streams import WASAPIInputStream
from protobridge.config import AppConfig
from protobridge.diarization.speaker_id import OnlineSpeakerTracker, WESPEAKER_ONNX_URL
from protobridge.main import AsyncOutputWorker
from protobridge.mt.translator import NLLB_CT2_REPO, NLLB_TOKENIZER_REPO, TacticalTranslatorEngine
from protobridge.stt.whisper_engine import WhisperTranslatorEngine
from protobridge.tts.kokoro_engine import KokoroTTSEngine, ensure_kokoro_assets
from protobridge.vad.silero import SileroVADSegmenter, ensure_silero_model

from huggingface_hub import hf_hub_download, snapshot_download


def _can_open_input(pa: pyaudio.PyAudio, ep: AudioEndpoint) -> bool:
    """Verifica si un endpoint de entrada o loopback WASAPI responde sin lanzar [Errno -9999]."""
    ch = max(1, ep.max_input_channels if ep.max_input_channels > 0 else ep.max_output_channels)
    try:
        st = pa.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=ep.default_sample_rate,
            input=True,
            input_device_index=ep.index,
            frames_per_buffer=512,
            start=False,
        )
        st.close()
        return True
    except Exception:
        return False


class StitchBackendController:
    def __init__(self) -> None:
        self.models_dir = Path("models")
        self.config = AppConfig.load()
        self.spatial_3d: bool = True

        self.running: bool = False
        self.loading: bool = False
        self.mic_db: float = -100.0
        self.chat_db: float = -100.0
        self.last_latency_ms: int = 0
        self.radar: dict[str, str] = {
            "left": "Inactivo",
            "center": "Inactivo",
            "right": "Inactivo",
        }

        self._log_lock = threading.Lock()
        self._logs: list[dict[str, object]] = []
        self._log_counter: int = 0

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._endpoints: list[AudioEndpoint] = []
        self.devices_payload: dict[str, list[str]] = {"inputs": [], "outputs": [], "loopbacks": []}
        self._scan_verified_devices()

    def add_log(self, msg: str, tag: str = "sys") -> None:
        with self._log_lock:
            self._log_counter += 1
            self._logs.append({"id": self._log_counter, "msg": msg, "tag": tag})
            if len(self._logs) > 250:
                self._logs = self._logs[-250:]

    def get_logs_since(self, since_id: int) -> list[dict[str, object]]:
        with self._log_lock:
            return [item for item in self._logs if int(item["id"]) > since_id]

    def _scan_verified_devices(self) -> None:
        mgr = WASAPIDeviceManager()
        pa = pyaudio.PyAudio()
        try:
            all_eps = mgr.list_wasapi_endpoints()
            topo = mgr.auto_detect_topology()
            self._endpoints = all_eps

            inputs = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.max_input_channels > 0 and not ep.is_loopback and _can_open_input(pa, ep)
            ]
            outputs = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.max_output_channels > 0 and not ep.is_loopback
            ]
            loopbacks = [
                f"[{ep.index}] {ep.name}"
                for ep in all_eps
                if ep.is_loopback and _can_open_input(pa, ep)
            ]

            self.devices_payload = {
                "inputs": inputs,
                "outputs": outputs,
                "loopbacks": loopbacks + inputs,
            }

            if not self.config.mic_device_name and topo.physical_mic:
                self.config.mic_device_name = f"[{topo.physical_mic.index}] {topo.physical_mic.name}"
            if not self.config.headphones_device_name and topo.physical_headphones:
                self.config.headphones_device_name = (
                    f"[{topo.physical_headphones.index}] {topo.physical_headphones.name}"
                )
            if not self.config.cable_a_device_name and topo.virtual_mic_sink_cable_a:
                self.config.cable_a_device_name = (
                    f"[{topo.virtual_mic_sink_cable_a.index}] {topo.virtual_mic_sink_cable_a.name}"
                )

            # Elegir un Loopback verificado que no falle con -9999 (ej. Steam Streaming Speakers o ROG CARNYX)
            valid_lb_Set = set(loopbacks)
            if self.config.team_loopback_device_name not in valid_lb_Set:
                preferred_lb = next(
                    (
                        lb
                        for lb in loopbacks
                        if "steam streaming speakers" in lb.lower()
                        or "carnyx" in lb.lower()
                        or "cable" in lb.lower()
                    ),
                    loopbacks[0] if loopbacks else (inputs[0] if inputs else ""),
                )
                self.config.team_loopback_device_name = preferred_lb

            self.config.save()
        finally:
            pa.terminate()
            mgr.close()

    def _resolve_endpoint(self, label: str) -> AudioEndpoint:
        if label.startswith("["):
            idx = int(label.split("]")[0][1:])
            for ep in self._endpoints:
                if ep.index == idx:
                    return ep
        for ep in self._endpoints:
            if ep.name.lower() in label.lower():
                return ep
        raise RuntimeError(f"No se encontró el dispositivo: {label}")

    def get_models_status(self) -> list[dict[str, object]]:
        has_vbcable = any("cable" in o.lower() for o in self.devices_payload["outputs"])
        has_silero = (self.models_dir / "silero_vad.onnx").exists()
        has_wespeaker = (self.models_dir / "wespeaker_campp.onnx").exists()
        has_kokoro = (self.models_dir / "kokoro-v1.0.int8.onnx").exists() and (
            self.models_dir / "voices-v1.0.bin"
        ).exists()
        has_nllb = (self.models_dir / "nllb_200_ct2_int8" / "model.bin").exists()

        return [
            {
                "title": f"faster-whisper-{self.config.whisper_model_size}-int8_float16 (STT GPU)",
                "subtitle": "GPU NVIDIA RTX 5060 • 99 Idiomas • Detección Automática",
                "ready": True,
            },
            {
                "title": "nllb-200-distilled-600M-int8 (Traducción Táctica)",
                "subtitle": "CTranslate2 CPU • Traducción Directa entre 200 Idiomas",
                "ready": has_nllb,
            },
            {
                "title": "kokoro-82m-v1.0-int8 (Síntesis Multi-Voz 3D)",
                "subtitle": "ONNX Runtime • Voces EN / ES / FR / PT / IT + Voice Blending",
                "ready": has_kokoro,
            },
            {
                "title": "wespeaker-voxceleb-cam++ (Identificación de Hablantes)",
                "subtitle": "Diarización en Paralelo (0 ms latencia añadida) + Detector F0",
                "ready": has_wespeaker,
            },
            {
                "title": "silero-vad-v5 + VB-Audio Virtual Cable",
                "subtitle": "Detección de Voz 32ms + Enrutamiento Kernel WASAPI",
                "ready": has_silero and has_vbcable,
            },
        ]

    def toggle_engine(self) -> None:
        if not self.running and not self.loading:
            self._stop_event.clear()
            self.loading = True
            self._worker_thread = threading.Thread(target=self._engine_loop, daemon=True)
            self._worker_thread.start()
        else:
            self._stop_event.set()

    def _engine_loop(self) -> None:
        if sys.platform == "win32":
            ctypes.windll.ole32.CoInitialize(None)

        pa: pyaudio.PyAudio | None = None
        mic_stream: WASAPIInputStream | None = None
        team_stream: WASAPIInputStream | None = None
        cable_out: AsyncOutputWorker | None = None
        hp_out: AsyncOutputWorker | None = None
        rb_mic: SPSCRingBuffer | None = None
        rb_team: SPSCRingBuffer | None = None
        executor = ThreadPoolExecutor(max_workers=2)

        try:
            mic_ep = self._resolve_endpoint(self.config.mic_device_name)
            hp_ep = self._resolve_endpoint(self.config.headphones_device_name)
            cable_ep = self._resolve_endpoint(self.config.cable_a_device_name)
            team_ep = self._resolve_endpoint(self.config.team_loopback_device_name)

            self.add_log(
                f"[INICIANDO] Cargando Faster-Whisper ({self.config.whisper_model_size}) en RTX 5060 + NLLB-200 + Kokoro...",
                "sys",
            )

            vad_mic = SileroVADSegmenter(
                models_dir=self.models_dir, min_silence_ms=self.config.vad_silence_ms
            )
            vad_team = SileroVADSegmenter(
                models_dir=self.models_dir, min_silence_ms=self.config.vad_silence_ms
            )
            speaker_tracker = OnlineSpeakerTracker(models_dir=self.models_dir, max_speakers=4)
            stt = WhisperTranslatorEngine(model_size=self.config.whisper_model_size)
            mt = TacticalTranslatorEngine(models_dir=self.models_dir)
            tts = KokoroTTSEngine(
                models_dir=self.models_dir, default_voice=self.config.my_voice_profile
            )

            rb_mic = SPSCRingBuffer(
                name="protobridge_stitch_mic", capacity_seconds=30.0, sample_rate=16000, create=True
            )
            rb_team = SPSCRingBuffer(
                name="protobridge_stitch_team", capacity_seconds=30.0, sample_rate=16000, create=True
            )

            pa = pyaudio.PyAudio()
            mic_stream = WASAPIInputStream(pa=pa, endpoint=mic_ep, ring_buffer=rb_mic)
            mic_stream.start()

            try:
                team_stream = WASAPIInputStream(pa=pa, endpoint=team_ep, ring_buffer=rb_team)
                team_stream.start()
            except Exception as team_err:
                self.add_log(
                    f"[AVISO] El bus '{team_ep.name}' está inactivo ({team_err}). Conmutando a bus Loopback de respaldo...",
                    "sys",
                )
                team_stream = None
                for candidate_str in self.devices_payload["loopbacks"]:
                    cand_ep = self._resolve_endpoint(candidate_str)
                    if cand_ep.index == team_ep.index:
                        continue
                    try:
                        team_stream = WASAPIInputStream(pa=pa, endpoint=cand_ep, ring_buffer=rb_team)
                        team_stream.start()
                        self.add_log(
                            f"[OK] Bus de compañeros conectado en: {cand_ep.name}", "rx"
                        )
                        break
                    except Exception:
                        team_stream = None

            cable_out = AsyncOutputWorker(pa=pa, endpoint=cable_ep)
            hp_out = AsyncOutputWorker(pa=pa, endpoint=hp_ep)

            self.loading = False
            self.running = True
            self.radar = {
                "left": "Escuchando [-45°]",
                "center": "Listo [0°]",
                "right": "Escuchando [+45°]",
            }
            self.add_log(
                f"[ONLINE] Motor WASAPI Full-Duplex activo. Mic: {mic_ep.name} -> {cable_ep.name}",
                "rx",
            )

            while not self._stop_event.is_set():
                self.mic_db = mic_stream.last_rms_dbfs if mic_stream else -100.0
                self.chat_db = team_stream.last_rms_dbfs if team_stream else -100.0

                # FLUJO 1: TU MICRÓFONO -> JUEGO
                mic_segs = vad_mic.process_available(rb_mic)
                if not self.config.mute_tx:
                    for seg in mic_segs:
                        t0 = time.perf_counter()
                        my_src = self.config.my_source_lang
                        my_tgt = self.config.my_target_lang

                        if my_tgt == "en":
                            out_text = stt.translate_es_to_en(seg, source_lang=my_src)
                            det_lang = my_src
                        else:
                            raw_txt, det_lang, _ = stt.transcribe_any(seg, source_lang=my_src)
                            out_text = (
                                mt.translate(raw_txt, src_lang=det_lang, tgt_lang=my_tgt)
                                if raw_txt
                                else ""
                            )

                        if not out_text:
                            continue

                        tts_pcm, tts_rate = tts.synthesize(
                            out_text,
                            voice=self.config.my_voice_profile,
                            lang=my_tgt,
                            speed=self.config.tts_speed_in,
                        )
                        dt_ms = int((time.perf_counter() - t0) * 1000.0)
                        self.last_latency_ms = dt_ms

                        self.add_log(
                            f"[TX TÚ ({det_lang.upper()} -> {my_tgt.upper()}) | {dt_ms}ms]: \"{out_text}\"",
                            "tx",
                        )
                        cable_out.enqueue(tts_pcm, sample_rate=tts_rate)
                        if self.config.monitor_own_voice:
                            hp_out.enqueue(tts_pcm, sample_rate=tts_rate, pan_left=0.5, pan_right=0.5)

                # FLUJO 2: COMPAÑEROS -> TUS AURICULARES (3D + AUTO-IDIOMA)
                if team_stream is not None:
                    team_segs = vad_team.process_available(rb_team)
                    if not self.config.mute_rx:
                        for seg in team_segs:
                            t0 = time.perf_counter()
                            tm_src = self.config.teammate_source_lang
                            tm_tgt = self.config.teammate_target_lang

                            fut_spk = executor.submit(speaker_tracker.identify_speaker, seg)
                            src_text, det_lang, _ = stt.transcribe_any(seg, source_lang=tm_src)
                            spk = fut_spk.result()

                            if not src_text:
                                continue

                            translated_text = mt.translate(
                                src_text, src_lang=det_lang, tgt_lang=tm_tgt
                            )
                            if not translated_text:
                                continue

                            spd = self.config.tts_speed_in
                            if hp_out.pending_items >= 1:
                                spd = min(1.30, spd + 0.10)

                            tts_pcm, tts_rate = tts.synthesize(
                                translated_text,
                                voice=spk.voice_name,
                                lang=tm_tgt,
                                speed=spd,
                            )
                            dt_ms = int((time.perf_counter() - t0) * 1000.0)
                            self.last_latency_ms = dt_ms

                            tag_info = f"{spk.voice_name} ({det_lang.upper()}->{tm_tgt.upper()})"
                            if spk.pos_label == "IZQUIERDA":
                                self.radar["left"] = tag_info
                            elif spk.pos_label == "CENTRO":
                                self.radar["center"] = tag_info
                            else:
                                self.radar["right"] = tag_info

                            self.add_log(
                                f"[RX {spk.label} ({spk.pos_label}) | {det_lang.upper()}->{tm_tgt.upper()} | {dt_ms}ms]: \"{src_text}\" => \"{translated_text}\"",
                                "rx",
                            )
                            pan_l = spk.pan_left if self.spatial_3d else 0.7
                            pan_r = spk.pan_right if self.spatial_3d else 0.7
                            hp_out.enqueue(
                                tts_pcm, sample_rate=tts_rate, pan_left=pan_l, pan_right=pan_r
                            )

                time.sleep(0.015)

        except Exception as exc:
            self.add_log(f"[ERROR MOTOR]: {exc}", "err")
        finally:
            executor.shutdown(wait=False)
            if mic_stream is not None:
                mic_stream.stop()
            if team_stream is not None:
                team_stream.stop()
            if cable_out is not None:
                cable_out.close()
            if hp_out is not None:
                hp_out.close()
            if pa is not None:
                pa.terminate()
            if rb_mic is not None:
                rb_mic.close()
            if rb_team is not None:
                rb_team.close()

            self.loading = False
            self.running = False
            self.mic_db = -100.0
            self.chat_db = -100.0
            self.radar = {"left": "Inactivo", "center": "Inactivo", "right": "Inactivo"}
            self.add_log("[SISTEMA] Motor detenido.", "sys")
            if sys.platform == "win32":
                ctypes.windll.ole32.CoUninitialize()

    def download_all_models_async(self) -> None:
        def _dl() -> None:
            try:
                self.add_log("[INSTALADOR] Verificando Silero VAD v5...", "sys")
                ensure_silero_model(self.models_dir)

                ws_path = self.models_dir / "wespeaker_campp.onnx"
                if not ws_path.exists():
                    self.add_log("[INSTALADOR] Descargando WeSpeaker CAM++ ONNX...", "sys")
                    urllib.request.urlretrieve(WESPEAKER_ONNX_URL, ws_path)

                self.add_log("[INSTALADOR] Verificando Kokoro-82M INT8 y Voces...", "sys")
                ensure_kokoro_assets(self.models_dir)

                self.add_log("[INSTALADOR] Verificando Traductor 200 Idiomas NLLB-200...", "sys")
                ct2_dir = self.models_dir / "nllb_200_ct2_int8"
                snapshot_download(
                    repo_id=NLLB_CT2_REPO, local_dir=str(ct2_dir), local_dir_use_symlinks=False
                )
                tok_file = ct2_dir / "tokenizer.json"
                if not tok_file.exists():
                    hf_hub_download(
                        repo_id=NLLB_TOKENIZER_REPO,
                        filename="tokenizer.json",
                        local_dir=str(ct2_dir),
                        local_dir_use_symlinks=False,
                    )

                self.add_log(
                    f"[INSTALADOR] Verificando Faster-Whisper ({self.config.whisper_model_size})...",
                    "sys",
                )
                WhisperTranslatorEngine(model_size=self.config.whisper_model_size)
                self.add_log(
                    "[INSTALADOR] Todos los modelos e idiomas están listos y verificados.", "rx"
                )
            except Exception as exc:
                self.add_log(f"[ERROR INSTALADOR]: {exc}", "err")

        threading.Thread(target=_dl, daemon=True).start()


CONTROLLER = StitchBackendController()
HTML_FILE = Path(__file__).parent / "ui_stitch.html"


class StitchHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html_bytes = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            since = int(qs.get("since", ["0"])[0])
            self._send_json(
                {
                    "running": CONTROLLER.running,
                    "loading": CONTROLLER.loading,
                    "mic_db": CONTROLLER.mic_db,
                    "chat_db": CONTROLLER.chat_db,
                    "last_latency_ms": CONTROLLER.last_latency_ms,
                    "radar": CONTROLLER.radar,
                    "logs": CONTROLLER.get_logs_since(since),
                    "devices": CONTROLLER.devices_payload,
                    "config": asdict(CONTROLLER.config),
                    "models_status": CONTROLLER.get_models_status(),
                }
            )
            return

        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        data = json.loads(raw) if raw else {}

        if self.path == "/api/toggle":
            CONTROLLER.toggle_engine()
            self._send_json({"ok": True})
            return

        if self.path == "/api/config":
            for key, val in data.items():
                if hasattr(CONTROLLER.config, key):
                    setattr(CONTROLLER.config, key, val)
            if "spatial_3d" in data:
                CONTROLLER.spatial_3d = bool(data["spatial_3d"])
            CONTROLLER.config.save()
            if not data.get("silent", False):
                CONTROLLER.add_log("[SISTEMA] Configuración guardada en config.json", "sys")
            self._send_json({"ok": True})
            return

        if self.path == "/api/action":
            act = data.get("action", "")
            if act == "open_github":
                os.startfile("https://github.com/MCJ-A")
            elif act == "open_linkedin":
                os.startfile("https://www.linkedin.com/in/jhon-cordova-martinez")
            elif act == "open_mmsys":
                subprocess.Popen(["control", "mmsys.cpl"])
            elif act == "download_models":
                CONTROLLER.download_all_models_async()
            elif act == "install_vbcable":
                vb_url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
                tmp_zip = Path(os.environ.get("TEMP", ".")) / "VBCABLE.zip"
                tmp_dir = Path(os.environ.get("TEMP", ".")) / "VBCABLE"
                urllib.request.urlretrieve(vb_url, tmp_zip)
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    zf.extractall(tmp_dir)
                os.startfile(str(tmp_dir / "VBCABLE_Setup_x64.exe"), "runas")
            self._send_json({"ok": True})
            return

        self.send_error(404)


def launch_stitch_window(url: str) -> None:
    edge_paths = [
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        / "Microsoft/Edge/Application/msedge.exe",
    ]
    for edge in edge_paths:
        if edge.exists():
            subprocess.Popen(
                [str(edge), f"--app={url}", "--window-size=1220,820"]
            )
            return
    os.startfile(url)


def main() -> None:
    port = 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), StitchHTTPHandler)
    url = f"http://127.0.0.1:{port}"
    print(f" [ProtoBridge Studio - Stitch Edition] Servidor activo en {url}")
    launch_stitch_window(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


