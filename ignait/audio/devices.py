from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pyaudiowpatch as pyaudio


@dataclass(frozen=True)
class AudioEndpoint:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    is_loopback: bool
    is_default_input: bool
    is_default_output: bool


@dataclass(frozen=True)
class RoutingTopology:
    physical_mic: Optional[AudioEndpoint]
    physical_headphones: Optional[AudioEndpoint]
    virtual_mic_sink_cable_a: Optional[AudioEndpoint]
    teammate_voip_source_cable_b: Optional[AudioEndpoint]


class WASAPIDeviceManager:
    def __init__(self) -> None:
        self._pa = pyaudio.PyAudio()
        try:
            self._wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:
            self._pa.terminate()
            raise RuntimeError("WASAPI no está disponible en este sistema Windows.") from exc

    def close(self) -> None:
        self._pa.terminate()

    def list_wasapi_endpoints(self) -> list[AudioEndpoint]:
        wasapi_idx: int = int(self._wasapi_info["index"])
        default_in: int = int(self._wasapi_info.get("defaultInputDevice", -1))
        default_out: int = int(self._wasapi_info.get("defaultOutputDevice", -1))

        endpoints: list[AudioEndpoint] = []
        for idx in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(idx)
            if int(dev.get("hostApi", -1)) != wasapi_idx:
                continue

            endpoints.append(
                AudioEndpoint(
                    index=int(dev["index"]),
                    name=str(dev["name"]),
                    max_input_channels=int(dev["maxInputChannels"]),
                    max_output_channels=int(dev["maxOutputChannels"]),
                    default_sample_rate=int(float(dev["defaultSampleRate"])),
                    is_loopback=bool(dev.get("isLoopbackDevice", False)),
                    is_default_input=(int(dev["index"]) == default_in),
                    is_default_output=(int(dev["index"]) == default_out),
                )
            )
        return endpoints

    def is_endpoint_valid(self, endpoint: AudioEndpoint) -> bool:
        """Verifica si un endpoint responde a la apertura WASAPI sin error -9996 / -9999."""
        try:
            if endpoint.is_loopback or endpoint.max_input_channels > 0:
                ch = max(1, endpoint.max_input_channels if endpoint.max_input_channels > 0 else 2)
                st = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=endpoint.default_sample_rate,
                    input=True,
                    input_device_index=endpoint.index,
                    frames_per_buffer=512,
                    start=False,
                )
                st.close()
                return True
            if endpoint.max_output_channels > 0:
                ch = max(1, endpoint.max_output_channels)
                st = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=endpoint.default_sample_rate,
                    output=True,
                    output_device_index=endpoint.index,
                    frames_per_buffer=512,
                    start=False,
                )
                st.close()
                return True
        except Exception:
            return False
        return False

    def auto_detect_topology(self) -> RoutingTopology:
        endpoints = self.list_wasapi_endpoints()

        # 1. Micrófono físico (Prioriza ROG CARNYX o Default Input verificado)
        physical_mic: Optional[AudioEndpoint] = next(
            (
                ep
                for ep in endpoints
                if not ep.is_loopback
                and ep.max_input_channels > 0
                and "carnyx" in ep.name.lower()
                and self.is_endpoint_valid(ep)
            ),
            None,
        )
        if physical_mic is None:
            physical_mic = next(
                (
                    ep
                    for ep in endpoints
                    if ep.is_default_input
                    and not ep.is_loopback
                    and "cable" not in ep.name.lower()
                    and self.is_endpoint_valid(ep)
                ),
                None,
            )
        if physical_mic is None:
            physical_mic = next(
                (
                    ep
                    for ep in endpoints
                    if not ep.is_loopback
                    and ep.max_input_channels > 0
                    and self.is_endpoint_valid(ep)
                ),
                None,
            )

        # 2. Auriculares físicos principales (Prioriza ROG Theta Ultimate 7.1 o Default Output verificado)
        physical_headphones: Optional[AudioEndpoint] = next(
            (
                ep
                for ep in endpoints
                if not ep.is_loopback
                and ep.max_output_channels > 0
                and "theta" in ep.name.lower()
                and self.is_endpoint_valid(ep)
            ),
            None,
        )
        if physical_headphones is None:
            physical_headphones = next(
                (
                    ep
                    for ep in endpoints
                    if ep.is_default_output
                    and not ep.is_loopback
                    and "cable" not in ep.name.lower()
                    and self.is_endpoint_valid(ep)
                ),
                None,
            )
        if physical_headphones is None:
            physical_headphones = next(
                (
                    ep
                    for ep in endpoints
                    if not ep.is_loopback
                    and ep.max_output_channels > 0
                    and self.is_endpoint_valid(ep)
                ),
                None,
            )

        # 3. Bus de inyección de tu voz traducida hacia el juego (VB-Cable verificado)
        cable_candidates = [
            ep
            for ep in endpoints
            if ep.max_output_channels > 0
            and not ep.is_loopback
            and "cable" in ep.name.lower()
            and self.is_endpoint_valid(ep)
        ]
        virtual_mic_sink: Optional[AudioEndpoint] = next(
            (ep for ep in cable_candidates if "cable input" in ep.name.lower()),
            None,
        )
        if virtual_mic_sink is None and cable_candidates:
            virtual_mic_sink = cable_candidates[0]

        # 4. Bus Loopback aislado para capturar el chat de voz de compañeros
        loopback_candidates = [
            ep
            for ep in endpoints
            if ep.is_loopback
            and self.is_endpoint_valid(ep)
        ]
        teammate_source: Optional[AudioEndpoint] = next(
            (
                ep
                for ep in loopback_candidates
                if any(
                    k in ep.name.lower()
                    for k in ("cable-b", "steam streaming speakers", "oculus virtual")
                )
            ),
            None,
        )
        if teammate_source is None and loopback_candidates:
            teammate_source = loopback_candidates[0]

        return RoutingTopology(
            physical_mic=physical_mic,
            physical_headphones=physical_headphones,
            virtual_mic_sink_cable_a=virtual_mic_sink,
            teammate_voip_source_cable_b=teammate_source,
        )
