from __future__ import annotations

from collections.abc import Callable
import ctypes
import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Virtual Key Codes para Windows GetAsyncKeyState
VK_MAP: dict[str, int] = {
    "ctrl": 0x11,
    "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "menu": 0x12,
    "caps_lock": 0x14,
    "capslock": 0x14,
    "capital": 0x14,
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "escape": 0x1B,
    "xbutton1": 0x05,
    "mouse4": 0x05,
    "xbutton2": 0x06,
    "mouse5": 0x06,
}

# Caracteres A-Z y 0-9
for c in range(ord("a"), ord("z") + 1):
    VK_MAP[chr(c)] = ord(chr(c).upper())
for c in range(ord("0"), ord("9") + 1):
    VK_MAP[chr(c)] = c
for f in range(1, 13):
    VK_MAP[f"f{f}"] = 0x6F + f


def parse_hotkey_spec(spec: str) -> tuple[int, ...]:
    """Parsea una cadena de atajo (ej. 'ctrl+shift+m') a una tupla de virtual-key codes."""
    if not spec:
        return ()
    tokens = [t.strip().lower() for t in spec.split("+") if t.strip()]
    vks: list[int] = []
    for token in tokens:
        vk = VK_MAP.get(token)
        if vk is not None:
            vks.append(vk)
        else:
            logger.debug("Tecla no reconocida en especificación de hotkey: %s", token)
    return tuple(vks)


class GlobalHotkeyListener:
    """Escuchador global no bloqueante de hotkeys para conmutación táctica de Mute y PTT.

    Opera en un hilo secundario sin interrumpir la cola de procesamiento de audio ni la GUI.
    """

    def __init__(
        self,
        hotkey_mute_tx: str = "ctrl+shift+m",
        hotkey_mute_rx: str = "ctrl+shift+n",
        hotkey_ptt: str = "caps_lock",
        hotkey_reset_squad: str = "ctrl+shift+r",
        ptt_enabled: bool = False,
        initial_mute_tx: bool = False,
        initial_mute_rx: bool = False,
        on_tx_mute_changed: Callable[[bool], None] | None = None,
        on_rx_mute_changed: Callable[[bool], None] | None = None,
        on_ptt_changed: Callable[[bool], None] | None = None,
        on_reset_squad: Callable[[], None] | None = None,
    ) -> None:
        self.hotkey_tx_vks: tuple[int, ...] = parse_hotkey_spec(hotkey_mute_tx)
        self.hotkey_rx_vks: tuple[int, ...] = parse_hotkey_spec(hotkey_mute_rx)
        self.hotkey_ptt_vks: tuple[int, ...] = parse_hotkey_spec(hotkey_ptt)
        self.hotkey_reset_vks: tuple[int, ...] = parse_hotkey_spec(hotkey_reset_squad)
        self.ptt_enabled: bool = ptt_enabled

        self._lock = threading.Lock()
        self._tx_muted: bool = initial_mute_tx
        self._rx_muted: bool = initial_mute_rx
        self._ptt_active: bool = False

        self._on_tx_mute_changed = on_tx_mute_changed
        self._on_rx_mute_changed = on_rx_mute_changed
        self._on_ptt_changed = on_ptt_changed
        self._on_reset_squad = on_reset_squad

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._prev_tx_combo_down: bool = False
        self._prev_rx_combo_down: bool = False
        self._prev_ptt_down: bool = False
        self._prev_reset_combo_down: bool = False

    @property
    def is_tx_muted(self) -> bool:
        with self._lock:
            return self._tx_muted

    @property
    def is_rx_muted(self) -> bool:
        with self._lock:
            return self._rx_muted

    @property
    def is_ptt_active(self) -> bool:
        with self._lock:
            return self._ptt_active

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_mute_tx(self, muted: bool) -> None:
        changed = False
        with self._lock:
            if self._tx_muted != muted:
                self._tx_muted = muted
                changed = True
        if changed and self._on_tx_mute_changed:
            self._on_tx_mute_changed(muted)

    def set_mute_rx(self, muted: bool) -> None:
        changed = False
        with self._lock:
            if self._rx_muted != muted:
                self._rx_muted = muted
                changed = True
        if changed and self._on_rx_mute_changed:
            self._on_rx_mute_changed(muted)

    def toggle_mute_tx(self) -> bool:
        with self._lock:
            self._tx_muted = not self._tx_muted
            new_state = self._tx_muted
        if self._on_tx_mute_changed:
            self._on_tx_mute_changed(new_state)
        return new_state

    def toggle_mute_rx(self) -> bool:
        with self._lock:
            self._rx_muted = not self._rx_muted
            new_state = self._rx_muted
        if self._on_rx_mute_changed:
            self._on_rx_mute_changed(new_state)
        return new_state

    def set_ptt_active(self, active: bool) -> None:
        changed = False
        with self._lock:
            if self._ptt_active != active:
                self._ptt_active = active
                changed = True
        if changed and self._on_ptt_changed:
            self._on_ptt_changed(active)

    def start(self) -> None:
        if sys.platform != "win32":
            logger.info("GlobalHotkeyListener inactivo: requiere plataforma Windows.")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="pb_hotkeys")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def _is_vk_down(self, vk: int) -> bool:
        try:
            state = ctypes.windll.user32.GetAsyncKeyState(vk)
            return bool(state & 0x8000)
        except (AttributeError, OSError):
            return False

    def _is_combo_down(self, vks: tuple[int, ...]) -> bool:
        if not vks:
            return False
        return all(self._is_vk_down(vk) for vk in vks)

    def _poll_once(self) -> None:
        # Mute TX Toggle (Edge-trigger)
        tx_down = self._is_combo_down(self.hotkey_tx_vks)
        if tx_down and not self._prev_tx_combo_down:
            self.toggle_mute_tx()
        self._prev_tx_combo_down = tx_down

        # Mute RX Toggle (Edge-trigger)
        rx_down = self._is_combo_down(self.hotkey_rx_vks)
        if rx_down and not self._prev_rx_combo_down:
            self.toggle_mute_rx()
        self._prev_rx_combo_down = rx_down

        # Reset de Escuadrón (Edge-trigger)
        reset_down = self._is_combo_down(self.hotkey_reset_vks)
        if reset_down and not self._prev_reset_combo_down:
            if self._on_reset_squad is not None:
                self._on_reset_squad()
        self._prev_reset_combo_down = reset_down

        # PTT (State-trigger)
        if self.ptt_enabled and self.hotkey_ptt_vks:
            ptt_down = self._is_combo_down(self.hotkey_ptt_vks)
            if ptt_down != self._prev_ptt_down:
                self.set_ptt_active(ptt_down)
            self._prev_ptt_down = ptt_down

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            time.sleep(0.033)  # ~30 Hz polling anti-cheat safe sin saturar CPU
