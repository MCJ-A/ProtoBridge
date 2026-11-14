from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
from typing import Any, BinaryIO

_WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_FRAME_BYTES = 8 * 1024 * 1024

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA


def compute_ws_accept_key(sec_websocket_key: str) -> str:
    """Calcula Sec-WebSocket-Accept según RFC 6455."""
    digest = hashlib.sha1((sec_websocket_key.strip() + _WS_MAGIC_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def pack_media_binary_packet(meta: dict[str, Any], pcm_i16_bytes: bytes = b"") -> bytes:
    """Empaqueta [u32le hdr_len][JSON UTF-8][PCM s16le] para streaming binario sin Base64."""
    hdr_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(hdr_bytes)) + hdr_bytes + pcm_i16_bytes


def unpack_media_binary_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    """Desempaqueta un paquete [u32le hdr_len][JSON UTF-8][PCM s16le]."""
    if len(packet) < 4:
        raise ValueError("Paquete binario incompleto (<4 bytes).")
    (hdr_len,) = struct.unpack("<I", packet[:4])
    if hdr_len > len(packet) - 4:
        raise ValueError("Longitud de cabecera inválida en paquete binario.")
    raw_meta = packet[4 : 4 + hdr_len].decode("utf-8")
    meta = json.loads(raw_meta)
    if not isinstance(meta, dict):
        raise ValueError("La cabecera binaria debe ser un objeto JSON.")
    return meta, packet[4 + hdr_len :]


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    buf = bytearray()
    while len(buf) < count:
        chunk = stream.read(count - len(buf))
        if not chunk:
            raise EOFError("Conexión WebSocket cerrada.")
        buf.extend(chunk)
    return bytes(buf)


def _unmask_payload(payload: bytes, mask_key: bytes) -> bytes:
    if len(mask_key) != 4 or not payload:
        return payload
    arr = bytearray(payload)
    for idx in range(len(arr)):
        arr[idx] ^= mask_key[idx & 3]
    return bytes(arr)


def build_ws_frame(opcode: int, payload: bytes = b"") -> bytes:
    """Construye una trama RFC 6455 desde servidor (sin máscara)."""
    fin_opcode = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length <= 125:
        header = bytes((fin_opcode, length))
    elif length <= 0xFFFF:
        header = struct.pack("!BBH", fin_opcode, 126, length)
    else:
        header = struct.pack("!BBQ", fin_opcode, 127, length)
    return header + payload


class WebSocketConnection:
    """Conexión WebSocket RFC 6455 thread-safe sobre rfile/wfile de BaseHTTPRequestHandler."""

    def __init__(self, rfile: BinaryIO, wfile: BinaryIO) -> None:
        self._rfile = rfile
        self._wfile = wfile
        self._write_lock = threading.Lock()
        self.closed = False

    def send_frame(self, opcode: int, payload: bytes = b"") -> bool:
        frame = build_ws_frame(opcode, payload)
        with self._write_lock:
            if self.closed:
                return False
            try:
                self._wfile.write(frame)
                self._wfile.flush()
                return True
            except OSError:
                self.closed = True
                return False

    def send_text(self, text: str) -> bool:
        return self.send_frame(OPCODE_TEXT, text.encode("utf-8"))

    def send_json(self, payload: dict[str, Any]) -> bool:
        return self.send_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def send_binary(self, data: bytes) -> bool:
        return self.send_frame(OPCODE_BINARY, data)

    def send_media_packet(self, meta: dict[str, Any], pcm_i16_bytes: bytes = b"") -> bool:
        return self.send_binary(pack_media_binary_packet(meta, pcm_i16_bytes))

    def close(self, code: int = 1000) -> None:
        with self._write_lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._wfile.write(build_ws_frame(OPCODE_CLOSE, struct.pack("!H", code)))
                self._wfile.flush()
            except OSError:
                pass

    def read_message(self) -> tuple[int, bytes] | None:
        """Lee el siguiente mensaje completo de texto o binario; responde PING automáticamente."""
        fragments: list[bytes] = []
        message_opcode: int | None = None

        while not self.closed:
            try:
                head = _read_exact(self._rfile, 2)
            except (EOFError, OSError, ValueError):
                self.closed = True
                return None

            b1, b2 = head[0], head[1]
            fin = bool(b1 & 0x80)
            opcode = b1 & 0x0F
            masked = bool(b2 & 0x80)
            length = b2 & 0x7F

            try:
                if length == 126:
                    (length,) = struct.unpack("!H", _read_exact(self._rfile, 2))
                elif length == 127:
                    (length,) = struct.unpack("!Q", _read_exact(self._rfile, 8))
            except (EOFError, OSError):
                self.closed = True
                return None

            if length > _MAX_FRAME_BYTES:
                self.close(1009)
                return None

            try:
                mask_key = _read_exact(self._rfile, 4) if masked else b""
                payload = _read_exact(self._rfile, length) if length > 0 else b""
            except (EOFError, OSError):
                self.closed = True
                return None

            if masked:
                payload = _unmask_payload(payload, mask_key)

            if opcode == OPCODE_CLOSE:
                self.close(1000)
                return None
            if opcode == OPCODE_PING:
                self.send_frame(OPCODE_PONG, payload[:125])
                continue
            if opcode == OPCODE_PONG:
                continue

            if opcode in (OPCODE_TEXT, OPCODE_BINARY):
                message_opcode = opcode
                fragments = [payload]
            elif opcode == OPCODE_CONTINUATION and message_opcode is not None:
                fragments.append(payload)
            else:
                self.close(1002)
                return None

            if fin and message_opcode is not None:
                return message_opcode, b"".join(fragments)

        return None
