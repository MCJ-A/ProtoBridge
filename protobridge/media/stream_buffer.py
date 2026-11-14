from __future__ import annotations

import threading
import numpy as np
from numpy.typing import NDArray


class InMemoryAudioRingBuffer:
    """Buffer circular en RAM thread-safe compatible con SileroVADSegmenter."""

    def __init__(self, capacity_seconds: float = 20.0, sample_rate: int = 16000) -> None:
        if capacity_seconds <= 0 or sample_rate <= 0:
            raise ValueError("capacity_seconds y sample_rate deben ser mayores que 0.")
        self.sample_rate = int(sample_rate)
        self.capacity = max(512, int(capacity_seconds * sample_rate))
        self._buf: NDArray[np.float32] = np.zeros((self.capacity,), dtype=np.float32)
        self._lock = threading.Lock()
        self._write_pos = 0
        self._read_pos = 0

    def _clamped_cursors_locked(self) -> tuple[int, int, bool]:
        w = self._write_pos
        r = self._read_pos
        clamped = (w - r) > self.capacity
        if clamped:
            r = w - self.capacity
        return w, r, clamped

    def _copy_locked(self, start_cursor: int, count: int, out: NDArray[np.float32]) -> None:
        idx = start_cursor % self.capacity
        first = min(count, self.capacity - idx)
        out[:first] = self._buf[idx : idx + first]
        rem = count - first
        if rem > 0:
            out[first:count] = self._buf[:rem]

    @property
    def read_cursor(self) -> int:
        with self._lock:
            _, r, _ = self._clamped_cursors_locked()
            return r

    @property
    def available_to_read(self) -> int:
        with self._lock:
            w, r, _ = self._clamped_cursors_locked()
            return max(0, w - r)

    def clear(self) -> None:
        with self._lock:
            self._buf.fill(0.0)
            self._write_pos = 0
            self._read_pos = 0

    def write(self, samples: NDArray[np.float32]) -> int:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = int(arr.shape[0])
        if n == 0:
            return 0
        with self._lock:
            if n >= self.capacity:
                arr = arr[-self.capacity :]
                n = self.capacity
            idx = self._write_pos % self.capacity
            first = min(n, self.capacity - idx)
            self._buf[idx : idx + first] = arr[:first]
            rem = n - first
            if rem > 0:
                self._buf[:rem] = arr[first:]
            self._write_pos += n
            if (self._write_pos - self._read_pos) > self.capacity:
                self._read_pos = self._write_pos - self.capacity
            return n

    def read_into(self, out: NDArray[np.float32]) -> int:
        req = int(out.shape[0])
        with self._lock:
            w, r, clamped = self._clamped_cursors_locked()
            avail = max(0, w - r)
            take = min(req, avail)
            if take > 0:
                self._copy_locked(r, take, out)
            if take > 0 or clamped:
                self._read_pos = r + take
            if take < req:
                out[take:] = 0.0
            return take

    def slice_window(
        self, start_cursor: int, end_cursor: int, preroll_samples: int = 4800
    ) -> NDArray[np.float32]:
        with self._lock:
            w = self._write_pos
            oldest = max(0, w - self.capacity)
            actual_start = max(oldest, start_cursor - preroll_samples)
            actual_end = min(w, end_cursor)
            total = max(0, actual_end - actual_start)
            result: NDArray[np.float32] = np.empty((total,), dtype=np.float32)
            if total > 0:
                self._copy_locked(actual_start, total, result)
            return result
