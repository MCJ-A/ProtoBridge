from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np
from numpy.typing import NDArray

HEADER_INTS: int = 4
HEADER_BYTES: int = HEADER_INTS * 8


@dataclass(frozen=True)
class RingBufferStats:
    capacity_samples: int
    available_read: int
    available_write: int
    write_cursor: int
    read_cursor: int
    overruns: int
    underruns: int


class SPSCRingBuffer:
    def __init__(
        self,
        name: str,
        capacity_seconds: float = 30.0,
        sample_rate: int = 16000,
        create: bool = True,
    ) -> None:
        if capacity_seconds <= 0 or sample_rate <= 0:
            raise ValueError("capacity_seconds y sample_rate deben ser mayores que 0.")

        self.name: str = name
        self.sample_rate: int = sample_rate
        self.capacity: int = int(capacity_seconds * sample_rate)
        self._data_bytes: int = self.capacity * np.dtype(np.float32).itemsize
        self._total_bytes: int = HEADER_BYTES + self._data_bytes
        self._owner: bool = create

        if create:
            try:
                stale = shared_memory.SharedMemory(name=self.name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self._total_bytes
            )
        else:
            self._shm = shared_memory.SharedMemory(name=self.name, create=False)

        self._header: NDArray[np.int64] = np.ndarray(
            (HEADER_INTS,), dtype=np.int64, buffer=self._shm.buf, offset=0
        )
        self._buffer: NDArray[np.float32] = np.ndarray(
            (self.capacity,),
            dtype=np.float32,
            buffer=self._shm.buf,
            offset=HEADER_BYTES,
        )

        if create:
            self._header[:] = 0
            self._buffer[:] = 0.0

    @property
    def write_cursor(self) -> int:
        return int(self._header[0])

    @property
    def read_cursor(self) -> int:
        return int(self._header[1])

    @property
    def available_to_read(self) -> int:
        return max(0, int(self._header[0] - self._header[1]))

    @property
    def available_to_write(self) -> int:
        return max(0, self.capacity - self.available_to_read)

    def stats(self) -> RingBufferStats:
        w = int(self._header[0])
        r = int(self._header[1])
        avail_r = max(0, w - r)
        return RingBufferStats(
            capacity_samples=self.capacity,
            available_read=avail_r,
            available_write=max(0, self.capacity - avail_r),
            write_cursor=w,
            read_cursor=r,
            overruns=int(self._header[2]),
            underruns=int(self._header[3]),
        )

    def write(self, samples: NDArray[np.float32]) -> int:
        n: int = int(samples.shape[0])
        if n == 0:
            return 0
        if n > self.capacity:
            samples = samples[-self.capacity :]
            n = self.capacity

        w: int = int(self._header[0])
        r: int = int(self._header[1])

        overflow: int = (w + n) - (r + self.capacity)
        if overflow > 0:
            self._header[1] = r + overflow
            self._header[2] += 1

        start_idx: int = w % self.capacity
        first_part: int = min(n, self.capacity - start_idx)
        self._buffer[start_idx : start_idx + first_part] = samples[:first_part]

        second_part: int = n - first_part
        if second_part > 0:
            self._buffer[0:second_part] = samples[first_part:]

        self._header[0] = w + n
        return n

    def read_into(self, out: NDArray[np.float32]) -> int:
        requested: int = int(out.shape[0])
        w: int = int(self._header[0])
        r: int = int(self._header[1])
        avail: int = max(0, w - r)
        to_read: int = min(requested, avail)

        if to_read > 0:
            start_idx: int = r % self.capacity
            first_part: int = min(to_read, self.capacity - start_idx)
            out[:first_part] = self._buffer[start_idx : start_idx + first_part]

            second_part: int = to_read - first_part
            if second_part > 0:
                out[first_part:to_read] = self._buffer[0:second_part]

            self._header[1] = r + to_read

        if to_read < requested:
            out[to_read:] = 0.0
            if avail == 0:
                self._header[3] += 1

        return to_read

    def slice_window(
        self, start_cursor: int, end_cursor: int, preroll_samples: int = 4800
    ) -> NDArray[np.float32]:
        w: int = int(self._header[0])
        oldest_valid: int = max(0, w - self.capacity)
        actual_start: int = max(oldest_valid, start_cursor - preroll_samples)
        actual_end: int = min(w, end_cursor)

        total: int = max(0, actual_end - actual_start)
        result: NDArray[np.float32] = np.empty((total,), dtype=np.float32)
        if total == 0:
            return result

        start_idx: int = actual_start % self.capacity
        first_part: int = min(total, self.capacity - start_idx)
        result[:first_part] = self._buffer[start_idx : start_idx + first_part]

        second_part: int = total - first_part
        if second_part > 0:
            result[first_part:] = self._buffer[0:second_part]

        return result

    def close(self) -> None:
        self._shm.close()
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
