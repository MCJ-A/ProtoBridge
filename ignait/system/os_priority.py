from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import ctypes
import logging
import sys

logger = logging.getLogger(__name__)

# Windows Process Creation / Priority Flags
HIGH_PRIORITY_CLASS = 0x00000080
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
THREAD_PRIORITY_HIGHEST = 2


def set_high_process_priority() -> bool:
    """Eleva la prioridad del proceso a nivel HIGH_PRIORITY_CLASS en Windows.

    Garantiza latencia determinista en DSP de audio y captura VAD frente a carga de CPU/GPU del juego.
    Retorna True si la prioridad fue elevada con éxito, False en caso contrario.
    """
    if sys.platform != "win32":
        logger.debug("Ajuste de prioridad OS omitido: plataforma no es Windows.")
        return False

    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()

        # Intentar HIGH_PRIORITY_CLASS (128)
        success = kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
        if success:
            logger.info("Prioridad de proceso establecida en HIGH_PRIORITY_CLASS.")
            return True

        # Fallback a ABOVE_NORMAL_PRIORITY_CLASS (32768)
        fallback = kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS)
        if fallback:
            logger.warning("Prioridad de proceso establecida en ABOVE_NORMAL_PRIORITY_CLASS (fallback).")
            return True

        logger.warning("No se pudo elevar la prioridad del proceso con SetPriorityClass.")
        return False
    except (AttributeError, OSError) as exc:
        logger.warning("Excepción al intentar ajustar la prioridad del proceso: %s", exc)
        return False


def set_thread_priority_realtime() -> bool:
    """Eleva la prioridad del hilo llamador a THREAD_PRIORITY_HIGHEST en Windows.

    Recomendado para los hilos de captura de audio y DSP de streaming.
    Retorna True si fue exitoso, False en caso contrario.
    """
    if sys.platform != "win32":
        return False

    try:
        kernel32 = ctypes.windll.kernel32
        thread_handle = kernel32.GetCurrentThread()
        success = kernel32.SetThreadPriority(thread_handle, THREAD_PRIORITY_HIGHEST)
        return bool(success)
    except (AttributeError, OSError) as exc:
        logger.debug("No se pudo elevar la prioridad del hilo: %s", exc)
        return False


@contextmanager
def mmcss_pro_audio() -> Iterator[bool]:
    """Registra el hilo actual en Windows MMCSS bajo el perfil 'Pro Audio' (RAII).

    Garantiza la liberación determinista del descriptor con AvRevertMmThreadCharacteristics
    al salir del ámbito, únicamente cuando el handle retornado es válido (no NULL/0).
    """
    if sys.platform != "win32":
        yield False
        return

    revert_fn = None
    task_handle = None
    try:
        avrt = ctypes.windll.Avrt
        set_fn = avrt.AvSetMmThreadCharacteristicsW
        revert_fn = avrt.AvRevertMmThreadCharacteristics

        if getattr(set_fn, "argtypes", "UNSET") is None:
            set_fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
            set_fn.restype = ctypes.c_void_p
        if getattr(revert_fn, "argtypes", "UNSET") is None:
            revert_fn.argtypes = [ctypes.c_void_p]
            revert_fn.restype = ctypes.c_int

        task_index = ctypes.c_uint32(0)
        task_handle = set_fn("Pro Audio", ctypes.byref(task_index))
    except (AttributeError, OSError) as exc:
        logger.warning("No se pudo registrar el hilo en MMCSS Pro Audio: %s", exc)
        task_handle = None

    registered = bool(task_handle) and task_handle != 0
    try:
        yield registered
    finally:
        if registered and revert_fn is not None and task_handle:
            try:
                revert_fn(task_handle)
            except (AttributeError, OSError) as exc:
                logger.warning("Error al revertir características MMCSS del hilo: %s", exc)
