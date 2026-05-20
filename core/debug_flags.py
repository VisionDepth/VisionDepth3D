# core/debug_flags.py

_DEBUG_ENABLED = False


def set_debug_enabled(enabled: bool):
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(enabled)


def is_debug_enabled() -> bool:
    return bool(_DEBUG_ENABLED)


def debug_print(*args, **kwargs):
    if _DEBUG_ENABLED:
        print(*args, **kwargs)