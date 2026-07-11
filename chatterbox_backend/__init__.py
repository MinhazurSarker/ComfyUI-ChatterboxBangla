"""
chatterbox_backend/__init__.py — Smart import wrapper

Tries import strategies in order:
  1. Vendored code (chatterbox_backend/tts.py) — populated by installer.py
  2. Pip 'chatterbox' package — if transformers version is compatible
  3. Raises a clear error if neither works

ChatterboxTTS and BACKEND_SOURCE are the public exports.
"""
from __future__ import annotations

import os
import sys
import logging
import pathlib

logger = logging.getLogger("ComfyUI-ChatterboxBangla.backend")

_HERE   = pathlib.Path(__file__).parent
_PKG    = _HERE.parent   # ComfyUI-ChatterboxBangla/

BACKEND_SOURCE: str = "unavailable"
ChatterboxTTS = None     # will be set below

_VENDORED_TTS = _HERE / "tts.py"


# ---------------------------------------------------------------------------
# Strategy 1: vendored tts.py already in chatterbox_backend/
# ---------------------------------------------------------------------------
if _VENDORED_TTS.exists():
    # Add the plugin root to sys.path so relative imports inside
    # chatterbox_backend work correctly (e.g. from .models.t3 import T3)
    _pkg_str = str(_PKG)
    if _pkg_str not in sys.path:
        sys.path.insert(0, _pkg_str)

    try:
        from chatterbox_backend.tts import ChatterboxTTS  # type: ignore[no-redef]
        BACKEND_SOURCE = "vendored"
        logger.info("[backend] Using vendored chatterbox_backend.")
    except Exception as _e:
        logger.warning(f"[backend] Vendored import failed: {_e}")

# ---------------------------------------------------------------------------
# Strategy 2: pip 'chatterbox' package
# ---------------------------------------------------------------------------
if ChatterboxTTS is None:
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore[no-redef]
        BACKEND_SOURCE = "pip"
        logger.info("[backend] Using pip chatterbox package.")
    except Exception as _e:
        logger.warning(f"[backend] Pip chatterbox import failed: {_e}")

# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------
if ChatterboxTTS is None:
    class ChatterboxTTS:  # type: ignore[no-redef]
        """Placeholder that raises a clear error on use."""
        @classmethod
        def from_local(cls, *args, **kwargs):
            raise ImportError(
                "ChatterboxTTS backend is not available.\n\n"
                "Restart ComfyUI once with internet access to auto-install "
                "the backend into chatterbox_backend/.\n\n"
                "Alternatively, run:\n"
                "  pip install chatterbox-tts\n"
                "and restart ComfyUI."
            )
    BACKEND_SOURCE = "unavailable"
    logger.error("[backend] No working ChatterboxTTS import found!")
