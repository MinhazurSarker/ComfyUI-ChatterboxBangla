"""
__init__.py — ComfyUI-ChatterboxBangla

Registers all nodes with ComfyUI.
Runs the one-time backend installer before importing nodes.
"""
from __future__ import annotations

import os
import logging
import traceback

# Enable PyTorch MPS fallback for stable Apple Silicon execution
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger("ComfyUI-ChatterboxBangla")

# ---------------------------------------------------------------------------
# One-time backend setup (idempotent after first run)
# ---------------------------------------------------------------------------
try:
    from .installer import ensure_backend
    _BACKEND_SOURCE = ensure_backend()
    logger.info(f"[ChatterboxBangla] Backend: {_BACKEND_SOURCE}")
except Exception as _e:
    logger.error(
        f"[ChatterboxBangla] Backend installer failed: {_e}\n"
        f"{traceback.format_exc()}"
    )
    _BACKEND_SOURCE = "unavailable"

# ---------------------------------------------------------------------------
# Import nodes
# ---------------------------------------------------------------------------
from .nodes import (
    ChatterboxBanglaLoader,
    ChatterboxBanglaGenerate,
    ChatterboxBanglaBatchGenerate,
    ChatterboxBanglaLoadReference,
    ChatterboxBanglaNormalize,
    ChatterboxBanglaRemoveSilence,
    ChatterboxBanglaMergeAudio,
    ChatterboxBanglaExportMP3,
    ChatterboxBanglaSplitText,
    ChatterboxBanglaXMLParser,
    ChatterboxBanglaJSONParser,
)

# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    # Core — load & generate
    "ChatterboxBanglaLoader":        ChatterboxBanglaLoader,
    "ChatterboxBanglaLoadReference": ChatterboxBanglaLoadReference,
    "ChatterboxBanglaGenerate":      ChatterboxBanglaGenerate,
    "ChatterboxBanglaBatchGenerate": ChatterboxBanglaBatchGenerate,

    # Audio utilities
    "ChatterboxBanglaNormalize":     ChatterboxBanglaNormalize,
    "ChatterboxBanglaRemoveSilence": ChatterboxBanglaRemoveSilence,
    "ChatterboxBanglaMergeAudio":    ChatterboxBanglaMergeAudio,
    "ChatterboxBanglaExportMP3":     ChatterboxBanglaExportMP3,

    # Text utilities
    "ChatterboxBanglaSplitText":     ChatterboxBanglaSplitText,
    "ChatterboxBanglaXMLParser":     ChatterboxBanglaXMLParser,
    "ChatterboxBanglaJSONParser":    ChatterboxBanglaJSONParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Core
    "ChatterboxBanglaLoader":        "🎙 Chatterbox Bangla Loader",
    "ChatterboxBanglaLoadReference": "🎤 Load Reference Audio",
    "ChatterboxBanglaGenerate":      "✨ Chatterbox Bangla Generate",
    "ChatterboxBanglaBatchGenerate": "📦 Chatterbox Batch Generate (JSON)",

    # Audio
    "ChatterboxBanglaNormalize":     "🔊 Normalize Loudness",
    "ChatterboxBanglaRemoveSilence": "✂️  Remove Silence",
    "ChatterboxBanglaMergeAudio":    "🔗 Merge Audio",
    "ChatterboxBanglaExportMP3":     "💾 Export MP3",

    # Text
    "ChatterboxBanglaSplitText":     "📝 Split Text",
    "ChatterboxBanglaXMLParser":     "🏷  XML Emotion Parser",
    "ChatterboxBanglaJSONParser":    "🏷  Chatterbox JSON Parser",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
