"""
nodes.py — ComfyUI-ChatterboxBangla node aggregator

Imports all node classes from their respective modules.
This file exists so __init__.py has a single import surface.
"""
from .loader    import ChatterboxBanglaLoader
from .generator import ChatterboxBanglaGenerate, ChatterboxBanglaBatchGenerate
from .audio     import (
    ChatterboxBanglaLoadReference,
    ChatterboxBanglaNormalize,
    ChatterboxBanglaRemoveSilence,
    ChatterboxBanglaMergeAudio,
    ChatterboxBanglaExportMP3,
)
from .text      import (
    ChatterboxBanglaSplitText,
    ChatterboxBanglaXMLParser,
    ChatterboxBanglaJSONParser,
)

__all__ = [
    "ChatterboxBanglaLoader",
    "ChatterboxBanglaGenerate",
    "ChatterboxBanglaBatchGenerate",
    "ChatterboxBanglaLoadReference",
    "ChatterboxBanglaNormalize",
    "ChatterboxBanglaRemoveSilence",
    "ChatterboxBanglaMergeAudio",
    "ChatterboxBanglaExportMP3",
    "ChatterboxBanglaSplitText",
    "ChatterboxBanglaXMLParser",
    "ChatterboxBanglaJSONParser",
]
