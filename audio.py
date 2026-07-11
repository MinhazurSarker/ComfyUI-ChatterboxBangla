"""
audio.py — Audio utility nodes for ComfyUI-ChatterboxBangla

Nodes:
  ChatterboxBanglaLoadReference  — load audio file + transcript for voice cloning
  ChatterboxBanglaNormalize      — loudness normalisation (ITU-R BS.1770)
  ChatterboxBanglaRemoveSilence  — trim silence (librosa)
  ChatterboxBanglaMergeAudio     — concatenate multiple AUDIO clips
  ChatterboxBanglaExportMP3      — export AUDIO as MP3 (requires ffmpeg)

NOTE: Save / Preview are handled by ComfyUI's built-in nodes.
      This package outputs standard AUDIO so those nodes connect directly.
"""
from __future__ import annotations

import os
import hashlib
import logging

import numpy as np
import torch
import torchaudio
import soundfile as sf
import folder_paths

from .utils import (
    SUPPORTED_AUDIO_FORMATS,
    load_audio_file,
    tensor_to_comfy_audio,
    comfy_audio_to_numpy_mono,
)

logger = logging.getLogger("ComfyUI-ChatterboxBangla.audio")


# ===========================================================================
# NODE 1 — ChatterboxBanglaLoadReference
# ===========================================================================

class ChatterboxBanglaLoadReference:
    """
    Prepare reference audio for voice cloning.


    Connect ComfyUI's built-in Load Audio node → AUDIO input here.
    Add the transcript (what's spoken in the audio) for better voice alignment.

    Workflow:
      [Load Audio]  ←  pick any mp3/wav/flac file
           ↓ AUDIO
      [Load Reference Audio]  ←  add transcript text
           ↓ REFERENCE_AUDIO
      [Chatterbox Bangla Generate]

    - Audio is auto-converted to mono 16 kHz float32 internally.
    - Optional silence trimming (recommended for cleaner voice cloning).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Connect ComfyUI's built-in 'Load Audio' node here. "
                            "Accepts any format it supports (wav, mp3, flac, ogg, etc.)"
                        ),
                    },
                ),
                "transcript": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": (
                            "Type what is spoken in the reference audio.\n"
                            "Example:\n"
                            "  আমি বাংলায় কথা বলতে পারি।\n\n"
                            "Helps the model align voice characteristics.\n"
                            "Leave blank if you don't have the transcript."
                        ),
                        "tooltip": (
                            "The text spoken in the reference audio. "
                            "Used for voice alignment when supported by the model."
                        ),
                    },
                ),
            },
            "optional": {
                "trim_silence": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Trim leading/trailing silence (recommended)",
                    },
                ),
                "top_db": (
                    "INT",
                    {
                        "default": 30, "min": 10, "max": 80, "step": 5,
                        "tooltip": "dB below peak to treat as silence",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("REFERENCE_AUDIO",)
    RETURN_NAMES  = ("REFERENCE_AUDIO",)
    FUNCTION      = "load_reference"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = (
        "Prepare reference audio for voice cloning. "
        "Connect ComfyUI's built-in 'Load Audio' node to the AUDIO input, "
        "then add the transcript of what is spoken."
    )

    def load_reference(
        self,
        audio: dict,
        transcript: str = "",
        trim_silence: bool = True,
        top_db: int = 30,
    ):
        # Convert ComfyUI AUDIO → mono 16 kHz float32 numpy
        audio_np, sr = comfy_audio_to_numpy_mono(audio, target_sr=16000)

        # Trim silence
        if trim_silence:
            try:
                import librosa  # type: ignore
                audio_np, _ = librosa.effects.trim(audio_np, top_db=top_db)
            except ImportError:
                logger.warning("[LoadReference] librosa not installed — silence trim skipped.")

        duration = len(audio_np) / sr
        print(
            f"[LoadReference] ✓ {duration:.1f}s @ {sr} Hz  "
            f"transcript={'yes' if transcript.strip() else 'none'}"
        )

        return ({
            "audio_np":    audio_np,
            "sample_rate": sr,
            "source_path": "",      # no file path (came from tensor)
            "transcript":  transcript.strip(),
        },)



# ===========================================================================
# NODE 2 — ChatterboxBanglaNormalize
# ===========================================================================

class ChatterboxBanglaNormalize:
    """
    Loudness normalisation (ITU-R BS.1770-3 / EBU R128).
    Requires pyloudnorm. Falls back to peak normalisation if unavailable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": (
                    "FLOAT",
                    {
                        "default": -23.0, "min": -60.0, "max": 0.0, "step": 0.5,
                        "tooltip": "-23 LUFS = EBU broadcast  |  -14 LUFS = Spotify/YouTube streaming",
                    },
                ),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("AUDIO",)
    FUNCTION      = "normalize"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Loudness normalisation. -23 LUFS = broadcast. -14 LUFS = streaming."

    def normalize(self, audio: dict, target_lufs: float = -23.0):
        audio_np, sr = comfy_audio_to_numpy_mono(audio)

        try:
            import pyloudnorm as pyln  # type: ignore
            meter    = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio_np)
            out      = pyln.normalize.loudness(audio_np, loudness, target_lufs)
            out      = np.clip(out, -1.0, 1.0).astype(np.float32)
            print(
                f"[Normalize] {loudness:.1f} LUFS → {target_lufs:.1f} LUFS"
            )
        except ImportError:
            logger.warning("[Normalize] pyloudnorm not installed — peak normalising.")
            peak = float(np.abs(audio_np).max())
            out  = (audio_np / peak * 0.95).astype(np.float32) if peak > 1e-6 else audio_np

        return (tensor_to_comfy_audio(torch.from_numpy(out), sr),)


# ===========================================================================
# NODE 3 — ChatterboxBanglaRemoveSilence
# ===========================================================================

class ChatterboxBanglaRemoveSilence:
    """Remove or trim silence using librosa."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "top_db": (
                    "INT",
                    {
                        "default": 30, "min": 10, "max": 80, "step": 5,
                        "tooltip": "dB below peak treated as silence",
                    },
                ),
            },
            "optional": {
                "trim_mode": (
                    ["edges_only", "all_silence"],
                    {
                        "default": "edges_only",
                        "tooltip": (
                            "edges_only: trim leading/trailing silence only. "
                            "all_silence: also collapse internal gaps."
                        ),
                    },
                ),
                "gap_ms": (
                    "INT",
                    {
                        "default": 50, "min": 0, "max": 500, "step": 10,
                        "tooltip": "Gap to leave between speech segments (all_silence mode only)",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("AUDIO",)
    FUNCTION      = "remove_silence"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Trim or remove silence using librosa."

    def remove_silence(self, audio: dict, top_db: int = 30,
                       trim_mode: str = "edges_only", gap_ms: int = 50):
        try:
            import librosa  # type: ignore
        except ImportError:
            logger.warning("[RemoveSilence] librosa not installed — returning unchanged.")
            return (audio,)

        audio_np, sr = comfy_audio_to_numpy_mono(audio)

        if trim_mode == "edges_only":
            trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
        else:
            intervals = librosa.effects.split(audio_np, top_db=top_db)
            if len(intervals) == 0:
                trimmed = audio_np
            else:
                gap_samples = int(sr * gap_ms / 1000)
                gap_silence = np.zeros(gap_samples, dtype=np.float32)
                parts = [audio_np[s:e] for s, e in intervals]
                pieces = []
                for i, p in enumerate(parts):
                    pieces.append(p)
                    if i < len(parts) - 1 and gap_samples > 0:
                        pieces.append(gap_silence)
                trimmed = np.concatenate(pieces)

        before = len(audio_np) / sr
        after  = len(trimmed)  / sr
        print(f"[RemoveSilence] {before:.2f}s → {after:.2f}s (saved {before-after:.2f}s)")
        return (tensor_to_comfy_audio(torch.from_numpy(trimmed.astype(np.float32)), sr),)


# ===========================================================================
# NODE 4 — ChatterboxBanglaMergeAudio
# ===========================================================================

class ChatterboxBanglaMergeAudio:
    """Concatenate up to 10 AUDIO clips with an optional silence gap between them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio_1": ("AUDIO",)},
            "optional": {
                **{f"audio_{i}": ("AUDIO",) for i in range(2, 11)},
                "gap_ms": (
                    "INT",
                    {
                        "default": 300, "min": 0, "max": 3000, "step": 50,
                        "tooltip": "Silence gap between clips (milliseconds)",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("AUDIO",)
    FUNCTION      = "merge_audio"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Concatenate up to 10 AUDIO clips into one. Useful for audiobook pipelines."

    def merge_audio(self, audio_1, gap_ms: int = 300, **kwargs):
        clips = [audio_1] + [kwargs.get(f"audio_{i}") for i in range(2, 11)]
        clips = [c for c in clips if c is not None]

        sr          = clips[0]["sample_rate"]
        gap_samples = int(sr * gap_ms / 1000)
        silence     = torch.zeros(1, 1, gap_samples) if gap_samples > 0 else None
        tensors     = []

        for i, clip in enumerate(clips):
            w = clip["waveform"]
            if w.dim() == 2:
                w = w.unsqueeze(0)
            if clip["sample_rate"] != sr:
                w = torchaudio.functional.resample(w, clip["sample_rate"], sr)
            tensors.append(w)
            if silence is not None and i < len(clips) - 1:
                tensors.append(silence)

        merged = torch.cat(tensors, dim=2)
        total  = merged.shape[-1] / sr
        print(f"[MergeAudio] {len(clips)} clips merged → {total:.1f}s")
        return ({"waveform": merged, "sample_rate": sr},)


# ===========================================================================
# NODE 5 — ChatterboxBanglaExportMP3
# ===========================================================================

class ChatterboxBanglaExportMP3:
    """
    Export AUDIO as MP3 file.
    Requires: pip install pydub  +  brew install ffmpeg
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": (
                    "STRING",
                    {"default": "ChatterboxBangla/output", "multiline": False},
                ),
                "bitrate": (
                    ["128k", "192k", "256k", "320k"],
                    {"default": "192k"},
                ),
            }
        }

    RETURN_TYPES  = ()
    OUTPUT_NODE   = True
    FUNCTION      = "export_mp3"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Export AUDIO as MP3. Requires: pip install pydub  +  brew install ffmpeg"

    def export_mp3(self, audio: dict, filename_prefix: str = "ChatterboxBangla/output",
                   bitrate: str = "192k"):
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError:
            raise ImportError(
                "pydub is required for MP3 export.\n"
                "Run: pip install pydub\n"
                "And: brew install ffmpeg"
            )

        out_dir  = folder_paths.get_output_directory()
        p_dir    = os.path.dirname(filename_prefix)
        p_name   = os.path.basename(filename_prefix)
        save_dir = os.path.join(out_dir, p_dir) if p_dir else out_dir
        os.makedirs(save_dir, exist_ok=True)

        counter = 1
        while True:
            fpath = os.path.join(save_dir, f"{p_name}_{counter:04d}.mp3")
            if not os.path.exists(fpath):
                break
            counter += 1

        audio_np, sr = comfy_audio_to_numpy_mono(audio)
        pcm = (audio_np * 32767).astype(np.int16)

        seg = AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        seg.export(fpath, format="mp3", bitrate=bitrate)
        print(f"[ExportMP3] ✓ → {fpath} ({bitrate})")
        return {}
