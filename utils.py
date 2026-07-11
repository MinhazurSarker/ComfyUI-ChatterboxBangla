"""
utils.py — shared helpers for ComfyUI-ChatterboxBangla
"""
from __future__ import annotations

import os
import logging
from typing import Tuple

import numpy as np
import torch
import torchaudio
import soundfile as sf

logger = logging.getLogger("ComfyUI-ChatterboxBangla")

SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".aac"]

# ---------------------------------------------------------------------------
# Language → model routing
# ---------------------------------------------------------------------------
# BosonLab/chatterbox-bangla : Bangla (Bengali) only
# ResembleAI/chatterbox      : 23 languages (everything except Bangla)

BANGLA_MODEL  = "BosonLab/chatterbox-bangla"
BASE_MODEL    = "ResembleAI/chatterbox"

# Full language list shown in the Loader dropdown.
# Order: Bangla first (recommended), then the 23 ResembleAI-supported languages.
LANGUAGES = [
    "Bangla",       # → BosonLab/chatterbox-bangla
    "English",      # → ResembleAI/chatterbox
    "Arabic",
    "Chinese",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Italian",
    "Japanese",
    "Korean",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Turkish",
]

# Languages that need the Bangla fine-tuned model
_BANGLA_LANGUAGES = {"Bangla"}

def model_for_language(language: str) -> str:
    """
    Return the correct HuggingFace repo ID for the given language.
      Bangla  →  BosonLab/chatterbox-bangla
      *       →  ResembleAI/chatterbox
    """
    return BANGLA_MODEL if language in _BANGLA_LANGUAGES else BASE_MODEL


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def auto_detect_device() -> str:
    """Detect best available device: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Audio format conversion
# ---------------------------------------------------------------------------

def tensor_to_comfy_audio(wav, sr: int) -> dict:
    """
    Any waveform shape → ComfyUI AUDIO dict.
    Output: {"waveform": Tensor[B=1, C=1, T], "sample_rate": int}
    """
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0).unsqueeze(0)   # → (1,1,T)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)                 # → (1,C,T)
    return {"waveform": wav.cpu(), "sample_rate": sr}


def comfy_audio_to_numpy_mono(audio: dict, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    ComfyUI AUDIO dict → mono float32 numpy at target_sr.
    """
    waveform: torch.Tensor = audio["waveform"]
    sr: int = audio["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform[0]   # (C, T)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]

    audio_np = waveform.float().numpy()

    if sr != target_sr:
        try:
            import librosa  # type: ignore
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            t = torch.from_numpy(audio_np).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            audio_np = t[0].numpy()

    return audio_np.astype(np.float32), target_sr


def load_audio_file(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load any supported audio file → mono float32 at target_sr.
    Tries soundfile first, falls back to torchaudio.
    """
    try:
        audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
        audio_np = audio_np.T   # (C, T)
    except Exception:
        try:
            waveform, sr = torchaudio.load(path)
            audio_np = waveform.numpy()
        except Exception as e:
            raise RuntimeError(
                f"Cannot load audio '{path}': {e}\n"
                "Ensure ffmpeg is installed: brew install ffmpeg"
            ) from e

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=0) if audio_np.shape[0] > 1 else audio_np[0]

    if sr != target_sr:
        try:
            import librosa  # type: ignore
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            t = torch.from_numpy(audio_np).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            audio_np = t[0].numpy()

    peak = float(np.abs(audio_np).max())
    if peak > 1e-6:
        audio_np = audio_np / peak

    return audio_np.astype(np.float32), target_sr
