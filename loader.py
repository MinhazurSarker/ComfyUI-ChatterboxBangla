"""
loader.py — ChatterboxBanglaLoader node

Language-based model routing:
  Bangla  →  BosonLab/chatterbox-bangla   (Bengali fine-tune)
  English, Arabic, Chinese, ... (any of 23 supported)
          →  ResembleAI/chatterbox         (original multilingual)

Both models are cached independently in MODEL_CACHE.
Switching language loads the correct model automatically.
"""
from __future__ import annotations

import os
import logging

import folder_paths

from .utils import (
    auto_detect_device,
    model_for_language,
    LANGUAGES,
    BANGLA_MODEL,
    BASE_MODEL,
)
from .downloader import (
    ensure_model_downloaded,
    DEFAULT_MODEL_REPO,
    BASE_MODEL_REPO,
)
from .patch import apply_runtime_patch


logger = logging.getLogger("ComfyUI-ChatterboxBangla.loader")

# ---------------------------------------------------------------------------
# Model cache — one entry per (repo_id, device)
# Persists for the lifetime of the ComfyUI process.
# ---------------------------------------------------------------------------
MODEL_CACHE: dict[str, object] = {}


class ChatterboxBanglaLoader:
    """
    Load the correct Chatterbox model for the chosen language.

    Language routing (automatic):
      Bangla  →  BosonLab/chatterbox-bangla   (Bengali fine-tune, 2530-token vocab)
      English, Arabic, Chinese, Hindi, ...
              →  ResembleAI/chatterbox          (23-language base model)

    Both models auto-download on first use and are cached for the session.
    You can use two Loader nodes in the same workflow if you need both languages.

    Manual override:
      Set model_name = "manual override" and enter a HuggingFace repo ID or
      local folder path in custom_model_path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "language": (
                    LANGUAGES,
                    {
                        "default": "Bangla",
                        "tooltip": (
                            "Select output language.\n"
                            "Bangla → BosonLab/chatterbox-bangla (fine-tuned)\n"
                            "All others → ResembleAI/chatterbox (23-language base)"
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "mps", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "auto → CUDA on Nvidia / MPS on Apple Silicon / CPU fallback",
                    },
                ),
            },
            "optional": {
                "custom_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Override: HuggingFace repo ID (e.g. MyOrg/my-model) "
                            "or absolute path to a local model folder. "
                            "Leave blank to use automatic language routing."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES  = ("CHATTERBOX_BANGLA_MODEL",)
    RETURN_NAMES  = ("MODEL",)
    FUNCTION      = "load_model"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = (
        "Load Chatterbox TTS model for the selected language.\n"
        "Bangla → BosonLab fine-tune | English/Arabic/... → ResembleAI base.\n"
        "Auto-downloads and caches both models independently."
    )

    def load_model(self, language: str, device: str, custom_model_path: str = ""):
        # --- Resolve repo / path ---
        if custom_model_path.strip():
            repo_or_path = custom_model_path.strip()
            routing_note = f"manual override: {repo_or_path}"
        else:
            repo_or_path = model_for_language(language)
            if repo_or_path == BANGLA_MODEL:
                routing_note = f"Bangla → {BANGLA_MODEL}"
            else:
                routing_note = f"{language} → {BASE_MODEL}"

        resolved_device = auto_detect_device() if device == "auto" else device
        cache_key       = f"{repo_or_path}::{resolved_device}"

        # --- Cache hit ---
        if cache_key in MODEL_CACHE:
            print(
                f"[ChatterboxBangla] ✓ Cache hit | {routing_note} | {resolved_device}"
            )
            return (MODEL_CACHE[cache_key],)

        print(f"[ChatterboxBangla] Loading: {routing_note} | device={resolved_device}")

        # --- Resolve local dir ---
        if os.path.isdir(repo_or_path):
            model_dir = repo_or_path
        else:
            model_dir = ensure_model_downloaded(repo_or_path)

        # --- Import ChatterboxTTS (pip or vendored) ---
        from .chatterbox_backend import ChatterboxTTS, BACKEND_SOURCE

        if BACKEND_SOURCE == "unavailable":
            raise ImportError(
                "ChatterboxTTS is not available.\n"
                "Restart ComfyUI with internet access to auto-install the backend."
            )

        # Apply vocab patch for pip backend
        if BACKEND_SOURCE == "pip":
            apply_runtime_patch(ChatterboxTTS)

        # --- Load ---
        print(f"[ChatterboxBangla] Loading weights from: {model_dir}")
        model = ChatterboxTTS.from_local(model_dir, device=resolved_device)

        # Attach metadata (used by generator for display/debug)
        model._cb_language = language
        model._cb_device   = resolved_device
        model._cb_repo     = repo_or_path
        model._cb_backend  = BACKEND_SOURCE

        print(
            f"[ChatterboxBangla] ✓ Ready | SR={model.sr} Hz | "
            f"language={language} | backend={BACKEND_SOURCE}"
        )

        MODEL_CACHE[cache_key] = model
        return (model,)
