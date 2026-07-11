"""
downloader.py — Model resolution for ComfyUI-ChatterboxBangla

Check order:
  1. Look directly in the HuggingFace local cache folder
     (~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/)
     If found → return that path immediately. Zero network calls.
  2. Only if NOT found locally → download to HF cache via huggingface_hub.

This means:
  - Models already downloaded by any app (transformers, diffusers, etc.)
    are detected and used automatically — no re-download.
  - The cache directory is never modified structurally; HF manages it normally.
  - Fully offline after first download.
"""
from __future__ import annotations

import os
import logging
import pathlib

logger = logging.getLogger("ComfyUI-ChatterboxBangla.downloader")

DEFAULT_MODEL_REPO = "BosonLab/chatterbox-bangla"
BASE_MODEL_REPO    = "ResembleAI/chatterbox"

KNOWN_MODELS = {
    DEFAULT_MODEL_REPO: "Bangla fine-tune (Bengali TTS)",
    BASE_MODEL_REPO:    "Original multilingual (23 languages)",
}

# Default HF cache root (respects HF_HOME / HUGGINGFACE_HUB_CACHE env overrides)
def _hf_cache_root() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    )


def _repo_id_to_cache_name(repo_id: str) -> str:
    """'BosonLab/chatterbox-bangla' → 'models--BosonLab--chatterbox-bangla'"""
    return "models--" + repo_id.replace("/", "--")


def _find_in_hf_cache(repo_id: str) -> str | None:
    """
    Look for the model in the local HF cache.
    Returns the snapshot directory path if present, otherwise None.
    """
    cache_root    = _hf_cache_root()
    model_folder  = cache_root / _repo_id_to_cache_name(repo_id)
    snapshots_dir = model_folder / "snapshots"

    if not snapshots_dir.is_dir():
        return None

    # Find the latest snapshot directory
    for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
        if snapshot.is_dir():
            logger.info(f"[downloader] Found in HF cache: {snapshot}")
            return str(snapshot)

    return None


def ensure_model_downloaded(repo_id: str = DEFAULT_MODEL_REPO) -> str:
    """
    Return the local directory path for a model.

    1. Check HF cache folder → return immediately if present (no network).
    2. Download to HF cache only if not found locally.
    """
    # --- Step 1: check local HF cache (no network at all) ---
    cached_path = _find_in_hf_cache(repo_id)
    if cached_path:
        print(f"[ChatterboxBangla] ✓ Using cached model: {cached_path}")
        return cached_path

    # --- Step 2: not found locally → download ---
    print(f"[ChatterboxBangla] '{repo_id}' not found in local cache.")
    print(f"[ChatterboxBangla] Downloading to: {_hf_cache_root()}")

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        raise ImportError(
            "huggingface_hub is required.\n"
            "Run: pip install huggingface_hub"
        )

    try:
        path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["*.gitattributes", "audios/*", "*.md"],
        )
        print(f"[ChatterboxBangla] ✓ Download complete: {path}")
        return path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{repo_id}'.\n"
            f"Check your internet connection and try again.\n"
            f"Error: {e}"
        ) from e
