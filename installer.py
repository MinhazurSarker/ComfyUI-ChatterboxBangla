"""
installer.py — One-time backend setup for ComfyUI-ChatterboxBangla

Runs at ComfyUI startup (called from __init__.py).
Checks a sentinel file — if present, skips all work.

Strategy:
  1. Try importing ChatterboxTTS from pip 'chatterbox' package.
     If it works → done (use pip backend).
  2. Otherwise, git-clone chatterbox-finetuning to a temp dir,
     copy src/chatterbox_/ into chatterbox_backend/,
     apply vocab file-patch,
     mark as vendored.

After first successful run, writes chatterbox_backend/.setup_complete
so subsequent startups skip all of this and load instantly.
"""
from __future__ import annotations

import os
import sys
import shutil
import logging
import pathlib
import subprocess
import tempfile

logger = logging.getLogger("ComfyUI-ChatterboxBangla.installer")

_PKG_DIR     = pathlib.Path(__file__).parent
_BACKEND_DIR = _PKG_DIR / "chatterbox_backend"
_SENTINEL    = _BACKEND_DIR / ".setup_complete"
_SOURCE_KEY  = _BACKEND_DIR / ".backend_source"   # "pip" or "vendored"

FINETUNING_REPO = "https://github.com/gokhaneraslan/chatterbox-finetuning"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_backend() -> str:
    """
    Ensure the chatterbox backend is ready.
    Returns "pip" or "vendored" indicating which backend is active.
    Called once per ComfyUI process from __init__.py.
    """
    _BACKEND_DIR.mkdir(parents=True, exist_ok=True)

    if _SENTINEL.exists():
        source = _SOURCE_KEY.read_text().strip() if _SOURCE_KEY.exists() else "pip"
        logger.info(f"[installer] Backend already set up ({source}). Skipping.")
        return source

    logger.info("[installer] First run — setting up chatterbox backend…")
    source = _run_setup()
    _SOURCE_KEY.write_text(source)
    _SENTINEL.touch()
    logger.info(f"[installer] ✓ Setup complete. Backend: {source}")
    return source


# ---------------------------------------------------------------------------
# Internal setup logic
# ---------------------------------------------------------------------------

def _run_setup() -> str:
    """Try pip backend first, then vendor."""
    if _try_pip_backend():
        return "pip"
    return _vendor_backend()


def _try_pip_backend() -> bool:
    """
    Check whether the pip 'chatterbox' package is importable AND
    that the runtime vocab patch can be applied.
    """
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore
        from .patch import apply_runtime_patch
        apply_runtime_patch(ChatterboxTTS)
        print("[ChatterboxBangla] ✓ Using pip chatterbox package as backend.")
        return True
    except Exception as exc:
        logger.info(f"[installer] Pip backend unavailable ({exc}). Falling back to vendored.")
        return False


def _vendor_backend() -> str:
    """
    Clone chatterbox-finetuning, copy src/chatterbox_/ into chatterbox_backend/,
    apply vocab file-patch.
    """
    print("[ChatterboxBangla] Setting up vendored chatterbox backend…")
    print(f"[ChatterboxBangla] Cloning: {FINETUNING_REPO}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)

        # --- Clone ---
        result = subprocess.run(
            ["git", "clone", "--depth", "1", FINETUNING_REPO, str(tmp)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed:\n{result.stderr}\n"
                "Ensure 'git' is installed. On Mac: xcode-select --install"
            )
        print("[ChatterboxBangla] ✓ Clone complete. Copying to backend…")

        # --- Copy src/chatterbox_/ → chatterbox_backend/ ---
        src_root = tmp / "src" / "chatterbox_"
        if not src_root.exists():
            # Try alternate path
            src_root = tmp / "chatterbox_"
        if not src_root.exists():
            raise RuntimeError(
                f"Cannot find chatterbox_ source in cloned repo.\n"
                f"Expected: {src_root}"
            )

        for item in src_root.rglob("*"):
            if item.is_file():
                rel  = item.relative_to(src_root)
                dest = _BACKEND_DIR / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

    print("[ChatterboxBangla] ✓ Files copied to chatterbox_backend/.")

    # --- Apply vocab file-patch ---
    tts_path = _BACKEND_DIR / "tts.py"
    from .patch import apply_file_patch
    ok = apply_file_patch(tts_path)
    if not ok:
        logger.warning(
            "[installer] File patch could not be applied. "
            "Will try runtime patch at load time."
        )

    # --- Install minimal deps needed by vendored code ---
    _install_deps()

    # --- Inject backend dir into sys.path ---
    parent = str(_PKG_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    print("[ChatterboxBangla] ✓ Vendored backend ready.")
    return "vendored"


def _install_deps() -> None:
    """Install packages that vendored chatterbox code needs at inference time."""
    needed = [
        "librosa",
        "soundfile",
        "safetensors",
        "resemble-perth",        # ResembleAI audio watermarking used by ChatterboxTTS
        "huggingface_hub",
    ]
    python = sys.executable
    for pkg in needed:
        check = subprocess.run(
            [python, "-m", "pip", "show", pkg],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            print(f"[ChatterboxBangla] Installing {pkg}…")
            subprocess.run(
                [python, "-m", "pip", "install", pkg, "-q"],
                check=False,
            )
