"""
patch.py — Vocabulary resize patch for BosonLab/chatterbox-bangla

The original ResembleAI chatterbox model uses a 704-token text vocabulary.
BosonLab fine-tuned it with 2530 Bengali tokens. Loading the Bengali weights
into an unpatched model raises a size mismatch error.

This module provides two patch strategies:

  1. FILE PATCH  — for vendored code in chatterbox_backend/
     Modifies tts.py on disk. Applied once by installer.py.

  2. RUNTIME PATCH — for pip chatterbox package
     Monkey-patches T3.__init__ before from_local() runs.
     Automatically restores the original after loading.

Usage:
    # Auto-select strategy after backend is determined:
    from .patch import ensure_patched
    ensure_patched(backend_source, backend_tts_path)
"""
from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger("ComfyUI-ChatterboxBangla.patch")

# ---------------------------------------------------------------------------
# 1. File patch (for vendored tts.py)
# ---------------------------------------------------------------------------

_FILE_PATCH_OLD = (
    '        t3 = T3()\n'
    '        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")'
)
_FILE_PATCH_NEW = (
    '        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")\n'
    '        from .models.t3.modules.t3_config import T3Config\n'
    '        t3 = T3(hp=T3Config('
    'text_tokens_dict_size=t3_state["text_emb.weight"].shape[0]))'
)

_PATCH_MARKER = 'T3Config(text_tokens_dict_size'


def apply_file_patch(tts_path: pathlib.Path) -> bool:
    """
    Patch src/chatterbox_/tts.py or chatterbox_backend/tts.py in place.

    Returns True if patch was applied or already present.
    Returns False if the target string wasn't found (upstream changed).
    """
    if not tts_path.exists():
        logger.warning(f"[patch] tts.py not found at {tts_path}")
        return False

    txt = tts_path.read_text(encoding="utf-8")

    if _PATCH_MARKER in txt:
        logger.info("[patch] File patch already present — skipping.")
        return True

    if _FILE_PATCH_OLD not in txt:
        logger.warning(
            "[patch] Target string not found in tts.py. "
            "Upstream may have changed. File patch skipped."
        )
        return False

    patched = txt.replace(_FILE_PATCH_OLD, _FILE_PATCH_NEW)
    tts_path.write_text(patched, encoding="utf-8")
    logger.info("[patch] ✓ File patch applied (hard-coded 704 → dynamic vocab).")
    return True


# ---------------------------------------------------------------------------
# 2. Runtime patch (for pip chatterbox package)
# ---------------------------------------------------------------------------

def apply_runtime_patch(ChatterboxTTS_cls) -> None:
    """
    Monkey-patch ChatterboxTTS.from_local() so it reads the actual vocab
    size from the safetensors file and passes it to T3() before construction.

    Safe to call multiple times — guards against double-patching.
    """
    if getattr(ChatterboxTTS_cls, "_vocab_patched", False):
        logger.info("[patch] Runtime patch already applied — skipping.")
        return

    _orig_from_local = ChatterboxTTS_cls.from_local.__func__   # unbound

    @classmethod  # type: ignore[misc]
    def _patched_from_local(cls, ckpt_dir, device="cuda"):
        import pathlib as _pl

        ckpt = _pl.Path(ckpt_dir)
        t3_weights = ckpt / "t3_cfg.safetensors"

        vocab_size = None
        if t3_weights.exists():
            try:
                from safetensors.torch import load_file  # type: ignore
                state = load_file(str(t3_weights))
                vocab_size = state["text_emb.weight"].shape[0]
                logger.info(f"[patch] Detected vocab size: {vocab_size}")
            except Exception as exc:
                logger.warning(f"[patch] Runtime vocab probe failed: {exc}.")

        if vocab_size is not None and vocab_size != 704:
            logger.info(
                f"[patch] Applying runtime T3 vocab patch "
                f"(704 → {vocab_size})."
            )
            return _load_with_vocab(cls, _orig_from_local,
                                    ckpt_dir, device, vocab_size)

        return _orig_from_local(cls, ckpt_dir, device=device)

    ChatterboxTTS_cls.from_local = _patched_from_local
    ChatterboxTTS_cls._vocab_patched = True
    logger.info("[patch] ✓ Runtime patch installed on ChatterboxTTS.from_local.")


def _load_with_vocab(cls, orig_from_local, ckpt_dir, device, vocab_size: int):
    """
    Temporarily patch T3.__init__ to use the correct vocab size,
    call the original from_local(), then restore T3.__init__.
    """
    T3 = _resolve_T3()
    if T3 is None:
        logger.warning("[patch] Cannot resolve T3 class — skipping runtime vocab patch.")
        return orig_from_local(cls, ckpt_dir, device=device)

    T3Config = _resolve_T3Config()
    _original_T3_init = T3.__init__

    def _vocab_aware_init(self, hp=None, **kwargs):
        if hp is None and T3Config is not None:
            hp = T3Config(text_tokens_dict_size=vocab_size)
        elif hp is None:
            logger.warning("[patch] T3Config unavailable — using default hp.")
        _original_T3_init(self, hp=hp, **kwargs)

    T3.__init__ = _vocab_aware_init
    try:
        result = orig_from_local(cls, ckpt_dir, device=device)
    finally:
        T3.__init__ = _original_T3_init   # always restore
    return result


def _resolve_T3():
    """Try to import T3 from known package locations."""
    candidates = [
        ("chatterbox.models.t3", "T3"),
        ("chatterbox_.models.t3", "T3"),
        ("chatterbox_backend.models.t3", "T3"),
    ]
    for mod_path, attr in candidates:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)
        except Exception:
            continue
    return None


def _resolve_T3Config():
    """Try to import T3Config from known package locations."""
    candidates = [
        "chatterbox.models.t3.modules.t3_config",
        "chatterbox_.models.t3.modules.t3_config",
        "chatterbox_backend.models.t3.modules.t3_config",
    ]
    for mod_path in candidates:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, "T3Config")
        except Exception:
            continue
    return None
