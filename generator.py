"""
generator.py — ChatterboxBanglaGenerate node

Inputs:
  MODEL           — from ChatterboxBanglaLoader
                    (Loader auto-routes Bangla→BosonLab, others→ResembleAI)
  REFERENCE_AUDIO — from ChatterboxBanglaLoadReference (optional, for voice cloning)
  text            — the text to synthesise (Bangla, English, or any loaded language)
  instruction     — style/delivery instruction (e.g. "speak slowly", "be expressive")
  exaggeration    — emotion intensity (0–2)
  cfg_weight      — classifier-free guidance (0–1)
  temperature     — sampling temperature (0–2)
  seed            — 0 = random

Output:
  AUDIO           — standard ComfyUI AUDIO dict
                    → connects directly to ComfyUI's built-in
                      Preview Audio / Save Audio / any audio node

NOTE: Language selection is on the Loader node, not here.
      The correct model (BosonLab or ResembleAI) is already loaded
      before this node runs.
"""
from __future__ import annotations

import os
import time
import logging
import tempfile

import numpy as np
import torch
import soundfile as sf

from .utils import LANGUAGES, tensor_to_comfy_audio

logger = logging.getLogger("ComfyUI-ChatterboxBangla.generator")

# Instruction keyword → parameter nudges
# If the user types one of these words in the instruction field,
# we adjust generation parameters automatically.
INSTRUCTION_HINTS: dict[str, dict] = {
    "slow":        {"temperature": -0.2, "exaggeration": -0.1},
    "slowly":      {"temperature": -0.2, "exaggeration": -0.1},
    "fast":        {"temperature":  0.2},
    "quickly":     {"temperature":  0.2},
    "expressive":  {"exaggeration":  0.3, "cfg_weight": 0.1},
    "excited":     {"exaggeration":  0.4, "temperature":  0.1},
    "calm":        {"exaggeration": -0.2, "cfg_weight": -0.1},
    "sad":         {"exaggeration":  0.2, "temperature": -0.1},
    "happy":       {"exaggeration":  0.3, "temperature":  0.1},
    "neutral":     {"exaggeration": -0.2},
    "whisper":     {"exaggeration": -0.3, "temperature": -0.2},
    "dramatic":    {"exaggeration":  0.5},
    "monotone":    {"exaggeration": -0.4, "cfg_weight":  0.2},
}


def _apply_instruction(instruction: str, exaggeration: float,
                       cfg_weight: float, temperature: float):
    """
    Parse instruction text for known keywords and nudge generation params.
    Clamps all values to valid ranges after adjustments.
    """
    if not instruction.strip():
        return exaggeration, cfg_weight, temperature

    lower = instruction.lower()
    for keyword, adjustments in INSTRUCTION_HINTS.items():
        if keyword in lower:
            exaggeration += adjustments.get("exaggeration", 0.0)
            cfg_weight   += adjustments.get("cfg_weight",   0.0)
            temperature  += adjustments.get("temperature",  0.0)
            logger.info(
                f"[generate] Instruction keyword '{keyword}' → "
                f"exaggeration={exaggeration:.2f} cfg={cfg_weight:.2f} temp={temperature:.2f}"
            )

    exaggeration = float(np.clip(exaggeration, 0.0, 2.0))
    cfg_weight   = float(np.clip(cfg_weight,   0.0, 1.0))
    temperature  = float(np.clip(temperature,  0.01, 2.0))
    return exaggeration, cfg_weight, temperature


class ChatterboxBanglaGenerate:
    """
    Generate Bengali TTS from text.

    Voice cloning: connect REFERENCE_AUDIO from ChatterboxBanglaLoadReference.
    Zero-shot:     leave REFERENCE_AUDIO disconnected.

    Output AUDIO connects to ComfyUI's built-in Preview Audio / Save Audio nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CHATTERBOX_BANGLA_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "আমি বাংলায় কথা বলতে পারি। "
                            "এটি একটি পরীক্ষামূলক বাক্য।"
                        ),
                        "tooltip": "The text to synthesise (Bengali, English, or mixed)",
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": (
                            "Style instruction, e.g.:\n"
                            "  speak slowly\n"
                            "  sound excited\n"
                            "  be calm and clear\n"
                            "Leave blank for default style."
                        ),
                        "tooltip": (
                            "Delivery style instruction. Keywords like 'slow', 'expressive', "
                            "'calm', 'excited', 'dramatic' automatically adjust generation params."
                        ),
                    },
                ),
            },
            "optional": {
                "REFERENCE_AUDIO": (
                    "REFERENCE_AUDIO",
                    {
                        "tooltip": (
                            "Connect ChatterboxBanglaLoadReference for voice cloning. "
                            "Disconnect for zero-shot synthesis."
                        ),
                    },
                ),
                "exaggeration": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                        "display": "slider",
                        "tooltip": "Emotion intensity. 0=flat, 0.5=natural, 2=very expressive",
                    },
                ),
                "cfg_weight": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                        "display": "slider",
                        "tooltip": "Classifier-free guidance strength",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8, "min": 0.01, "max": 2.0, "step": 0.05,
                        "display": "slider",
                        "tooltip": "Sampling temperature. Lower = more stable, higher = more varied",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 0xFFFFFFFF,
                        "tooltip": "Random seed. 0 = new random seed each run",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("AUDIO",)
    FUNCTION      = "generate"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = (
        "Generate TTS using the model loaded by ChatterboxBanglaLoader. "
        "Language is selected in the Loader (Bangla→BosonLab, others→ResembleAI). "
        "Connect REFERENCE_AUDIO for voice cloning. "
        "Output AUDIO works with ComfyUI's built-in Preview Audio / Save Audio nodes."
    )

    @classmethod
    def IS_CHANGED(cls, MODEL, text, instruction="",
                   REFERENCE_AUDIO=None, exaggeration=0.5, cfg_weight=0.5,
                   temperature=0.8, seed=0):
        return seed if seed != 0 else time.time()

    def generate(
        self,
        MODEL,
        text: str,
        instruction: str    = "",
        REFERENCE_AUDIO     = None,
        exaggeration: float = 0.5,
        cfg_weight: float   = 0.5,
        temperature: float  = 0.8,
        seed: int           = 0,
    ):
        if not text.strip():
            raise ValueError("[ChatterboxBanglaGenerate] Text cannot be empty.")

        # --- Seed ---
        actual_seed = seed if seed != 0 else int(time.time()) % 0xFFFFFFFF
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)
        np.random.seed(actual_seed % (2 ** 31))
        language_tag = getattr(MODEL, "_cb_language", "unknown")
        print(f"[ChatterboxBanglaGenerate] Seed={actual_seed}  model_language={language_tag}")

        # --- Apply instruction hints ---
        exaggeration, cfg_weight, temperature = _apply_instruction(
            instruction, exaggeration, cfg_weight, temperature
        )
        if instruction.strip():
            print(
                f"[ChatterboxBanglaGenerate] Instruction: \"{instruction.strip()[:60]}\" "
                f"→ exag={exaggeration:.2f} cfg={cfg_weight:.2f} temp={temperature:.2f}"
            )

        # --- Prepare reference audio file (if voice cloning) ---
        ref_path     = None
        tmp_wav_file = None

        if REFERENCE_AUDIO is not None:
            audio_np   = REFERENCE_AUDIO.get("audio_np")
            sr_ref     = REFERENCE_AUDIO.get("sample_rate", 16000)
            src_path   = REFERENCE_AUDIO.get("source_path", "")
            transcript = REFERENCE_AUDIO.get("transcript", "")

            if transcript:
                print(f"[ChatterboxBanglaGenerate] Reference transcript: \"{transcript[:60]}…\"")

            # Use source .wav directly if already 16 kHz
            if src_path and src_path.lower().endswith(".wav") and sr_ref == 16000:
                ref_path = src_path
            else:
                tmp_wav_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="cbangla_ref_"
                )
                sf.write(tmp_wav_file.name, audio_np, sr_ref)
                ref_path = tmp_wav_file.name

            print(f"[ChatterboxBanglaGenerate] Voice-clone mode | ref={ref_path}")
        else:
            print("[ChatterboxBanglaGenerate] Zero-shot mode")

        # --- Text preview ---
        preview = text[:80] + "…" if len(text) > 80 else text
        print(f'[ChatterboxBanglaGenerate] Text: "{preview}"')

        # --- Build generate() kwargs ---
        kwargs: dict = {
            "exaggeration": exaggeration,
            "cfg_weight":   cfg_weight,
            "temperature":  temperature,
        }
        if ref_path is not None:
            kwargs["audio_prompt_path"] = ref_path

        # Some model versions accept audio_prompt_transcript
        if REFERENCE_AUDIO is not None:
            transcript = REFERENCE_AUDIO.get("transcript", "")
            if transcript:
                kwargs["audio_prompt_transcript"] = transcript

        # --- Generate ---
        try:
            wav = MODEL.generate(text, **kwargs)
        except TypeError as exc:
            # Gracefully degrade if model doesn't support some kwargs
            logger.warning(
                f"[generate] generate() rejected kwargs: {exc}\n"
                "Retrying with minimal args…"
            )
            minimal: dict = {}
            if ref_path is not None:
                minimal["audio_prompt_path"] = ref_path
            wav = MODEL.generate(text, **minimal)
        finally:
            # Always clean up temp file
            if tmp_wav_file is not None:
                try:
                    os.unlink(tmp_wav_file.name)
                except OSError:
                    pass

        sr = MODEL.sr
        print(
            f"[ChatterboxBanglaGenerate] ✓ Done. "
            f"shape={wav.shape}  sr={sr}  "
            f"duration={wav.shape[-1]/sr:.2f}s"
        )
        return (tensor_to_comfy_audio(wav, sr),)


import json

class ChatterboxBanglaBatchGenerate:
    """
    Batch generate speech from a JSON array of segments.
    Example JSON:
    [
      {
        "text": "আপনার ব্যবসা কি এখনও আলাদা আলাদা সফটওয়্যার দিয়ে পরিচালনা করছেন?",
        "instruction": "excited"
      },
      {
        "text": "এক জায়গায় হিসাব, আরেক জায়গায় স্টক, অন্য কোথাও অনলাইন অর্ডার।",
        "instruction": "sad"
      }
    ]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CHATTERBOX_BANGLA_MODEL",),
                "BATCH_DATA": ("BATCH_DATA", {
                    "tooltip": "Connect the output of the JSON Parser node here."
                }),
            },
            "optional": {
                "REFERENCE_AUDIO": (
                    "REFERENCE_AUDIO",
                    {
                        "tooltip": (
                            "Connect ChatterboxBanglaLoadReference for voice cloning. "
                            "Disconnect for zero-shot synthesis."
                        ),
                    },
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Default delivery style instruction (unless overridden in JSON)",
                    },
                ),
                "exaggeration": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "Default emotion intensity",
                    },
                ),
                "cfg_weight": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Default classifier-free guidance strength",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8, "min": 0.01, "max": 2.0, "step": 0.05,
                        "tooltip": "Default sampling temperature",
                    },
                ),
                "silence_between": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Seconds of silence between merged segments",
                    },
                ),
                "save_to_folder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Path to local folder to save individual wav segments (e.g. ~/Downloads/my_speech)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 0xFFFFFFFF,
                        "tooltip": "Random seed. 0 = random",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("AUDIO",)
    FUNCTION      = "generate_batch"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Synthesize and merge a batch of speech segments defined in a JSON array."

    @classmethod
    def IS_CHANGED(cls, MODEL, BATCH_DATA, REFERENCE_AUDIO=None, instruction="", exaggeration=0.5, cfg_weight=0.5, temperature=0.8, silence_between=0.5, save_to_folder="", seed=0):
        return seed if seed != 0 else time.time()

    def generate_batch(
        self,
        MODEL,
        BATCH_DATA: list[dict],
        REFERENCE_AUDIO     = None,
        instruction: str    = "",
        exaggeration: float = 0.5,
        cfg_weight: float   = 0.5,
        temperature: float  = 0.8,
        silence_between: float = 0.5,
        save_to_folder: str = "",
        seed: int           = 0,
    ):
        if not BATCH_DATA:
            raise ValueError("[ChatterboxBanglaBatchGenerate] BATCH_DATA is empty or not connected.")
        segments = BATCH_DATA

        # Prepare save folder if specified
        save_dir = None
        if save_to_folder.strip():
            save_dir = os.path.expanduser(save_to_folder.strip())
            os.makedirs(save_dir, exist_ok=True)
            print(f"[ChatterboxBanglaBatchGenerate] Individual segments will be saved to: {save_dir}")

        # Seed setup
        actual_seed = seed if seed != 0 else int(time.time()) % 0xFFFFFFFF
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)
        np.random.seed(actual_seed % (2 ** 31))

        # Prepare reference audio path (if voice cloning)
        ref_path     = None
        tmp_wav_file = None

        if REFERENCE_AUDIO is not None:
            audio_np = REFERENCE_AUDIO.get("audio_np")
            sr_ref   = REFERENCE_AUDIO.get("sample_rate", 16000)
            src_path = REFERENCE_AUDIO.get("source_path", "")

            # Use source .wav directly if already 16 kHz
            if src_path and src_path.lower().endswith(".wav") and sr_ref == 16000:
                ref_path = src_path
            else:
                tmp_wav_file = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="cbangla_batch_ref_"
                )
                sf.write(tmp_wav_file.name, audio_np, sr_ref)
                ref_path = tmp_wav_file.name

        output_sr = MODEL.sr
        generated_clips = []

        try:
            for idx, item in enumerate(segments):
                if not isinstance(item, dict) or "text" not in item:
                    continue

                text_segment = item["text"].strip()
                if not text_segment:
                    continue

                # Parameter overrides from JSON segment if present, otherwise default
                seg_inst = item.get("instruction", instruction)
                seg_exag = float(item.get("exaggeration", exaggeration))
                seg_cfg  = float(item.get("cfg_weight", cfg_weight))
                seg_temp = float(item.get("temperature", temperature))

                seg_exag, seg_cfg, seg_temp = _apply_instruction(
                    seg_inst, seg_exag, seg_cfg, seg_temp
                )

                kwargs = {
                    "exaggeration": seg_exag,
                    "cfg_weight":   seg_cfg,
                    "temperature":  seg_temp,
                }
                if ref_path is not None:
                    kwargs["audio_prompt_path"] = ref_path
                if REFERENCE_AUDIO is not None:
                    transcript = REFERENCE_AUDIO.get("transcript", "")
                    if transcript:
                        kwargs["audio_prompt_transcript"] = transcript

                print(f"[ChatterboxBanglaBatchGenerate] Segment {idx+1}/{len(segments)}: \"{text_segment[:40]}…\"")
                
                try:
                    clip = MODEL.generate(text_segment, **kwargs)
                except TypeError as exc:
                    minimal = {}
                    if ref_path is not None:
                        minimal["audio_prompt_path"] = ref_path
                    clip = MODEL.generate(text_segment, **minimal)

                # Convert to 1D float32 numpy array
                if isinstance(clip, torch.Tensor):
                    clip = clip.cpu().numpy()
                
                if clip.ndim == 3:
                    clip = clip[0]
                if clip.ndim == 2:
                    clip = clip.mean(axis=0) if clip.shape[0] > 1 else clip[0]

                if clip.dtype == np.int16:
                    clip = clip.astype(np.float32) / 32768.0
                elif clip.dtype != np.float32:
                    clip = clip.astype(np.float32)

                generated_clips.append(clip)

                # Save to folder if requested
                if save_dir:
                    file_name = f"{idx+1:03d}.wav"
                    file_path = os.path.join(save_dir, file_name)
                    sf.write(file_path, clip, output_sr)

        finally:
            if tmp_wav_file is not None:
                try:
                    os.unlink(tmp_wav_file.name)
                except OSError:
                    pass

        if not generated_clips:
            raise RuntimeError("No audio clips were successfully generated in batch.")

        # Merge segments with silence padding
        silence_samples = int(silence_between * output_sr)
        silence_padding = np.zeros(silence_samples, dtype=np.float32)

        merged_audio = []
        for i, clip in enumerate(generated_clips):
            merged_audio.append(clip)
            if i < len(generated_clips) - 1 and silence_samples > 0:
                merged_audio.append(silence_padding)

        final_merged = np.concatenate(merged_audio)
        print(f"[ChatterboxBanglaBatchGenerate] Batch completed. Merged duration: {len(final_merged)/output_sr:.1f}s")

        return (tensor_to_comfy_audio(final_merged, output_sr),)

