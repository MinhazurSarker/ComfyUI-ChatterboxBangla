"""
text.py — Text processing nodes for ComfyUI-ChatterboxBangla

Nodes:
  ChatterboxBanglaSplitText  — smart sentence-level splitter (Priority: । > . > ? > ! > , > length)
  ChatterboxBanglaXMLParser  — parse emotion/style XML tags into chunks + param overrides
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger("ComfyUI-ChatterboxBangla.text")


# ===========================================================================
# NODE 1 — ChatterboxBanglaSplitText
# ===========================================================================

_SPLIT_PRIORITY = [
    r"(?<=[।])\s*",          # 1. Bengali danda
    r"(?<=[\.!?])\s+",        # 2. Sentence-ending punctuation
    r"(?<=[,;])\s+",          # 3. Clause separators (last resort)
]

MAX_CHUNKS = 20


def _smart_split(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks ≤ max_chars using priority hierarchy:
      1. Bengali danda (।)
      2. Sentence end (. ! ?)
      3. Clause (,  ;)
      4. Word boundary fallback
    """
    # Try each split pattern in priority order
    for pattern in _SPLIT_PRIORITY:
        parts = [p.strip() for p in re.split(pattern, text.strip()) if p.strip()]
        if all(len(p) <= max_chars for p in parts):
            return _merge_short(parts, max_chars)

    # Fallback: split at word boundaries
    return _word_split(text, max_chars)


def _merge_short(parts: list[str], max_chars: int) -> list[str]:
    """Merge consecutive short parts to reduce chunk count."""
    merged, current = [], ""
    for p in parts:
        candidate = (current + " " + p).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = p
    if current:
        merged.append(current)
    return merged


def _word_split(text: str, max_chars: int) -> list[str]:
    """Split at word boundaries when no sentence break fits."""
    words, chunks, current = text.split(), [], ""
    for w in words:
        candidate = (current + " " + w).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = w
    if current:
        chunks.append(current)
    return chunks


class ChatterboxBanglaSplitText:
    """
    Intelligently split long Bengali text into TTS-friendly chunks.

    Split priority:
      ।  (Bengali danda — highest priority)
      .  !  ?  (sentence end)
      ,  ;     (clause separator — last resort)
      word boundary fallback

    Returns up to 20 STRING outputs + a chunk count.
    Connect each chunk_N to a separate Generate node for long-form audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "max_chars": (
                    "INT",
                    {
                        "default": 250, "min": 50, "max": 1000, "step": 10,
                        "tooltip": (
                            "Maximum characters per chunk. "
                            "200–300 is recommended for natural TTS pacing."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES  = tuple(["STRING"] * MAX_CHUNKS + ["INT"])
    RETURN_NAMES  = tuple([f"chunk_{i+1}" for i in range(MAX_CHUNKS)] + ["chunk_count"])
    FUNCTION      = "split_text"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = (
        "Split long Bangla text into TTS-friendly chunks. "
        "Split priority: ।  →  . ! ?  →  , ;  →  word boundary."
    )

    def split_text(self, text: str, max_chars: int = 250):
        if not text.strip():
            return tuple([""] * MAX_CHUNKS + [0])

        chunks = _smart_split(text.strip(), max_chars)
        n      = len(chunks)

        if n > MAX_CHUNKS:
            logger.warning(
                f"[SplitText] Got {n} chunks but max is {MAX_CHUNKS}. "
                f"Increase max_chars or split text manually."
            )
            chunks = chunks[:MAX_CHUNKS]
            n      = MAX_CHUNKS

        print(f"[SplitText] {len(text)} chars → {n} chunks")
        padded = (chunks + [""] * MAX_CHUNKS)[:MAX_CHUNKS]
        return tuple(padded + [n])


# ===========================================================================
# NODE 2 — ChatterboxBanglaXMLParser
# ===========================================================================

# Emotion tag → default parameter overrides
_EMOTION_DEFAULTS: dict[str, dict] = {
    "happy":    {"exaggeration": 0.8,  "temperature": 0.9,  "cfg_weight": 0.6},
    "sad":      {"exaggeration": 0.6,  "temperature": 0.7,  "cfg_weight": 0.4},
    "angry":    {"exaggeration": 1.0,  "temperature": 1.0,  "cfg_weight": 0.7},
    "excited":  {"exaggeration": 0.9,  "temperature": 1.1,  "cfg_weight": 0.6},
    "calm":     {"exaggeration": 0.3,  "temperature": 0.6,  "cfg_weight": 0.4},
    "neutral":  {"exaggeration": 0.5,  "temperature": 0.8,  "cfg_weight": 0.5},
    "dramatic": {"exaggeration": 1.2,  "temperature": 0.9,  "cfg_weight": 0.8},
    "whisper":  {"exaggeration": 0.2,  "temperature": 0.6,  "cfg_weight": 0.3},
    "speak":    {"exaggeration": 0.5,  "temperature": 0.8,  "cfg_weight": 0.5},
}

MAX_XML_CHUNKS = 10


class ChatterboxBanglaXMLParser:
    """
    Parse emotion/style XML tags in text into chunks + generation parameter overrides.

    Supported format:
      <speak>
        <happy>এটা দারুণ খবর!</happy>
        <sad exaggeration="0.7">আমি বুঝতে পারছি না।</sad>
        <neutral>এরপর কী হবে?</neutral>
      </speak>

    Supported tags:
      happy, sad, angry, excited, calm, neutral, dramatic, whisper, speak

    Supported attributes (override defaults):
      exaggeration  (0.0–2.0)
      cfg_weight    (0.0–1.0)
      temperature   (0.0–2.0)

    Returns up to 10 text chunks + their parameter overrides as strings.
    Connect chunk_N + params_N to separate Generate nodes.

    Example params string:
      "exaggeration=0.8,cfg_weight=0.6,temperature=0.9"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xml_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "<speak>\n"
                            "  <happy>এটা সত্যিই দারুণ!</happy>\n"
                            "  <neutral>তারপর সে চলে গেল।</neutral>\n"
                            "  <sad>আমি মিস করব।</sad>\n"
                            "</speak>"
                        ),
                        "tooltip": "XML-tagged text with emotion markers",
                    },
                ),
            }
        }

    # 10 text chunks + 10 param strings + count
    RETURN_TYPES  = tuple(["STRING"] * MAX_XML_CHUNKS * 2 + ["INT"])
    RETURN_NAMES  = tuple(
        [f"text_{i+1}"   for i in range(MAX_XML_CHUNKS)] +
        [f"params_{i+1}" for i in range(MAX_XML_CHUNKS)] +
        ["chunk_count"]
    )
    FUNCTION      = "parse_xml"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = (
        "Parse emotion XML tags into text chunks + generation params. "
        "Connect text_N → Generate text field, params_N → (future) param input."
    )

    def parse_xml(self, xml_text: str):
        chunks: list[tuple[str, dict]] = []

        try:
            root = ET.fromstring(xml_text.strip())
        except ET.ParseError:
            # If no root <speak> tag, try wrapping
            try:
                root = ET.fromstring(f"<speak>{xml_text.strip()}</speak>")
            except ET.ParseError as e:
                logger.warning(f"[XMLParser] Parse error: {e}. Returning text as-is.")
                text_out = [xml_text.strip()] + [""] * (MAX_XML_CHUNKS - 1)
                param_out = [""] * MAX_XML_CHUNKS
                return tuple(text_out + param_out + [1])

        for child in root:
            tag  = child.tag.lower()
            text = (child.text or "").strip()
            if not text:
                continue

            # Start with emotion defaults
            params = dict(_EMOTION_DEFAULTS.get(tag, _EMOTION_DEFAULTS["neutral"]))

            # Override with XML attributes
            for attr_name in ("exaggeration", "cfg_weight", "temperature"):
                if attr_name in child.attrib:
                    try:
                        params[attr_name] = float(child.attrib[attr_name])
                    except ValueError:
                        pass

            chunks.append((text, params))

            if len(chunks) >= MAX_XML_CHUNKS:
                break

        if not chunks:
            return tuple([""] * MAX_XML_CHUNKS + [""] * MAX_XML_CHUNKS + [0])

        n = len(chunks)
        texts  = [c[0] for c in chunks] + [""] * (MAX_XML_CHUNKS - n)
        params = [
            ",".join(f"{k}={v:.3f}" for k, v in c[1].items())
            for c in chunks
        ] + [""] * (MAX_XML_CHUNKS - n)

        print(f"[XMLParser] Parsed {n} tagged segments.")
        for i, (t, p) in enumerate(chunks):
            print(f"  [{i+1}] {t[:40]}… | {params[i]}")

        return tuple(texts + params + [n])


import json
import os

class ChatterboxBanglaJSONParser:
    """
    Parse a JSON string or file containing a list of speech segments.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "json_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "[\n"
                            "  {\n"
                            "    \"text\": \"আপনার ব্যবসা কি এখনও আলাদা আলাদা সফটওয়্যার দিয়ে পরিচালনা করছেন?\",\n"
                            "    \"instruction\": \"excited\"\n"
                            "  },\n"
                            "  {\n"
                            "    \"text\": \"এক জায়গায় হিসাব, আরেক জায়গায় স্টক, অন্য কোথাও অনলাইন অর্ডার।\",\n"
                            "    \"instruction\": \"sad\"\n"
                            "  }\n"
                            "]"
                        ),
                        "tooltip": "JSON array of objects. Each object must have a 'text' key.",
                    },
                ),
                "json_file_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: Path to a local .json file to load instead.",
                    },
                ),
            },
        }

    RETURN_TYPES  = ("BATCH_DATA", "INT")
    RETURN_NAMES  = ("BATCH_DATA", "count")
    FUNCTION      = "parse"
    CATEGORY      = "ChatterboxBangla"
    DESCRIPTION   = "Parse a JSON array of text segments for Batch Generate."

    def parse(self, json_text: str = "", json_file_path: str = ""):
        content = json_text.strip()
        file_path = json_file_path.strip()

        if file_path:
            resolved_path = os.path.expanduser(file_path)
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"JSON file not found at: {resolved_path}")
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

        if not content:
            return ([], 0)

        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(f"[JSONParser] Invalid JSON format: {e}")

        if not isinstance(data, list):
            raise ValueError("[JSONParser] JSON root must be an array (list of objects).")

        # Validate entries
        valid_data = []
        for idx, item in enumerate(data):
            if isinstance(item, dict) and "text" in item and item["text"].strip():
                valid_data.append(item)
            else:
                print(f"[JSONParser] Warning: segment at index {idx} ignored (missing 'text' key or not a dictionary).")

        return (valid_data, len(valid_data))

