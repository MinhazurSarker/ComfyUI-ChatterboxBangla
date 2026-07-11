# ComfyUI-ChatterboxBangla

A comprehensive custom node extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) providing state-of-the-art **Bengali (Bangla) and Multilingual Text-to-Speech (TTS)** and **Voice Cloning**.

This package integrates the powerful [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) multilingual foundation model and the [BosonLab/chatterbox-bangla](https://huggingface.co/BosonLab/chatterbox-bangla) fine-tuned weights for natural Bengali speech synthesis.

---

## 🌟 Key Features

* **Zero-shot TTS & Voice Cloning:** Instantly synthesize speech or clone a target voice using a short reference audio clip + its transcript.
* **Smart Language Routing:** Automatically selects the fine-tuned BosonLab model for Bengali inputs, and dynamically switches to the ResembleAI base model for other languages.
* **🚀 Batch Generation (JSON) - *Breakthrough feature*:** Feed in a JSON array of sentences with distinct emotion/style prompts (e.g. `excited`, `whispering`, `excited ending`). The node will loop through them sequentially, optionally save the individual clips on disk, and stitch them into a single audio output with configurable silence gaps.
* **Native Audio I/O:** Uses ComfyUI's standard `AUDIO` tensor format, making it fully compatible with built-in or third-party audio nodes.
* **Fast Session Caching:** Keeps loaded models in memory for instantaneous consecutive generations.
* **Robust macOS/Apple Silicon Support:** Fully optimized for Metal GPU Acceleration (MPS) with automatic PyTorch fallback safety flags.
* **Audio Post-Processing:** Built-in nodes for Loudness Normalization (LUFS), Silence removal, and MP3 exporting.

---

## 🛠 Installation

### 1. Clone the Repository
Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-ChatterboxBangla.git
```

### 2. Install Dependencies
Activate ComfyUI's virtual environment and install the required packages:
```bash
source /path/to/ComfyUI/.venv/bin/activate
pip install -r requirements.txt
```

*Note: The loader node runs an automated dependency check on startup and will auto-install missing requirements.*

### 3. Extra Post-Processing Dependencies (Optional)
To use MP3 Exporting and Loudness Normalization:
```bash
pip install pydub pyloudnorm
# On macOS, install FFmpeg:
brew install ffmpeg
```

---

## 💎 Credits & Attributions

* **[ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox):** Multilingual voice cloning base model architecture.
* **[BosonLab/chatterbox-bangla](https://huggingface.co/BosonLab/chatterbox-bangla):** Fine-tuned weights providing exceptional Bengali expression.

---

## 📦 Node Catalog

### 1. 🎙 Chatterbox Bangla Loader (`ChatterboxBanglaLoader`)
Initializes the model in memory. Automatically downloads weights from HuggingFace to your local cache.
* **Inputs:**
  * `language` (Combo): Choose `bengali` (BosonLab) or any of the 23 other languages (ResembleAI).
  * `device` (Combo): Choose `auto` (auto-detects CUDA/MPS/CPU), `cuda`, `mps`, or `cpu`.
  * `custom_model_path` (String): Optional override path to a local directory or custom HuggingFace repo.
* **Outputs:**
  * `MODEL` (CHATTERBOX_BANGLA_MODEL)

### 2. 🎤 Load Reference Audio (`ChatterboxBanglaLoadReference`)
Loads and prepares a reference audio file for voice cloning.
* **Inputs:**
  * `audio` (AUDIO): Reference audio input.
  * `transcript` (String): Exact transcript of the reference audio (mandatory for voice cloning).
  * `trim_silence` (Boolean): Removes quiet edges.
  * `top_db` (Int): Noise threshold for trimming silence (default: `30`).
* **Outputs:**
  * `REFERENCE_AUDIO`

### 3. ✨ Chatterbox Bangla Generate (`ChatterboxBanglaGenerate`)
Generates speech for a single segment of text.
* **Inputs:**
  * `MODEL`: Loaded Chatterbox model.
  * `text` (String): The Bengali or multilingual text to synthesize.
  * `instruction` (String): Style modifier (e.g. `excited`, `whispering`, `fast`).
  * `REFERENCE_AUDIO`: Optional reference clip for voice cloning.
  * `exaggeration` (Float): Intensity of the emotional instruction (`0.0` - `1.0`).
  * `cfg_weight` (Float): Guidance strength.
  * `temperature` (Float): Sampling randomness.
  * `seed` (Int): Control randomness (0 = random).
* **Outputs:**
  * `AUDIO`

### 4. 🏷 Chatterbox JSON Parser (`ChatterboxBanglaJSONParser`)
Parses structured JSON strings or files into segment lists for batch generation.
* **Inputs:**
  * `json_text` (String): JSON-formatted string or absolute path to a `.json` file on disk.
* **Outputs:**
  * `BATCH_DATA`: Parsed batch instruction.
  * `count` (Int): Total number of segments.

### 5. 📦 Chatterbox Batch Generate (JSON) (`ChatterboxBanglaBatchGenerate`)
Generates multiple audio segments sequentially and stitches them together.
* **Inputs:**
  * `MODEL`: Loaded model.
  * `BATCH_DATA`: Connection from the **JSON Parser** node.
  * `REFERENCE_AUDIO`: Connection from the **Load Reference Audio** node.
  * `speed` (Float): Default speed multiplier.
  * `remove_silence` (Boolean): Strip silence from individual segments.
  * `silence_between` (Float): Seconds of silence padding between merged sentences.
  * `save_to_folder` (String): Optional path (e.g. `~/Downloads/narrations`) to save each individual sentence as a numbered file (`001.wav`, `002.wav`, etc.).
  * `seed` (Int): Random seed.
* **Outputs:**
  * `AUDIO`: Unified merged audio track.

### 6. 🔊 Normalize Loudness (`ChatterboxBanglaNormalize`)
Normalizes target loudness in LUFS.
* **Inputs:** `audio` (AUDIO), `target_lufs` (Float, default: `-16.0`).
* **Outputs:** `AUDIO`.

### 7. ✂️ Remove Silence (`ChatterboxBanglaRemoveSilence`)
Splits and filters out silent intervals from an audio track.
* **Outputs:** `AUDIO`.

### 8. 🔗 Merge Audio (`ChatterboxBanglaMergeAudio`)
Stitches multiple audio tracks together with custom silences.
* **Outputs:** `AUDIO`.

### 9. 💾 Export MP3 (`ChatterboxBanglaExportMP3`)
Saves ComfyUI audio out as an MP3 file.
* **Inputs:** `audio` (AUDIO), `output_path` (String), `bitrate` (Combo, default: `192k`).
* **Outputs:** `path` (String).

---

## 📝 Writing JSON for Batch Generation

To generate long audiobooks or narrations with dynamic styles, configure the **JSON Parser** with a list of dictionaries. Each dictionary must contain a `"text"` key, and can optionally override the voice style or speed:

```json
[
  {
    "text": "আপনার ব্যবসা কি এখনও আলাদা আলাদা সফটওয়্যার দিয়ে পরিচালনা করছেন?",
    "instruction": "this is a question. ask with a higher pitch ending and excited.",
    "speed": 0.95
  },
  {
    "text": "এক জায়গায় হিসাব, আরেক জায়গায় স্টক, অন্য কোথাও অনলাইন অর্ডার।",
    "instruction": "sound like you are giving example of unpleasant staff",
    "speed": 1.0
  },
  {
    "text": "এবার সবকিছু নিয়ে আসুন একটি স্মার্ট প্ল্যাটফর্মে।",
    "instruction": "sound extremely confident and reassuring.",
    "speed": 1.05
  }
]
```

---

## 🚀 Running on macOS (Apple Silicon)

If running on Apple Silicon (M1/M2/M3), you will automatically benefit from **Metal GPU Acceleration (MPS)**. 

To ensure stability across all operations, our package automatically initializes PyTorch's fallback flag:
```python
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```
This guarantees that operations unsupported on GPU are handled seamlessly on CPU without hanging.

---

## 📄 License

This project is licensed under the MIT License.
