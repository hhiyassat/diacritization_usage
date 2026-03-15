# Arabic Diacritization Model

A single-file Arabic text diacritization tool using a character-level Transformer, optionally fused with AraBERT contextual embeddings for improved accuracy.

---

## Overview

This script takes undiacritized Arabic text and predicts full diacritics (tashkeel) using a trained neural model. It supports two modes:

- **Interactive mode** — type sentences directly in the terminal
- **File mode** — process an entire `.txt` file and save the output

The model is a character-level Transformer that optionally integrates `aubmindlab/bert-base-arabertv2` (AraBERT) contextual embeddings via a learned gating mechanism. The active architecture is auto-detected from the checkpoint.

---

## Requirements

```bash
pip install torch transformers
```

A GPU is recommended but not required — the script will fall back to CPU automatically.

---

## Checkpoint

Download the pretrained checkpoint from Google Drive and place it at:

```
checkpoints/best.pt
```

**Download link:** https://drive.google.com/file/d/1k7ql9nIVrzMLHVi3MVVhd9iFTm1gJpGR/view?usp=sharing

The checkpoint contains:
- `model` — model weights (state dict)
- `itos` — character vocabulary (index-to-string)
- `cfg` — hyperparameters used during training

---

## Usage

### Interactive Mode

```bash
python diacritizer.py
```

You will be prompted to type Arabic sentences one at a time:

```
اكتب جملة (أو exit): ذهب الولد إلى المدرسة
🔹 النتيجة: ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ
```

Type `exit` to quit.

### File Mode

```bash
python diacritizer.py input.txt
```

The script reads each line from `input.txt`, diacritizes it, and writes the result to `diacritized.txt` in the current working directory.

---

## How It Works

1. **Preprocessing** — input text is canonicalized (NFKC normalization, tatweel removal) and stripped of any existing diacritics before inference.
2. **Chunking** — long texts are split into word-boundary-aligned chunks that fit within the model's positional embedding limit.
3. **Inference** — the model predicts one diacritic tag per character from a 12-class tagset: `Ø F D K TF TD TK S SH SHF SHD SHK`.
4. **Rendering** — predicted tags are converted back to Unicode combining marks and appended to each base character.

### Tag Reference

| Tag | Diacritic |
|-----|-----------|
| `Ø` | No diacritic (sukun-free) |
| `F` | Fatha ( َ ) |
| `D` | Damma ( ُ ) |
| `K` | Kasra ( ِ ) |
| `S` | Sukun ( ْ ) |
| `SH` | Shadda ( ّ ) |
| `SHF` | Shadda + Fatha |
| `SHD` | Shadda + Damma |
| `SHK` | Shadda + Kasra |
| `TF` | Tanwin Fath ( ً ) |
| `TD` | Tanwin Damm ( ٌ ) |
| `TK` | Tanwin Kasr ( ٍ ) |

---

## Model Architecture

Two model variants are supported, auto-detected from the checkpoint:

### `CharTransformer`
A standard character-level Transformer encoder with learned positional embeddings and a linear classification head.

### `CharTransformerWithAraBERT`
Extends `CharTransformer` by injecting token-aligned AraBERT embeddings at the character level. A learned sigmoid gate blends character embeddings with AraBERT context:

```
fused = gate_char * char_h + gate_context * arabert_h
```

This variant requires an internet connection on first run to download the AraBERT model weights from Hugging Face.

---

## Project Structure

```
.
├── diacritizer.py       # Main script (this file)
└── checkpoints/
    └── best.pt          # Pretrained model checkpoint
```

---

## Notes

- The script auto-configures all model hyperparameters from the checkpoint — no manual config needed.
- Input text may be fully undiacritized or partially diacritized; existing diacritics are stripped before inference.
- AraBERT tokenization is performed on the stripped (undiacritized) text. Offset mappings are used to align subword embeddings back to character positions.
