#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unicodedata
import contextlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# =========================
# 0) Global device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1) Diacritics tag system
# =========================
AR_DIACR = {
    "F":  "\u064E", "D":  "\u064F", "K":  "\u0650", "S":  "\u0652",
    "SH": "\u0651", "TF": "\u064B", "TD": "\u064C", "TK": "\u064D",
}

RENDER_ORDER = [
    AR_DIACR["SH"],
    AR_DIACR["F"], AR_DIACR["D"], AR_DIACR["K"],
    AR_DIACR["S"],
    AR_DIACR["TF"], AR_DIACR["TD"], AR_DIACR["TK"],
]

TAGS = ["Ø", "F", "D", "K", "TF", "TD", "TK", "S", "SH", "SHF", "SHD", "SHK"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG_BASE = {i: t for t, i in TAG2ID.items()}

def id2tag_safe(idx: int) -> str:
    return ID2TAG_BASE.get(int(idx), "Ø")

def tag_to_marks(tag: str) -> List[str]:
    table = {
        "Ø":  [], "F":  [AR_DIACR["F"]], "D":  [AR_DIACR["D"]], "K":  [AR_DIACR["K"]],
        "TF": [AR_DIACR["TF"]], "TD": [AR_DIACR["TD"]], "TK": [AR_DIACR["TK"]],
        "S":  [AR_DIACR["S"]], "SH": [AR_DIACR["SH"]],
        "SHF":[AR_DIACR["SH"], AR_DIACR["F"]],
        "SHD":[AR_DIACR["SH"], AR_DIACR["D"]],
        "SHK":[AR_DIACR["SH"], AR_DIACR["K"]],
    }
    return table.get(tag, [])

def canonicalize(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    return s.replace("\u0640", "") 

def is_combining(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Me")

def strip_diacritics(s: str) -> str:
    return "".join(ch for ch in s if not is_combining(ch))

# =========================
# 2) Model Definitions
# =========================
ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv2"
arabert_tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL_NAME, use_fast=True)
arabert_model = AutoModel.from_pretrained(ARABERT_MODEL_NAME).to(DEVICE)
arabert_model.eval()

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, layers, dim_ff, dropout, num_tags, pos_size: int):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(pos_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_tags)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, original_sentences=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.dropout(self.char_emb(x) + self.pos_emb(pos))
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.classifier(h)

class CharTransformerWithAraBERT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, layers, dim_ff, dropout, num_tags, pos_size: int):
        super().__init__()
        self.d_model, self.arabert_dim = d_model, 768
        self.char_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(pos_size, d_model)
        self.arabert_proj = nn.Linear(self.arabert_dim, d_model) if self.arabert_dim != d_model else nn.Identity()
        self.gate_char, self.gate_context = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder, self.classifier, self.dropout = nn.TransformerEncoder(enc_layer, num_layers=layers), nn.Linear(d_model, num_tags), nn.Dropout(dropout)

    def _get_arabert_char_embeddings(self, sentences: List[str]) -> List[torch.Tensor]:
        embs = []
        for s in sentences:
            s_clean = strip_diacritics(canonicalize(s))
            if not s_clean:
                embs.append(torch.empty((0, self.arabert_dim), device=DEVICE))
                continue
            tok = arabert_tokenizer(s_clean, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True, truncation=True)
            with torch.no_grad():
                out = arabert_model(input_ids=tok["input_ids"].to(DEVICE), attention_mask=tok["attention_mask"].to(DEVICE))
            last_hidden, offsets = out.last_hidden_state.squeeze(0), tok["offset_mapping"][0].tolist()
            char_level = torch.zeros(len(s_clean), self.arabert_dim, device=DEVICE)
            for t_i, (a, b) in enumerate(offsets):
                if t_i < last_hidden.shape[0]:
                    for cpos in range(a, b):
                        if 0 <= cpos < len(s_clean): char_level[cpos] = last_hidden[t_i]
            embs.append(char_level)
        return embs

    def forward(self, x: torch.Tensor, mask: torch.Tensor, original_sentences: List[str]):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        char_h = self.dropout(self.char_emb(x) + self.pos_emb(pos))
        arabert_embs = self._get_arabert_char_embeddings(original_sentences)
        ctx_h, ctx_mask = torch.zeros(B, T, self.arabert_dim, device=x.device), torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i, emb in enumerate(arabert_embs):
            l = min(emb.shape[0], T)
            if l > 0: ctx_h[i, :l], ctx_mask[i, :l] = emb[:l], True
        ctx_h = self.arabert_proj(ctx_h)
        gate_c, gate_x = torch.sigmoid(self.gate_char(char_h)), torch.sigmoid(self.gate_context(ctx_h))
        fused = (gate_c * char_h + gate_x * ctx_h) * (mask & ctx_mask).unsqueeze(-1).float()
        return self.classifier(self.encoder(fused, src_key_padding_mask=~mask))

# =========================
# 3) Inference Core
# =========================
def diacritize_chunk(model, text, stoi, max_len):
    base = strip_diacritics(canonicalize(text))
    if not base: return ""
    ids = [stoi.get(ch, 1) for ch in base[:max_len]]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    mask = (x != 0)
    with torch.no_grad():
        logits = model(x, mask, [base[:max_len]])
        pred = logits.argmax(-1)[0].tolist()
    return "".join(ch + "".join(tag_to_marks(id2tag_safe(tid))) for ch, tid in zip(base[:max_len], pred))

def diacritize_text(model, text, stoi, max_len, overlap=20):
    # Handles long text by chunking
    words = text.split()
    results, current_chunk = [], []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 > max_len:
            results.append(diacritize_chunk(model, " ".join(current_chunk), stoi, max_len))
            current_chunk, current_len = [], 0
        current_chunk.append(word)
        current_len += len(word) + 1
    
    if current_chunk:
        results.append(diacritize_chunk(model, " ".join(current_chunk), stoi, max_len))
    
    return " ".join(results)

# =========================
# 4) Main Logic
# =========================
def main():
    BEST_CKPT = "checkpoints/best.pt"
    if not os.path.exists(BEST_CKPT):
        print(f"Error: {BEST_CKPT} not found."); return

    ck = torch.load(BEST_CKPT, map_location=DEVICE)
    state, itos = ck["model"], ck["itos"]
    stoi = {c: i for i, c in enumerate(itos)}
    
    # Auto-config from checkpoint
    v_size, d_mod, p_size, n_tag = state["char_emb.weight"].shape[0], state["char_emb.weight"].shape[1], state["pos_emb.weight"].shape[0], state["classifier.weight"].shape[0]
    cfg = ck.get("cfg", {})
    n_h, lay, d_ff, drop = int(cfg.get("N_HEAD", 4)), int(cfg.get("LAYERS", 6)), int(cfg.get("DIM_FF", 1024)), float(cfg.get("DROPOUT", 0.1))

    is_arabert = any(k.startswith("gate_char") for k in state.keys())
    ModelClass = CharTransformerWithAraBERT if is_arabert else CharTransformer
    model = ModelClass(v_size, d_mod, n_h, lay, d_ff, drop, n_tag, p_size).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    max_len = p_size - 5 # Buffer for safety

    # FILE MODE
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} not found."); return
        
        print(f"Reading from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        output_lines = []
        for line in lines:
            if line.strip():
                output_lines.append(diacritize_text(model, line.strip(), stoi, max_len))
            else:
                output_lines.append("")

        with open("diacritized.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        
        print(f"✅ Success! File saved to: {os.path.join(os.getcwd(), 'diacritized.txt')}")

    # INTERACTIVE MODE
    else:
        print(f"Model: {ModelClass.__name__} | Mode: Interactive")
        while True:
            try:
                s = input("\nاكتب جملة (أو exit): ").strip()
                if s.lower() == "exit": break
                if s: print("🔹 النتيجة:", diacritize_text(model, s, stoi, max_len))
            except EOFError: break

if __name__ == "__main__":
    main()
