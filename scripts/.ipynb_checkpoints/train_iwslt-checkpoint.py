import os, math, argparse, importlib, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- 运行时注入，不改 models.py ---
models = importlib.import_module("models")
setattr(models, "math", math)
setattr(models, "log_softmax", F.log_softmax)
make_model = models.make_model

from iwslt.data import (
    TokenizerPair, Vocab, load_iwslt17, to_token_pairs,
    TranslationDataset, collate_pad
)
from engine.train_eval import evaluate, train_one_epoch
from engine.noam import build_noam

def build_vocab(train_pairs, min_freq=2, max_size=40000):
    src_all, tgt_all = [], []
    for s,t in train_pairs: src_all+=s; tgt_all+=t
    return Vocab(src_all, min_freq, max_size), Vocab(tgt_all, min_freq, max_size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iwslt_root", required=True, help="指向 texts/ 目录，上层含 de-en/ ar-en/ en-ar/ 子目录")
    ap.add_argument("--src_lang", default="de")
    ap.add_argument("--tgt_lang", default="en")
    ap.add_argument("--save_dir", default="./checkpoints")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--vocab_max", type=int, default=40000)
    ap.add_argument("--lr_factor", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=8000)
