from dataclasses import dataclass
from typing import Dict, List, Optional
import os

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    dataset_path: str = "/root/autodl-tmp/Transformer/data/IWSLT2017/iwslt2017.py"
    dataset_config: str = "iwslt2017-en-de"
    src_lang: str = "en"
    tgt_lang: str = "de"
    tokenizer_name: str = "t5-small"
    max_source_length: int = 128
    max_target_length: int = 128
    batch_size: int = 64
    num_workers: int = min(4, os.cpu_count() or 0)
    pin_memory: bool = True
    shuffle_train: bool = True
    use_hf_tokenizer: bool = True


def _load_raw_dataset(cfg: DataConfig):
    ds = load_dataset(cfg.dataset_path, cfg.dataset_config)
    return ds["train"], ds.get("validation"), ds.get("test")


def _prepare_tokenizer(cfg: DataConfig):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _build_processors(cfg: DataConfig, tokenizer):
    src_key, tgt_key = cfg.src_lang, cfg.tgt_lang
    def process_batch(batch):
        # IWSLT: batch["translation"] 为字典列表
        src_texts = [ex[src_key] for ex in batch["translation"]]
        tgt_texts = [ex[tgt_key] for ex in batch["translation"]]

        model_inputs = tokenizer(
            src_texts,
            max_length=cfg.max_source_length,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts,
                max_length=cfg.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return process_batch


class Seq2SeqCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.label_pad = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        enc = self.tok.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "attention_mask": [f["attention_mask"] for f in features] if "attention_mask" in features[0] else None},
            padding=True,
            return_tensors="pt",
        )
        max_len = max(len(f["labels"]) for f in features)
        labels = []
        for f in features:
            x = f["labels"]
            pad = [self.label_pad] * (max_len - len(x))
            labels.append(x + pad)
        labels = torch.tensor(labels, dtype=torch.long)

        bos_id = self.tok.bos_token_id
        if bos_id is None:
            assert self.tok.eos_token_id is not None, "Need eos_token_id as BOS fallback"
            bos_id = self.tok.eos_token_id

        shifted = labels.clone()
        shifted[shifted == self.label_pad] = self.pad_id
        decoder_input_ids = torch.full_like(shifted, fill_value=self.pad_id)
        decoder_input_ids[:, 0] = bos_id
        decoder_input_ids[:, 1:] = shifted[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.pad_id).long()

        batch = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"] if "attention_mask" in enc else (enc["input_ids"] != self.pad_id).long(),
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return batch


def make_dataloaders(cfg: Optional[DataConfig] = None):
    cfg = cfg or DataConfig()
    tokenizer = _prepare_tokenizer(cfg)

    train_raw, val_raw, test_raw = _load_raw_dataset(cfg)
    process_fn = _build_processors(cfg, tokenizer)

    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_ds = train_raw.map(process_fn, batched=True, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(process_fn, batched=True, remove_columns=val_raw.column_names) if val_raw is not None else None
    test_ds = test_raw.map(process_fn, batched=True, remove_columns=test_raw.column_names) if test_raw is not None else None

    collate = Seq2SeqCollator(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    ) if val_ds is not None else None

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate,
    ) if test_ds is not None else None

    meta = {
        "tokenizer_name": cfg.tokenizer_name,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "src_lang": cfg.src_lang,
        "tgt_lang": cfg.tgt_lang,
        "max_source_length": cfg.max_source_length,
        "max_target_length": cfg.max_target_length,
    }

    return train_loader, val_loader, test_loader, tokenizer, meta