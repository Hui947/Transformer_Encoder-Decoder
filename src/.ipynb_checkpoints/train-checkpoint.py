import os
import math
import time
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
import matplotlib.pyplot as plt

from model import make_model
from config import TransformerConfig
from dataloader import make_dataloaders, DataConfig

# mask
def subsequent_mask(size: int, device=None):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    return ~mask

def make_src_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    return (attn_mask > 0)[:, None, None, :]  # [B,1,1,Ts]

def make_tgt_mask(dec_attn_mask: torch.Tensor) -> torch.Tensor:
    # [B,Tt] -> [B,1,Tt,Tt] (padding ∧ causal)
    B, Tt = dec_attn_mask.shape
    device = dec_attn_mask.device
    pad = (dec_attn_mask > 0)[:, None, None, :]  # [B,1,1,Tt]
    causal = subsequent_mask(Tt, device=device)  # [1,Tt,Tt]
    return pad & causal[None, ...]

# utils
def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    f.write(f"\n===== New Run at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    f.flush()
    return f

def log_line(fh, s: str, also_print: bool = True):
    if also_print:
        print(s, flush=True)
    if fh is not None:
        fh.write(s + "\n")
        fh.flush()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_loss_curve(loss_history, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label=f'Epoch {epoch} Loss')
    plt.title(f'Training Loss - Epoch {epoch}')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'outputs/loss_curve_epoch_{epoch}.png')
    plt.close()
    
def compute_bleu(references: List[List[str]], hypotheses: List[str]) -> float:
    refs_transposed = [[rlist[0] if len(rlist) > 0 else "" for rlist in references]]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed).score
    return float(bleu)

# scheduler
class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step_num = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        scale = self.factor * (self.d_model ** -0.5) * \
                min(self._step_num ** -0.5 if self._step_num > 0 else float('inf'),
                    self._step_num * (self.warmup_steps ** -1.5) if self._step_num > 0 else 0.0)
        return [base_lr * 0 + scale for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self._step_num += 1
        super().step()

# test
@torch.no_grad()
def greedy_generate(model, tokenizer, src, src_pad_mask, max_new_tokens=128, min_new_tokens=4):
    device = src.device
    B = src.size(0)
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    # src mask
    src_mask = (src_pad_mask > 0)[:, None, None, :]  # [B,1,1,Ts]

    # init
    dec = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)

    memory = model.encode(src, src_mask)

    for t in range(max_new_tokens):
        # tgt mask = padding ∧ causal
        pad = (dec != tokenizer.pad_token_id)[:, None, None, :]   # [B,1,1,Tt]
        causal = subsequent_mask(dec.size(1), device=device)      # [1,Tt,Tt]
        tgt_mask = pad & causal[None, ...]

        out = model.decode(memory, src_mask, dec, tgt_mask)       # [B,Tt,D]
        logits = model.generator(out)                              # [B,Tt,V]
        next_token = logits[:, -1, :].argmax(-1)                  # [B]

        dec = torch.cat([dec, next_token.unsqueeze(1)], dim=1)

        alive &= (next_token != eos_id)
        if not alive.any():
            break

    return dec[:, 1:]

def decode_batch(tokenizer, ids: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def labels_to_texts(tokenizer, labels: torch.Tensor) -> List[str]:
    pad_id = tokenizer.pad_token_id
    arr = labels.clone()
    arr[arr == -100] = pad_id
    return decode_batch(tokenizer, arr)


def evaluate_bleu(model, loader: Optional[DataLoader], tokenizer, device, max_new_tokens: int, log_fh=None) -> float:
    if loader is None:
        return float("nan")

    model.eval()
    all_refs: List[List[str]] = []
    all_hyps: List[str] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            gen_ids = greedy_generate(
                model, tokenizer, input_ids, attention_mask, max_new_tokens=max_new_tokens
            )
            hyps = decode_batch(tokenizer, gen_ids)

            refs = labels_to_texts(tokenizer, batch["labels"]).copy()
            refs = [r.strip() for r in refs]
            hyps = [h.strip() for h in hyps]

            if len(all_hyps) < 3:
                for i in range(min(3, len(hyps))):
                    try:
                        pbar.write(f"[GEN] {hyps[i]}")
                        pbar.write(f"[REF] {refs[i]}")
                    except Exception:
                        print(f"[GEN] {hyps[i]}", flush=True)
                        print(f"[REF] {refs[i]}", flush=True)

            all_refs.extend([[r] for r in refs])
            all_hyps.extend(hyps)

    bleu = compute_bleu(all_refs, all_hyps)
    log_line(log_fh, f"[Eval] BLEU-4 = {bleu:.2f}")
    return bleu

# train
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_accum_steps: int,
    max_norm: float,
    log_fh=None,
    epoch: int = 0,
    loss_history = None,
):
    model.train()
    running = 0.0
    step_count = 0
    loss_history = loss_history or []

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc=f"Train E{epoch}")
    for step, batch in pbar:
        src = batch["input_ids"].to(device, non_blocking=True)
        tgt = batch["decoder_input_ids"].to(device, non_blocking=True)
        src_mask = make_src_mask(batch["attention_mask"]).to(device)
        tgt_mask = make_tgt_mask(batch["decoder_attention_mask"]).to(device)
        labels = batch["labels"].to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(src, tgt, src_mask, tgt_mask)
            logits = model.generator(out)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_history.append(loss.item())
        running += float(loss.item()) * grad_accum_steps
        step_count += 1
        pbar.set_postfix(loss=f"{running/step_count:.4f}")

    avg = running / max(step_count, 1)
    log_line(log_fh, f"[Train] Epoch {epoch} | Loss {avg:.4f}")
    plot_loss_curve(loss_history, epoch)
    return avg, loss_history


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument("--max_target_length", type=int, default=None)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--enc_layers", type=int, default=8)
    parser.add_argument("--dec_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--sparse_window", type=int, default=64)
    parser.add_argument("--pos_encoding", type=str, default="absolute", choices=["absolute", "relative"])
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--tie_embeddings", action="store_true")

    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--log_file", type=str, default="train_log.txt")
    parser.add_argument("--ckpt_name", type=str, default="last.ckpt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fh = setup_logger(out_dir / args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_line(log_fh, f"Device: {device}")
    log_line(log_fh, f"Args: {vars(args)}")

    # ---- Build dataloaders ----
    dcfg = DataConfig()
    if args.batch_size is not None:
        dcfg.batch_size = args.batch_size
    if args.max_source_length is not None:
        dcfg.max_source_length = args.max_source_length
    if args.max_target_length is not None:
        dcfg.max_target_length = args.max_target_length

    train_loader, val_loader, test_loader, tokenizer, meta = make_dataloaders(dcfg)
    log_line(log_fh, f"Tokenizer: {meta['tokenizer_name']} | V={meta['vocab_size']}")
    log_line(log_fh, f"Dataloader: train={len(train_loader.dataset)} "
                     f"val={(len(val_loader.dataset) if val_loader is not None else 0)} "
                     f"test={(len(test_loader.dataset) if test_loader is not None else 0)}")

    # ---- Build model ----
    mcfg = TransformerConfig(
        vocab_size=meta["vocab_size"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dropout=args.dropout,
        attention_type=args.attention_type,
        sparse_window=args.sparse_window,
        pos_encoding=args.pos_encoding,
        max_seq_len=args.max_seq_len,
        use_tied_embeddings=args.tie_embeddings,
    )
    model = make_model(mcfg).to(device)
    log_line(log_fh, f"Model params: {count_parameters(model)/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = NoamScheduler(
        optimizer,
        d_model=args.d_model,
        warmup_steps=4000,
        factor=args.lr,
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    ckpt_path = out_dir / args.ckpt_name
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        log_line(log_fh, f"Resumed from {ckpt_path}")

    # ---- load ----
    if args.load_ckpt is not None:
        load_path = Path(args.load_ckpt)
        assert load_path.exists(), f"Checkpoint not found: {load_path}"
        state = torch.load(load_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        log_line(log_fh, f"Loaded model weights from {load_path}")

    if args.eval_only:
        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            if test_loader is not None:
                test_bleu = evaluate_bleu(model, test_loader, tokenizer, device, args.max_new_tokens, log_fh=log_fh)
                log_line(log_fh, f"[EvalOnly] Test BLEU-4 = {test_bleu:.2f}")
        log_line(log_fh, "Eval-only finished.")
    else:
        # ---- Train ----
        best_bleu = -1.0
        loss_history = []
        for epoch in range(1, args.epochs + 1):
            _, loss_history = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device,
                grad_accum_steps=args.grad_accum, max_norm=args.max_grad_norm,
                log_fh=log_fh, epoch=epoch, loss_history=loss_history
            )
    
            to_save = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
            }
            torch.save(to_save, ckpt_path)
    
            if epoch % args.eval_every == 0:
                _ = evaluate_bleu(model, val_loader, tokenizer, device, args.max_new_tokens, log_fh=log_fh)
                test_bleu = evaluate_bleu(model, test_loader, tokenizer, device, args.max_new_tokens, log_fh=log_fh)
                if test_bleu > best_bleu:
                    best_bleu = test_bleu
                    torch.save(to_save, out_dir / "best.ckpt")
                    log_line(log_fh, f"[Checkpoint] New best BLEU {best_bleu:.2f} saved to best.ckpt")
    
        final_bleu = evaluate_bleu(model, test_loader, tokenizer, device, args.max_new_tokens, log_fh=log_fh)
        log_line(log_fh, f"[Final] Test BLEU-4 = {final_bleu:.2f}")
        log_line(log_fh, "Done.")


if __name__ == "__main__":
    main()