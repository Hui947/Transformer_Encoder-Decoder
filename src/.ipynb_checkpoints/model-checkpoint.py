import math, copy
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        mean = x.mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.a_2 * x_hat + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, qlen: int, klen: int, device=None):
        q = torch.arange(qlen, device=device)[:, None]
        k = torch.arange(klen, device=device)[None, :]
        dist = (k - q).clamp(-self.max_distance, self.max_distance) + self.max_distance  # [Tq,Tk] in [0,2*max]
        values = self.bias(dist)  # [Tq,Tk,H]
        return values.permute(2, 0, 1)  # [H,Tq,Tk]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

def attention(
    query, key, value,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
    additive_mask: Optional[torch.Tensor] = None,
    relative_bias: Optional[torch.Tensor] = None,  # [H,Tq,Tk]
    attention_type: Literal["dense", "sparse"] = "dense",
    sparse_window: int = 64,
    is_self_attention: bool = False,
):
    d_k = query.size(-1)
    B, H, Tq, _ = query.shape
    Tk = key.size(-2)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,Tq,Tk]

    if relative_bias is not None:
        scores = scores + relative_bias.unsqueeze(0)

    if is_self_attention and attention_type == "sparse":
        device = scores.device
        q = torch.arange(Tq, device=device)[:, None]
        k = torch.arange(Tk, device=device)[None, :]
        local = (k >= q - sparse_window) & (k <= q + sparse_window)
        scores = scores.masked_fill(~local.unsqueeze(0).unsqueeze(0), -1e9)

    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = scores.masked_fill(~mask, -1e9)

    if additive_mask is not None:
        scores = scores + additive_mask

    p_attn = scores.softmax(dim=-1)
    p_attn = torch.nan_to_num(p_attn, nan=0.0)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(
        self, h, d_model, dropout=0.1,
        attention_type: Literal["dense", "sparse"] = "dense",
        sparse_window: int = 64,
        use_relative_bias: bool = False,
        max_relative_distance: int = 128,
        apply_relative_to_cross: bool = False,
    ):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # Q,K,V,O
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.attention_type = attention_type
        self.sparse_window = sparse_window
        self.use_relative_bias = use_relative_bias
        self.rel_bias = RelativePositionBias(h, max_relative_distance) if use_relative_bias else None
        self.apply_relative_to_cross = apply_relative_to_cross

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None, is_cross: bool = False):
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]  # [B,H,T*,d_k]

        rel_bias = None
        if self.use_relative_bias and (not is_cross or self.apply_relative_to_cross):
            rel_bias = self.rel_bias(query.size(-2), key.size(-2), device=query.device)  # [H,Tq,Tk]

        x, self.attn = attention(
            query, key, value,
            mask=mask, dropout=self.dropout,
            relative_bias=rel_bias,
            attention_type=self.attention_type,
            sparse_window=self.sparse_window,
            is_self_attention=not is_cross,
        )

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, tgt_mask))
        x = self.sublayer[1](x, lambda x_: self.src_attn(x_, m, m, src_mask, is_cross=True))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(cfg):
    use_relative_bias = (cfg.pos_encoding == "relative")
    position = PositionalEncoding(cfg.d_model, cfg.dropout, max_len=cfg.max_seq_len) \
               if cfg.pos_encoding == "absolute" else nn.Identity()
    
    c = copy.deepcopy

    self_attn = MultiHeadedAttention(
        h=cfg.n_heads,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        attention_type=cfg.attention_type,
        sparse_window=cfg.sparse_window,
        use_relative_bias=use_relative_bias,
        max_relative_distance=min(cfg.max_seq_len, 128),
        apply_relative_to_cross=False,
    )
    cross_attn = MultiHeadedAttention(
        h=cfg.n_heads,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        attention_type="dense",
        use_relative_bias=False,
        max_relative_distance=min(cfg.max_seq_len, 128),
        apply_relative_to_cross=False,
    )

    ff = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    src_vocab = tgt_vocab = cfg.vocab_size

    model = EncoderDecoder(
        Encoder(EncoderLayer(cfg.d_model, c(self_attn), c(ff), cfg.dropout), cfg.num_encoder_layers),
        Decoder(DecoderLayer(cfg.d_model, c(self_attn), c(cross_attn), c(ff), cfg.dropout), cfg.num_decoder_layers),
        nn.Sequential(Embeddings(cfg.d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(cfg.d_model, tgt_vocab), c(position)),
        Generator(cfg.d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if cfg.use_tied_embeddings:
        tgt_embedding_weight = model.tgt_embed[0].lut.weight
        model.generator.proj.weight = tgt_embedding_weight

    return model