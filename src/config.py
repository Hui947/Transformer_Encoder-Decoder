from dataclasses import dataclass
from typing import Optional, Literal, Tuple

@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1

    attention_type: Literal["dense", "sparse"] = "dense"
    sparse_window: int = 64
    pos_encoding: Literal["absolute", "relative"] = "absolute"
    max_seq_len: int = 512
    use_tied_embeddings: bool = True

    label_ignore_index: int = -100