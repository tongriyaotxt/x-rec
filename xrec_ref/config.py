"""Hyper-parameters: `paper` = X-Rec §4.1.4 production values, `demo` = CPU-runnable reproduction."""
from dataclasses import asdict, dataclass


@dataclass
class CFG:
    # ---- item / attribute space ----
    d_e: int = 128                 # item + sub-attribute embedding dim (paper: 128)
    n_attr: int = 2                # F: number of sub-attribute fields
    n_action: int = 8              # cardinality of attribute field 0
    n_duration: int = 4            # cardinality of attribute field 1
    n_items: int = 6000            # |X| in the demo (production: billion-scale)
    n_modes: int = 8               # synthetic semantic clusters (not a paper hparam)

    # ---- backbone (Qwen3-style, RoPE + QK-norm) ----
    d_m: int = 160                 # model dim (paper: Qwen3-0.6B hidden)
    n_head: int = 5
    n_layer: int = 3               # L   (paper: 28; L' = 1 late-interaction layer)
    ffn_mult: int = 2.67           # SwiGLU hidden = int(d_m * ffn_mult * 2/3) rounded
    n_anchor: int = 32             # |A| K-means anchors (paper: RQ codebook, 4096/cluster head)

    # ---- sequence ----
    seq_len: int = 48              # N (paper: 200)
    n_noise_sft: int = 8           # R noise replicas per target in SFT (§3.4.2)
    n_noise_pre: int = 4           # R noise replicas per prefix position in all-prefix pretraining
    pretrain_mask: str = "all"     # all (Fig.3a, one pass) | last (random-prefix, legacy)
    ctx_drop: float = 0.05         # CFG context dropout (§4.1.4)

    # ---- training ----
    batch: int = 64
    lr_pre: float = 5e-4
    lr_sft: float = 1e-4
    steps_pre: int = 400
    steps_sft: int = 800
    cosine: bool = True            # cosine decay after warmup (was: constant after warmup)
    pop_alpha: float = 1.0         # popularity re-weighting strength; 0 = the old uniform loss
    pop_gamma: float = 1.0         # shape of the log-frequency curve
    alpha: float = 0.1             # anchor loss weight
    beta: float = 1.0              # RFM loss weight
    wd: float = 0.01
    grad_clip: float = 1.0

    # ---- inference ----
    K: int = 20                    # triggers per request
    steps: int = 8                 # denoising steps T (paper: 30)
    anchor_temp: float = 1.4       # §4.4.2
    cfg_scale: float = 3.0         # §4.1.4
    R: int = 1                     # ANN top-R per trigger -> Recall@(K x R)

    # ---- eval ----
    n_eval: int = 1024
    n_eval_chunk: int = 256        # requests per sampling chunk (K/V replication is K x per request)
    seed: int = 0

    # ---- ablation switches ----
    use_anchor: bool = True
    use_rfm: bool = True
    use_pretrain: bool = True
    arch: str = "xrec"             # xrec | u2i | u2i_k | sid_ar
    n_interest: int = 20           # K for the ComiRec-SA style U2I-K variant
    sid_codebooks: int = 4         # RQ levels for the SID-AR control (paper: 4 tokens per item)
    sid_size: int = 256            # RQ codebook size per level

    def to_dict(self):
        return asdict(self)


PAPER = dict(
    d_e=128, d_m=1024, n_head=16, n_layer=28, seq_len=200, batch=4096,
    lr_pre=5e-4, lr_sft=1e-4, steps=30, anchor_temp=1.4, cfg_scale=3.0,
    K=20, R=1, n_anchor=4096, n_noise_sft=8, alpha=0.1, beta=1.0,
)
