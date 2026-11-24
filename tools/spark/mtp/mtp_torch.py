# SPDX-License-Identifier: Apache-2.0
"""A differentiable plain-PyTorch Qwen3.8-Flash-Next MTP draft head (`hibana/model/flashnext/mtp.py`).

The math follows Hibana's head step for step:

    u  = fc_embedding(norm_e(embed(x_i))) broadcast over the 4 streams + fc_hidden(norm_h(h_{i-1})) per stream
    h, n = attn_hc.mix(u)                  grouped Gemma norm, gate = sigmoid(up(silu(down(n)/4))), mean of gate*n
    a  = o_proj(attn(q, k, v) * sigmoid(gate))   24 q / 2 kv heads of 256, q/k Gemma norm, NeoX RoPE on 64 dims
    u' = u + a * 2 sigmoid(inject(n)/4)    per stream
    h', n' = mlp_hc.mix(u')
    d  = u' + moe(h') * 2 sigmoid(inject'(n')/4)   top-10 of 512 routed experts + sigmoid-gated shared expert
    logits = lm_head(mixer.mix(d))

Attention is dense and causal over the draft KV of the window (the indexer is never used by the MTP layer). Chained
drafting feeds d back as the next hidden; the KV row of a chain's own later steps is visible only to that chain, as in
`proposers/mtp.py` (step j of the chain that starts at head row r attends the target-conditioned KV of rows < r plus
the chain's own steps 1..j).

Modes:
- `float`: training; trainable weights are FP32 master parameters, matmuls run in `compute_dtype` (BF16 on the GPU),
  norms/softmax/sigmoids in FP32, activations rounded to `compute_dtype` where Hibana rounds to BF16.
- `fp8`: forward-only emulation of Hibana's serving quantization: every trunk projection (fc, qkv, o, shared expert,
  hyper-connection down/up, lm_head) E4M3 with one weight scale per matrix and a dynamic activation scale per call
  (per row for a hyper-connection's down projection from 1024 rows: Hibana's wide prefill path); the inject matrices
  and the router stay BF16; routed experts E4M3 with one scale per expert and BF16 activations.

Window convention (capture shards): window row i has token x_i and the target's hyper stream h_i after it. Head row r
is window row r + 1: input (x_{r+1}, h_r) at position p_{r+1}; its step j drafts token x_{r+1+j}, which the target
predicted at window row r + j (`topk_ids[r + j]`).
"""

from __future__ import annotations

import glob
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- geometry (hibana/model/flashnext/config.py) ----
HIDDEN = 2560
HC = 4
WIDTH = HC * HIDDEN
LOWRANK = 320
N_HEADS = 24
N_KV = 2
HEAD_DIM = 256
Q_SIZE = N_HEADS * HEAD_DIM
KV_SIZE = N_KV * HEAD_DIM
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
EPS = 1e-6
N_EXPERTS = 512
TOPK = 10
INTER = 640
VOCAB = 248_320
FP8_MAX = 448.0
WIDE_ROWS = 1024  # hibana/kernels/hyper_prefill.ROWS
MODEL_DIR = "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
EMBED_NAME = "model.language_model.embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"

L0 = "mtp.layers.0"
_HC_SITES = {"attn": f"{L0}.attn_hyper_connection", "mlp": f"{L0}.mlp_hyper_connection", "mixer": "mtp.hyper_connection_mixer"}


def _expected() -> dict[str, tuple[int, ...]]:
    e = {
        "mtp.fc_embedding.weight": (HIDDEN, HIDDEN),
        "mtp.fc_hidden.weight": (HIDDEN, HIDDEN),
        "mtp.pre_fc_norm_embedding.weight": (HIDDEN,),
        "mtp.pre_fc_norm_hidden.weight": (WIDTH,),
    }
    for site, p in _HC_SITES.items():
        e[f"{p}.hc_norm.weight"] = (WIDTH,)
        e[f"{p}.input_mix_weight_down.weight"] = (LOWRANK, WIDTH)
        e[f"{p}.input_mix_weight_up.weight"] = (WIDTH, LOWRANK)
        if site != "mixer":
            e[f"{p}.block_inject_weight.weight"] = (HC, WIDTH)
    a = f"{L0}.self_attn"
    e.update({
        f"{a}.q_proj.weight": (2 * Q_SIZE, HIDDEN), f"{a}.k_proj.weight": (KV_SIZE, HIDDEN), f"{a}.v_proj.weight": (KV_SIZE, HIDDEN),
        f"{a}.o_proj.weight": (HIDDEN, Q_SIZE), f"{a}.q_norm.weight": (HEAD_DIM,), f"{a}.k_norm.weight": (HEAD_DIM,),
        f"{a}.indexer.index_qk_proj.weight": (640, HIDDEN), f"{a}.indexer.k_layernorm.weight": (128,),
        f"{a}.indexer.q_layernorm.weight": (128,),
    })
    m = f"{L0}.mlp"
    e.update({
        f"{m}.gate.weight": (N_EXPERTS, HIDDEN), f"{m}.experts.gate_up_proj": (N_EXPERTS, 2 * INTER, HIDDEN),
        f"{m}.experts.down_proj": (N_EXPERTS, HIDDEN, INTER), f"{m}.shared_expert.gate_proj.weight": (INTER, HIDDEN),
        f"{m}.shared_expert.up_proj.weight": (INTER, HIDDEN), f"{m}.shared_expert.down_proj.weight": (HIDDEN, INTER),
        f"{m}.shared_expert_gate.weight": (1, HIDDEN),
    })
    return e


EXPECTED: dict[str, tuple[int, ...]] = _expected()
assert len(EXPECTED) == 31
INDEXER = tuple(n for n in EXPECTED if ".indexer." in n)  # loaded, never used (the MTP layer's attention is dense)
ROUTER = f"{L0}.mlp.gate.weight"
EXPERTS_GU = f"{L0}.mlp.experts.gate_up_proj"
EXPERTS_DOWN = f"{L0}.mlp.experts.down_proj"
FROZEN = (ROUTER, EXPERTS_GU, EXPERTS_DOWN) + INDEXER
TRAINABLE = tuple(n for n in EXPECTED if n not in FROZEN)
assert len(TRAINABLE) == 25

# logical projections -> checkpoint matrices fused along the output dimension (as Hibana's TrunkLinear.load)
_A = f"{L0}.self_attn"
_S = f"{L0}.mlp.shared_expert"
LINEARS: dict[str, tuple[str, ...]] = {
    "fc_e": ("mtp.fc_embedding.weight",),
    "fc_h": ("mtp.fc_hidden.weight",),
    "qkv": (f"{_A}.q_proj.weight", f"{_A}.k_proj.weight", f"{_A}.v_proj.weight"),
    "o": (f"{_A}.o_proj.weight",),
    "shared_gu": (f"{_S}.gate_proj.weight", f"{_S}.up_proj.weight"),
    "shared_down": (f"{_S}.down_proj.weight",),
}
for _site, _p in _HC_SITES.items():
    LINEARS[f"{_site}.down"] = (f"{_p}.input_mix_weight_down.weight",)
    LINEARS[f"{_site}.up"] = (f"{_p}.input_mix_weight_up.weight",)


def key(name: str) -> str:
    """A checkpoint name as a ParameterDict / buffer key."""
    return name.replace(".", "__")


# ---- snapshots and safetensors ----
def resolve_snapshot(root: str | os.PathLike) -> Path:
    """A snapshot directory, or an HF cache root holding the pinned revision (directly or under hub/)."""
    p = Path(root)
    cands = [p, p / "hub" / MODEL_DIR / "snapshots" / REVISION, p / MODEL_DIR / "snapshots" / REVISION, p / "snapshots" / REVISION]
    for c in cands:
        if (c / "model.safetensors.index.json").is_file():
            return c
    raise FileNotFoundError(f"no Flash-Next snapshot (model.safetensors.index.json) under {p}")


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    """(safetensors header, absolute byte offset of the data section)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


class Snapshot:
    """Name -> file of a sharded safetensors snapshot, with headers for strict shape checks and byte-exact copies."""

    def __init__(self, root):
        self.dir = resolve_snapshot(root)
        self.map: dict[str, str] = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}

    def header(self, name: str) -> tuple[dict, int, Path]:
        f = self.map[name]
        if f not in self._headers:
            self._headers[f] = read_header(self.dir / f)
        h, base = self._headers[f]
        return h[name], base, self.dir / f

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.header(name)[0]["shape"])

    def get(self, name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        from safetensors import safe_open
        with safe_open(str(self.dir / self.map[name]), framework="pt", device="cpu") as fh:
            if rows is None:
                return fh.get_tensor(name)
            return fh.get_slice(name)[rows[0]:rows[1]]

    def mtp_names(self) -> list[str]:
        return sorted(n for n in self.map if n.startswith("mtp."))


def check_names(snap: Snapshot, overlay: dict[str, torch.Tensor] | None = None) -> dict:
    """Strict: the snapshot's mtp.* tensors are exactly the 31 expected names with the expected shapes and BF16; an
    overlay holds a subset of them with the same shapes. Raises on any mismatch."""
    names = set(snap.mtp_names())
    missing, unexpected = sorted(set(EXPECTED) - names), sorted(names - set(EXPECTED))
    if missing or unexpected:
        raise ValueError(f"MTP tensors: missing {missing}, unexpected {unexpected}")
    bad = []
    for n, shape in EXPECTED.items():
        info = snap.header(n)[0]
        if tuple(info["shape"]) != shape or info["dtype"] != "BF16":
            bad.append((n, info["dtype"], info["shape"], shape))
    if overlay is not None:
        extra = sorted(set(overlay) - set(EXPECTED))
        if extra:
            raise ValueError(f"overlay holds tensors that are not MTP head tensors: {extra[:5]}")
        for n, t in overlay.items():
            if tuple(t.shape) != EXPECTED[n] or t.dtype != torch.bfloat16:
                bad.append((n, str(t.dtype), list(t.shape), EXPECTED[n]))
    if bad:
        raise ValueError(f"MTP tensor shape/dtype mismatches: {bad}")
    return {"tensors": len(EXPECTED), "trainable": len(TRAINABLE), "frozen": list(FROZEN), "unused": list(INDEXER),
            "overlay": sorted(overlay) if overlay else []}


def load_overlay(path: str | os.PathLike | None) -> dict[str, torch.Tensor] | None:
    """Trained MTP tensors (mtp_train.py's output) to use instead of the snapshot's; frozen tensors in it must match."""
    if not path:
        return None
    from safetensors.torch import load_file
    return load_file(str(path))


_ST_DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16", torch.int32: "I32", torch.int64: "I64",
              torch.uint8: "U8"}


def write_safetensors(path: str | os.PathLike, entries: dict, metadata: dict[str, str] | None = None) -> None:
    """Write a safetensors file whose entries are tensors or (source file, absolute offset, nbytes, dtype, shape)
    tuples copied byte for byte (streamed: frozen 5 GB expert stacks never sit in host memory)."""
    header, offset, plan = {}, 0, []
    for name in sorted(entries):
        e = entries[name]
        if isinstance(e, torch.Tensor):
            t = e.detach().contiguous().cpu()
            n = t.numel() * t.element_size()
            header[name] = {"dtype": _ST_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
            plan.append(("tensor", t))
        else:
            src, start, n, dtype, shape = e
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
            plan.append(("copy", (src, start, n)))
        offset += n
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for kind, item in plan:
            if kind == "tensor":
                out.write(item.view(torch.uint8).numpy().tobytes() if item.dtype == torch.bfloat16 else item.numpy().tobytes())
            else:
                src, start, n = item
                with open(src, "rb") as fh:
                    fh.seek(start)
                    left = n
                    while left:
                        chunk = fh.read(min(left, 64 << 20))
                        if not chunk:
                            raise IOError(f"short read from {src}")
                        out.write(chunk)
                        left -= len(chunk)
    os.replace(tmp, path)


def raw_entry(snap: Snapshot, name: str):
    info, base, f = snap.header(name)
    a, b = info["data_offsets"]
    return (f, base + a, b - a, info["dtype"], info["shape"])


# ---- numerics shared by both modes ----
def gemma_rms(x: torch.Tensor, w: torch.Tensor, eps: float = EPS, out_dtype=None) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    return y.to(out_dtype or x.dtype)


def grouped_rms(x: torch.Tensor, w: torch.Tensor, group: int = HIDDEN, eps: float = EPS, out_dtype=None) -> torch.Tensor:
    xf = x.float()
    g = xf.reshape(*xf.shape[:-1], -1, group)
    y = (g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2) * (1.0 + w.float())
    return y.to(out_dtype or x.dtype)


def rope_cos_sin(positions: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 cos/sin [T, ROTARY_DIM/2] as hibana/model/rope.PartialRotary's table."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=device) / ROTARY_DIM))
    ang = positions.to(device=device, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def head_norm_rope(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dtype) -> torch.Tensor:
    """x [T, heads, D]: per-head Gemma norm rounded to `dtype`, then NeoX RoPE on the first ROTARY_DIM dims in FP32
    (kernels/qk_norm_rope.py)."""
    y = gemma_rms(x, w, out_dtype=dtype)
    half = ROTARY_DIM // 2
    x1, x2 = y[..., :half].float(), y[..., half:ROTARY_DIM].float()
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([(x1 * c - x2 * s).to(dtype), (x2 * c + x1 * s).to(dtype), y[..., ROTARY_DIM:]], dim=-1)


def silu_mul(gu: torch.Tensor, dtype) -> torch.Tensor:
    i = gu.shape[-1] // 2
    g, u = gu[..., :i].float(), gu[..., i:].float()
    return (g * torch.sigmoid(g) * u).to(dtype)


def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One E4M3 scale per matrix (TrunkLinear.from_bf16 / FlashNextMtpHead._fp8_experts)."""
    wf = w.float()
    s = (wf.abs().amax() / FP8_MAX).clamp_min(1e-12)
    return (wf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s.reshape(1, 1)


def quant_act(x: torch.Tensor, rowwise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic E4M3 activations: scale = max(amax * (1/448), 1e-12) per call (or per row), q = clamp(x / scale)."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True) if rowwise else xf.abs().amax().reshape(1, 1)
    s = torch.clamp(amax * (1.0 / FP8_MAX), min=1e-12)
    return (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def fp8_matmul(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """BF16 (xq * xs) @ (wq * ws)^T with FP32 accumulation: torch._scaled_mm on CUDA, an FP32 matmul elsewhere."""
    m, k = xq.shape
    n = wq.shape[0]
    if xq.is_cuda and k % 16 == 0 and n % 16 == 0:
        try:
            sb = ws if xs.numel() == 1 else ws.reshape(1, 1).expand(1, n).contiguous()
            return torch._scaled_mm(xq, wq.t(), scale_a=xs.float().contiguous(), scale_b=sb.float(), out_dtype=torch.bfloat16)
        except (RuntimeError, NotImplementedError):
            pass
    xf = xq.float()
    outs = []
    for a in range(0, n, chunk):
        outs.append(((xf @ wq[a:a + chunk].float().t()) * xs * ws).to(torch.bfloat16))
    return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


# ---- the head ----
@dataclass
class StepOut:
    d: torch.Tensor  # [B, WIDTH] the layer's hyper stream
    kv: tuple[torch.Tensor, torch.Tensor]  # this step's K, V rows [B, N_KV, HEAD_DIM]
    taps: dict = field(default_factory=dict)  # fuse / attn / moe when requested


class MtpHeadTorch(nn.Module):
    """See the module docstring. Construct with `from_snapshot`."""

    def __init__(self, tensors: dict[str, torch.Tensor], embed: torch.Tensor, lm_head: torch.Tensor, *, mode: str = "float",
                 compute_dtype: torch.dtype = torch.bfloat16, master_dtype: torch.dtype = torch.float32, act_scale: str = "tensor",
                 expert_range: tuple[int, int] | None = None, vocab_ids: torch.Tensor | None = None, expert_storage: str = "bf16",
                 load_report: dict | None = None, attn_chunk: int = 256, grad_ckpt: bool = False):
        super().__init__()
        missing = sorted(set(EXPECTED) - set(tensors) - set(INDEXER))
        unexpected = sorted(set(tensors) - set(EXPECTED))
        if missing or unexpected:
            raise ValueError(f"MtpHeadTorch: missing {missing}, unexpected {unexpected}")
        self.mode, self.compute_dtype, self.act_scale = mode, compute_dtype, act_scale
        self.expert_range = expert_range or (0, N_EXPERTS)
        self.n_experts = self.expert_range[1] - self.expert_range[0]
        if self.n_experts < TOPK:
            raise ValueError("at least TOPK experts")
        self.attn_chunk, self.grad_ckpt = attn_chunk, grad_ckpt
        self.load_report = load_report or {}
        self.w = nn.ParameterDict({key(n): nn.Parameter(tensors[n].to(master_dtype).clone(), requires_grad=(mode == "float"))
                                   for n in TRAINABLE})
        self.register_buffer("router", tensors[ROUTER].to(torch.bfloat16).contiguous(), persistent=False)
        self.expert_storage = expert_storage
        if expert_storage == "fp8":
            q, s = self._quant_experts(tensors.pop(EXPERTS_GU))
            self.register_buffer("gu_q", q, persistent=False)
            self.register_buffer("gu_s", s, persistent=False)
            q, s = self._quant_experts(tensors.pop(EXPERTS_DOWN))
            self.register_buffer("down_q", q, persistent=False)
            self.register_buffer("down_s", s, persistent=False)
            self.experts_gu = self.experts_down = None
        else:
            self.register_buffer("experts_gu", tensors[EXPERTS_GU].to(torch.bfloat16), persistent=False)
            self.register_buffer("experts_down", tensors[EXPERTS_DOWN].to(torch.bfloat16), persistent=False)
            self.gu_q = self.gu_s = self.down_q = self.down_s = None
        for n in INDEXER:  # loaded and unused: the MTP layer's attention is dense (hibana/model/mtp_layer.py)
            if n in tensors:
                self.register_buffer(key(n), tensors[n].to(torch.bfloat16), persistent=False)
        self.register_buffer("embed", embed, persistent=False)
        self.register_buffer("lm_head", lm_head, persistent=False)
        self.register_buffer("vocab_ids", vocab_ids if vocab_ids is not None else torch.empty(0, dtype=torch.long), persistent=False)
        self._fp8: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---- construction ----
    @classmethod
    def from_snapshot(cls, root, *, device="cpu", mode="float", compute_dtype=torch.bfloat16, overlay=None,
                      expert_range: tuple[int, int] | None = None, vocab_rows: int | None = None, embed=None, lm_head=None,
                      **kw) -> "MtpHeadTorch":
        """Strict load of the 31 mtp.* tensors (overlay: a trained file's tensors replace the snapshot's), plus the
        target's embedding and lm_head (or the given ones). `expert_range` / `vocab_rows` load a slice (CPU tests)."""
        snap = Snapshot(root)
        ov = load_overlay(overlay) if isinstance(overlay, (str, os.PathLike)) else overlay
        report = check_names(snap, ov)
        tensors = {}
        for n in EXPECTED:
            rows = expert_range if n in (ROUTER, EXPERTS_GU, EXPERTS_DOWN) and expert_range else None
            if ov is not None and n in ov:
                t = ov[n] if rows is None else ov[n][rows[0]:rows[1]]
                report.setdefault("from_overlay", []).append(n)
            else:
                t = snap.get(n, rows)
            tensors[n] = t
        if embed is None:
            embed = snap.get(EMBED_NAME, (0, vocab_rows) if vocab_rows else None)
        if lm_head is None:
            lm_head = snap.get(LM_HEAD_NAME, (0, vocab_rows) if vocab_rows else None)
        report["expert_range"] = list(expert_range) if expert_range else [0, N_EXPERTS]
        report["vocab_rows"] = int(lm_head.shape[0])
        m = cls(tensors, embed, lm_head, mode=mode, compute_dtype=compute_dtype, expert_range=expert_range, load_report=report, **kw)
        return m.to(device)

    @staticmethod
    def _quant_experts(stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty(stack.shape, dtype=torch.float8_e4m3fn, device=stack.device)
        scale = torch.empty(stack.shape[0], dtype=torch.float32, device=stack.device)
        for i in range(stack.shape[0]):
            q, s = quant_weight(stack[i])
            out[i].copy_(q)
            scale[i] = s.reshape(())
        return out, scale

    # ---- weights ----
    def param(self, name: str) -> torch.Tensor:
        return self.w[key(name)]

    def trainable_parameters(self):
        return [self.w[key(n)] for n in TRAINABLE]

    def refresh_fp8(self) -> None:
        """Drop the FP8 copies of the trainable matrices (call after an optimizer step before an fp8-mode forward)."""
        self._fp8 = {k: v for k, v in self._fp8.items() if k.startswith("experts")}

    def _weight(self, lin: str) -> torch.Tensor:
        if lin == "lm_head":
            return self.lm_head
        parts = [self.w[key(n)] for n in LINEARS[lin]]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _fp8_weight(self, lin: str):
        if lin not in self._fp8:
            with torch.no_grad():
                self._fp8[lin] = quant_weight(self._weight(lin).to(torch.bfloat16))  # Hibana quantizes the BF16 matrix
        return self._fp8[lin]

    def linear(self, x: torch.Tensor, lin: str, rowwise: bool | None = None, scale: float = 1.0, quant_input: bool = True) -> torch.Tensor:
        """y = x @ W^T (times `scale`, a power of two Hibana folds into the matrix). fp8 mode: `quant_input` False
        multiplies BF16 activations with the E4M3 matrix (the wide hyper-connection up projection)."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if self.mode == "fp8":
            wq, ws = self._fp8_weight(lin)
            if not quant_input:
                y = ((flat.to(torch.bfloat16).float() @ wq.float().t()) * (ws * scale)).to(torch.bfloat16)
                return y.reshape(*shape[:-1], y.shape[-1])
            xq, xs = quant_act(flat, self.act_scale == "row" if rowwise is None else rowwise)
            y = fp8_matmul(xq, xs, wq, ws * scale if scale != 1.0 else ws)
        else:
            w = self._weight(lin).to(self.compute_dtype)
            y = F.linear(flat.to(self.compute_dtype), w * scale if scale != 1.0 else w)
        return y.reshape(*shape[:-1], y.shape[-1])

    @property
    def act_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.mode == "fp8" else self.compute_dtype

    # ---- pieces ----
    def fuse(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        dt = self.act_dtype
        if self.vocab_ids.numel() == 0 and tokens.numel() and int(tokens.max()) >= self.embed.shape[0]:
            raise ValueError("token id outside the loaded embedding rows")
        e = gemma_rms(F.embedding(tokens, self.embed), self.param("mtp.pre_fc_norm_embedding.weight"), out_dtype=dt)
        e = self.linear(e, "fc_e")
        hn = gemma_rms(hidden, self.param("mtp.pre_fc_norm_hidden.weight"), out_dtype=dt)  # over the whole hc*H vector
        h = self.linear(hn.reshape(-1, HC, HIDDEN), "fc_h")
        return (h.float() + e.float().unsqueeze(-2)).to(dt).reshape(-1, WIDTH)

    def mix(self, hyper: torch.Tensor, site: str) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (block input [T, H], the normalized streams [T, hc*H])."""
        dt = self.act_dtype
        p = _HC_SITES[site]
        normed = grouped_rms(hyper, self.param(f"{p}.hc_norm.weight"), out_dtype=dt)
        wide = self.mode == "fp8" and self.act_scale == "tensor" and normed.shape[0] >= WIDE_ROWS
        z = self.linear(normed, f"{site}.down", rowwise=True if wide else None, scale=1.0 / HC)
        g = self.linear(F.silu(z.float()).to(dt), f"{site}.up", quant_input=not wide)
        mixed = (torch.sigmoid(g.float()) * normed.float()).reshape(-1, HC, HIDDEN).mean(-2)
        return mixed.to(dt), normed

    def combine(self, block: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, site: str) -> torch.Tensor:
        dt = self.act_dtype
        wi = self.param(f"{_HC_SITES[site]}.block_inject_weight.weight")
        cd = torch.bfloat16 if self.mode == "fp8" else self.compute_dtype
        inj = F.linear(normed.to(cd), (wi.to(torch.bfloat16) if self.mode == "fp8" else wi.to(cd)) * (1.0 / HC))  # BF16 in Hibana
        g = 2.0 * torch.sigmoid(inj.float())
        out = hyper.float().reshape(-1, HC, HIDDEN) + block.float().unsqueeze(-2) * g.unsqueeze(-1)
        return out.reshape(-1, WIDTH).to(dt)

    def qkv(self, h: torch.Tensor, positions: torch.Tensor):
        dt = self.act_dtype
        t = h.shape[0]
        qkv = self.linear(h, "qkv")
        qg = qkv[:, :2 * Q_SIZE].reshape(t, N_HEADS, 2 * HEAD_DIM)
        q_raw, gate = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]
        k_raw = qkv[:, 2 * Q_SIZE:2 * Q_SIZE + KV_SIZE].reshape(t, N_KV, HEAD_DIM)
        v = qkv[:, 2 * Q_SIZE + KV_SIZE:].reshape(t, N_KV, HEAD_DIM).to(dt)
        cos, sin = rope_cos_sin(positions, h.device)
        q = head_norm_rope(q_raw, self.param(f"{_A}.q_norm.weight"), cos, sin, dt)
        k = head_norm_rope(k_raw, self.param(f"{_A}.k_norm.weight"), cos, sin, dt)
        return q, k, v, gate

    def attend(self, q: torch.Tensor, main: tuple[torch.Tensor, torch.Tensor], ends: torch.Tensor,
               own: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """q [B, Hq, D]; row b sees main K/V rows [0, ends[b]) and its own chain rows own[s][b] (every s)."""
        b_total = q.shape[0]
        ok = torch.stack([k for k, _ in own])  # [J, B, Hkv, D]
        ov = torch.stack([v for _, v in own])
        outs = []
        for a in range(0, b_total, self.attn_chunk):
            z = min(b_total, a + self.attn_chunk)
            args = (q[a:z], main[0], main[1], ends[a:z], ok[:, a:z], ov[:, a:z])
            if torch.is_grad_enabled() and q.requires_grad:
                from torch.utils.checkpoint import checkpoint
                outs.append(checkpoint(_attend_rows, *args, use_reentrant=False))
            else:
                outs.append(_attend_rows(*args))
        return torch.cat(outs)

    def out_proj(self, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = (attn.float() * torch.sigmoid(gate.float())).to(self.act_dtype).reshape(attn.shape[0], Q_SIZE)
        return self.linear(gated, "o")

    def _expert(self, x: torch.Tensor, which: str, e: int) -> torch.Tensor:
        dt = self.act_dtype
        if self.mode == "fp8" or self.expert_storage == "fp8":
            if self.expert_storage == "fp8":
                q, s = (self.gu_q, self.gu_s) if which == "gu" else (self.down_q, self.down_s)
                wq, ws = q[e], s[e]
            else:
                ck = f"experts.{which}"
                if ck not in self._fp8:
                    self._fp8[ck] = self._quant_experts(self.experts_gu if which == "gu" else self.experts_down)
                wq, ws = self._fp8[ck][0][e], self._fp8[ck][1][e]
            if self.mode == "fp8":  # FP32 accumulation of E4M3 weights times BF16 activations (expert_gemv_fused)
                return ((x.float() @ wq.float().t()) * ws).to(torch.bfloat16)
            return F.linear(x.to(dt), (wq.float() * ws).to(dt))
        w = (self.experts_gu if which == "gu" else self.experts_down)[e]
        return F.linear(x.to(dt), w.to(dt))

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.act_dtype
        t = x.shape[0]
        logits = F.linear(x.to(torch.bfloat16) if self.mode == "fp8" else x, self.router.to(torch.bfloat16 if self.mode == "fp8" else x.dtype))
        p = torch.softmax(logits.float(), dim=-1)
        w, ids = p.topk(TOPK, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        flat = ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.n_experts).tolist()
        xs = x[order // TOPK]
        parts, start = [], 0
        for e, n in enumerate(counts):
            if n:
                seg = xs[start:start + n]
                parts.append(self._expert(silu_mul(self._expert(seg, "gu", e), dt), "down", e))
                start += n
        y_sorted = torch.cat(parts)
        y = torch.empty_like(y_sorted)
        y = y.index_copy(0, order, y_sorted)
        routed = (y.reshape(t, TOPK, HIDDEN).float() * w.unsqueeze(-1)).sum(1).to(dt)
        shared = self.linear(silu_mul(self.linear(x, "shared_gu"), dt), "shared_down")
        gd = torch.float64 if self.mode == "float" and self.compute_dtype == torch.float64 else torch.float32
        gate = torch.sigmoid(x.to(gd) @ self.param(f"{L0}.mlp.shared_expert_gate.weight").to(gd).reshape(-1, 1))
        return (routed.float() + gate * shared.float()).to(dt)

    # ---- one draft step ----
    def step(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor,
             main: tuple[torch.Tensor, torch.Tensor] | None, ends: torch.Tensor,
             own: list[tuple[torch.Tensor, torch.Tensor]], taps: bool = False) -> StepOut:
        """One MTP layer pass for B chain rows. `main` None: this step's own K/V rows are the main KV (step 1 over a
        whole window, causal). `own`: the chain's earlier steps' K/V rows ([B, Hkv, D] each)."""
        u = self.fuse(tokens, hidden)
        h, n = self.mix(u, "attn")
        q, k, v, gate = self.qkv(h, positions)
        kv = (k, v)
        a = self.out_proj(self.attend(q, kv if main is None else main, ends, own + [kv]), gate)
        u2 = self.combine(a, u, n, "attn")
        h2, n2 = self.mix(u2, "mlp")
        d = self.combine(self.moe(h2), u2, n2, "mlp")
        return StepOut(d, kv, {"fuse": u, "attn": u2, "moe": d} if taps else {})

    def prefill_kv(self, tokens, hidden, positions) -> tuple[torch.Tensor, torch.Tensor]:
        """Target-conditioned draft KV of every row (`MtpLayer.write_kv`)."""
        h, _ = self.mix(self.fuse(tokens, hidden), "attn")
        _, k, v, _ = self.qkv(h, positions)
        return k, v

    # ---- vocabulary ----
    def head_input(self, d: torch.Tensor) -> torch.Tensor:
        return self.mix(d, "mixer")[0]

    def logits(self, d: torch.Tensor) -> torch.Tensor:
        x = self.head_input(d)
        if self.mode == "fp8":
            return self.linear(x, "lm_head")
        return F.linear(x.to(self.compute_dtype), self.lm_head.to(self.compute_dtype))

    def global_ids(self, local: torch.Tensor) -> torch.Tensor:
        return local if self.vocab_ids.numel() == 0 else self.vocab_ids[local]

    @torch.no_grad()
    def draft_tokens(self, d: torch.Tensor, chunk: int = 512, draft_head=None) -> torch.Tensor:
        """The argmax over the full (loaded) vocabulary, or `draft_head.argmax(head_input)` (Hibana's ShortlistHead)."""
        out = []
        for a in range(0, d.shape[0], chunk):
            part = d[a:a + chunk]
            if draft_head is not None:
                out.append(draft_head.argmax(self.head_input(part).to(torch.bfloat16).contiguous()).long())
            else:
                out.append(self.global_ids(self.logits(part).float().argmax(-1)))
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=d.device)

    # ---- chains ----
    def chain_all(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, depth: int,
                  feed: str = "teacher", taps: bool = False, draft: bool = True, draft_head=None) -> dict:
        """Every head row starts a chain: step j (1-based) at row r has position p_r + j - 1, token tokens[r + j - 1]
        (`teacher`; zero past the window) or the chain's own previous draft (`argmax`), hidden d_{j-1}[r]. Returns
        {"d": [d_j], "draft": [tokens_j], "taps": step-1 taps}."""
        m = tokens.shape[0]
        ends = torch.arange(m, device=tokens.device)
        main, own, ds, drafts, tap = None, [], [], [], {}
        tok, hid = tokens, hidden
        for j in range(depth):
            if self.grad_ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                d, k, v = checkpoint(_step_d, self, tok, hid, positions + j, main, ends, own, use_reentrant=False)
                out = StepOut(d, (k, v))
            else:
                out = self.step(tok, hid, positions + j, main, ends, own, taps=taps and j == 0)
            if main is None:
                main = out.kv
            own = own + [out.kv]
            ds.append(out.d)
            if j == 0 and taps:
                tap = out.taps
            dr = self.draft_tokens(out.d, draft_head=draft_head) if (draft or feed == "argmax") else None
            drafts.append(dr)
            if feed == "teacher":
                tok = torch.cat([tokens[j + 1:], tokens.new_zeros(min(j + 1, m))])[:m]
            else:
                tok = dr if self.vocab_ids.numel() == 0 else _local_ids(self.vocab_ids, dr)
            hid = out.d
        return {"d": ds, "draft": drafts, "taps": tap}

    def chain_subset(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, starts: torch.Tensor,
                     depth: int, main: tuple[torch.Tensor, torch.Tensor], feed: str = "teacher", taps: bool = False,
                     draft_head=None) -> dict:
        """Chains starting at head rows `starts` only, over a main KV from `prefill_kv` (Hibana's proposer layout: step
        1 recomputes the start row's KV in the chain's own call)."""
        m = tokens.shape[0]
        own, ds, drafts, tap = [], [], [], {}
        tok, hid = tokens[starts], hidden[starts]
        for j in range(depth):
            out = self.step(tok, hid, positions[starts] + j, main, starts, own, taps=taps and j == 0)
            own = own + [out.kv]
            ds.append(out.d)
            if j == 0 and taps:
                tap = out.taps
            dr = self.draft_tokens(out.d, draft_head=draft_head)
            drafts.append(dr)
            if feed == "teacher":
                idx = starts + j + 1
                tok = torch.where(idx < m, tokens[idx.clamp(max=m - 1)], torch.zeros_like(idx))
            else:
                tok = dr if self.vocab_ids.numel() == 0 else _local_ids(self.vocab_ids, dr)
            hid = out.d
        return {"d": ds, "draft": drafts, "taps": tap}


def _local_ids(vocab_ids, global_ids):
    return torch.searchsorted(vocab_ids, global_ids)


def _step_d(head, tok, hid, pos, main, ends, own):
    out = head.step(tok, hid, pos, main, ends, own)
    return out.d, out.kv[0], out.kv[1]


def _attend_rows(q, km, vm, ends, ok, ov):
    """Dense attention of query rows q [b, Hq, D] over main rows [0, ends[i]) and their own rows ok[:, i]."""
    b = q.shape[0]
    j = ok.shape[0]
    n_main = int(ends.max().item()) if b else 0
    keys = torch.cat([km[:n_main], ok.reshape(j * b, N_KV, HEAD_DIM)])
    vals = torch.cat([vm[:n_main], ov.reshape(j * b, N_KV, HEAD_DIM)])
    cols = torch.arange(n_main, device=q.device)
    mask = torch.cat([cols[None, :] < ends[:, None], torch.eye(b, dtype=torch.bool, device=q.device).repeat(1, j)], dim=1)
    rep = N_HEADS // N_KV  # q head h reads kv head h // 12
    k = keys.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
    v = vals.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
    o = F.scaled_dot_product_attention(q.permute(1, 0, 2).unsqueeze(0), k, v, attn_mask=mask[None, None], scale=HEAD_DIM ** -0.5)
    return o[0].permute(1, 0, 2)


# ---- capture shards ----
SHARD_TENSORS = ("tokens", "positions", "h", "topk_ids", "topk_logits", "assist", "seq_starts")


@dataclass
class Window:
    tokens: torch.Tensor  # int64 [N]
    positions: torch.Tensor  # int64 [N]
    h: torch.Tensor  # bf16 [N, WIDTH]
    topk_ids: torch.Tensor  # int64 [N, 8]
    topk_logits: torch.Tensor  # fp32 [N, 8]
    assist: torch.Tensor  # bool [N]
    context_rows: int
    seq_id: str
    source: str
    shard: str
    index: int

    def head_inputs(self, device):
        """Head rows r = window rows r + 1: (tokens x_{r+1}, hidden h_r, positions p_{r+1})."""
        return (self.tokens[1:].to(device), self.h[:-1].to(device), self.positions[1:].to(device))

    def targets(self, depth: int, device=None):
        """Per head row r and step j (0-based): the target's top-8 at window row r + 1 + j, and validity: the row
        exists, its next token is assistant-produced, and the chain starts at or after context_rows."""
        n = self.tokens.shape[0]
        m = n - 1
        r = torch.arange(m)
        rows = r[:, None] + 1 + torch.arange(depth)[None, :]  # window row whose top-k the step is checked against
        inside = rows <= n - 1
        safe = rows.clamp(max=n - 1)
        valid = inside & self.assist[safe] & ((r + 1) >= self.context_rows)[:, None]
        # a step only counts while its chain's earlier steps exist too (cumulative)
        valid = torch.cumprod(valid.to(torch.int64), dim=1).bool()
        out = (self.topk_ids[safe], self.topk_logits[safe], valid)
        return tuple(t.to(device) for t in out) if device is not None else out


class CaptureShard:
    def __init__(self, path: str | os.PathLike):
        from safetensors import safe_open
        self.path = str(path)
        with safe_open(self.path, framework="pt", device="cpu") as fh:
            meta = fh.metadata() or {}
            names = set(fh.keys())
            missing = [n for n in SHARD_TENSORS if n not in names]
            if missing:
                raise ValueError(f"{path}: missing tensors {missing}")
            self.seq_starts = fh.get_tensor("seq_starts").long()
        # the producer's metadata either as one JSON object under a key, or one JSON value per key
        blob = next((meta[k] for k in ("__metadata__", "metadata", "meta", "json") if k in meta), None)
        info = json.loads(blob) if isinstance(blob, str) else {k: _maybe_json(v) for k, v in meta.items()}
        self.seq_ids = [str(s) for s in info.get("seq_ids", [])]
        self.sources = [str(s) for s in info.get("sources", [])]
        self.context_rows = int(info.get("context_rows", 0))
        s = self.seq_starts.numel() - 1
        per = info.get("context_rows_per_seq")  # capfmt (mtp-capture-v1): one value per sequence, preferred
        self.context_per_seq = [int(x) for x in per] if isinstance(per, list) and len(per) == s else [self.context_rows] * s
        if len(self.seq_ids) != s or len(self.sources) != s:
            raise ValueError(f"{path}: {s} windows but {len(self.seq_ids)} seq_ids / {len(self.sources)} sources")

    def __len__(self):
        return self.seq_starts.numel() - 1

    def rows(self, i: int) -> int:
        return int(self.seq_starts[i + 1] - self.seq_starts[i])

    def window(self, i: int, max_rows: int | None = None) -> Window:
        from safetensors import safe_open
        a, b = int(self.seq_starts[i]), int(self.seq_starts[i + 1])
        if max_rows and b - a > max_rows:
            b = a + max_rows
        with safe_open(self.path, framework="pt", device="cpu") as fh:
            g = lambda n: fh.get_slice(n)[a:b]  # noqa: E731
            return Window(g("tokens").long(), g("positions").long(), g("h").to(torch.bfloat16), g("topk_ids").long(),
                          g("topk_logits").float(), g("assist").bool(), self.context_per_seq[i], self.seq_ids[i], self.sources[i],
                          self.path, i)


def _maybe_json(v):
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def list_shards(capture_root: str | os.PathLike, name: str) -> list[str]:
    return sorted(glob.glob(os.path.join(str(capture_root), name, "shard-*.safetensors")))


def write_synthetic_shard(path: str | os.PathLike, windows: list[dict], context_rows: int = 0, seed: int = 0,
                          vocab: int = 1000, scale: float = 1.0, meta_format: str = "keys") -> None:
    """A capture shard in the shared format, for tests: windows = [{"rows": n, "source": s, "seq_id": id, "start": p}]."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    toks, pos, hs, ids, lg, ast, starts = [], [], [], [], [], [], [0]
    for w in windows:
        n = int(w["rows"])
        toks.append(torch.randint(0, vocab, (n,), generator=g, dtype=torch.int32))
        pos.append(torch.arange(int(w.get("start", 0)), int(w.get("start", 0)) + n, dtype=torch.int32))
        hs.append((torch.randn(n, WIDTH, generator=g) * scale).to(torch.bfloat16))
        top = torch.stack([torch.randperm(vocab, generator=g)[:8] for _ in range(n)]).to(torch.int32)
        ids.append(top)
        lg.append(torch.sort(torch.randn(n, 8, generator=g) * 2 + 10, dim=-1, descending=True).values)
        ast.append((torch.rand(n, generator=g) < float(w.get("assist_frac", 0.8))).to(torch.uint8))
        starts.append(starts[-1] + n)
    meta = {"seq_ids": [str(w.get("seq_id", i)) for i, w in enumerate(windows)],
            "sources": [str(w.get("source", "synthetic")) for w in windows], "context_rows": int(context_rows)}
    save_file({"tokens": torch.cat(toks), "positions": torch.cat(pos), "h": torch.cat(hs), "topk_ids": torch.cat(ids),
               "topk_logits": torch.cat(lg), "assist": torch.cat(ast), "seq_starts": torch.tensor(starts, dtype=torch.int64)},
              str(path), metadata={"__metadata__": json.dumps(meta)} if meta_format == "blob"
              else {k: json.dumps(v) for k, v in meta.items()})


# ---- acceptance ----
def parse_rule(spec: str | None):
    """'K:M' -> (K, M) (accept a draft the target ranks top-K within M logits), None/'exact'/'off' -> None."""
    if spec is None or spec in ("exact", "off", ""):
        return None
    k, m = spec.split(":")
    return int(k), float(m)


def accept_matches(drafts: torch.Tensor, topk_ids: torch.Tensor, topk_logits: torch.Tensor, rule) -> torch.Tensor:
    """drafts [..., K] vs the target's top-8 at the drafted positions -> bool matches (engine/accept.accept_block:
    exact argmax, or rank < K and lmax - l(draft) <= M; a draft outside the top 8 ranks >= 8)."""
    exact = drafts == topk_ids[..., 0]
    if rule is None:
        return exact
    top, margin = rule
    hit = topk_ids == drafts.unsqueeze(-1)
    ly = torch.where(hit, topk_logits, torch.full_like(topk_logits, -float("inf"))).max(-1).values
    rank = (topk_logits > ly.unsqueeze(-1)).sum(-1)
    lmax = topk_logits.max(-1).values
    relaxed = hit.any(-1) & (rank < top) & ((lmax - ly) <= margin)
    return exact | relaxed


@dataclass
class AcceptStats:
    """Per-sequence sums for teacher-forced acceptance: chains with all `depth` steps valid, and how many of them
    accept their first j drafts, per rule."""
    depth: int
    rules: dict
    per_seq: dict = field(default_factory=dict)  # seq key -> {"source", "chains", rule: [prefix counts j=1..depth]}

    def add(self, seq_key: str, source: str, drafts: torch.Tensor, topk_ids, topk_logits, valid) -> None:
        full = valid[:, :self.depth].all(-1)
        if not bool(full.any()):
            return
        d, ti, tl = drafts[full, :self.depth], topk_ids[full, :self.depth], topk_logits[full, :self.depth]
        e = self.per_seq.setdefault(seq_key, {"source": source, "chains": 0, **{r: [0] * self.depth for r in self.rules}})
        e["chains"] += int(full.sum())
        for name, rule in self.rules.items():
            prefix = torch.cumprod(accept_matches(d, ti, tl, rule).to(torch.int64), dim=1).sum(0).tolist()
            e[name] = [a + int(b) for a, b in zip(e[name], prefix)]

    def summary(self, bootstrap: int = 1000, seed: int = 0, by_source: bool = True) -> dict:
        out = {"all": _summarize(list(self.per_seq.values()), self.rules, self.depth, bootstrap, seed)}
        if by_source:
            for src in sorted({e["source"] for e in self.per_seq.values()}):
                out[src] = _summarize([e for e in self.per_seq.values() if e["source"] == src], self.rules, self.depth, bootstrap, seed)
        return out


def _summarize(entries, rules, depth, bootstrap, seed):
    import numpy as np
    if not entries:
        return {"sequences": 0, "chains": 0}
    chains = np.array([e["chains"] for e in entries], dtype=np.float64)
    res = {"sequences": len(entries), "chains": int(chains.sum())}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(entries), size=(bootstrap, len(entries))) if bootstrap else None
    for name in rules:
        num = np.array([e[name] for e in entries], dtype=np.float64)  # [S, depth]
        acc = num.sum(0) / chains.sum()
        r = {"prefix_accept": acc.round(6).tolist(),  # P(first j drafts accepted), j = 1..depth
             "conditional_accept": [round(float(acc[0]), 6)] + [round(float(acc[j] / acc[j - 1]), 6) if acc[j - 1] > 0 else 0.0
                                                                for j in range(1, depth)],
             "tokens_per_round": [round(1.0 + float(acc[:k].sum()), 6) for k in range(1, depth + 1)]}
        if bootstrap:
            bn = num[idx].sum(1)  # [B, depth]
            bc = chains[idx].sum(1)[:, None]
            ba = bn / bc
            bt = 1.0 + np.cumsum(ba, axis=1)
            r["prefix_accept_ci95"] = np.percentile(ba, [2.5, 97.5], axis=0).T.round(6).tolist()
            r["tokens_per_round_ci95"] = np.percentile(bt, [2.5, 97.5], axis=0).T.round(6).tolist()
        res[name] = r
    return res


def cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)
