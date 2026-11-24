# SPDX-License-Identifier: Apache-2.0
"""CPU judges for MtpHeadTorch against Hibana's plain-torch pieces, on the real checkpoint where it fits in RAM.

Runs under pytest or as a script (prints every measured number). Needs the Flash-Next snapshot (MTP_SNAPSHOT, default
/hf) and Hibana importable (run from /opt/hibana in the image). The full head is loaded with a slice of the experts
(MTP_EXPERTS, default 0:16) and of the vocabulary (MTP_VOCAB, default 4096); MTP_FULL_LOAD=1 also loads all 512 experts.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mtp_torch as mt  # noqa: E402

SNAP = os.environ.get("MTP_SNAPSHOT", "/hf")
ER = tuple(int(x) for x in os.environ.get("MTP_EXPERTS", "0:16").split(":"))
VOCAB = int(os.environ.get("MTP_VOCAB", "4096"))
RESULTS: dict = {}
_HEAD = {}


def have_snapshot() -> bool:
    try:
        mt.resolve_snapshot(SNAP)
        return True
    except FileNotFoundError:
        return False


def skip_without_snapshot():
    if not have_snapshot():
        import pytest
        pytest.skip("no snapshot")


def head(mode="float") -> mt.MtpHeadTorch:
    if "h" not in _HEAD:
        torch.manual_seed(0)
        _HEAD["h"] = mt.MtpHeadTorch.from_snapshot(SNAP, mode="float", compute_dtype=torch.float32, expert_range=ER, vocab_rows=VOCAB)
    h = _HEAD["h"]
    h.mode = mode
    return h


def hidden_rows(n, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, mt.WIDTH, generator=g) * scale


def tokens_of(n, seed=0):
    return torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


def record(name, **kv):
    RESULTS[name] = kv
    print(f"[{name}] " + json.dumps(kv), flush=True)


# ---- strict load ----
def test_strict_load():
    skip_without_snapshot()
    snap = mt.Snapshot(SNAP)
    rep = mt.check_names(snap)
    assert rep["tensors"] == 31 and rep["trainable"] == 25 and len(rep["frozen"]) == 6
    h = head()
    n_train = sum(p.numel() for p in h.trainable_parameters())
    # corrupted snapshots: a missing and an unexpected mtp tensor, a wrong overlay shape
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = json.loads((snap.dir / "model.safetensors.index.json").read_text())
        for case in ("missing", "unexpected"):
            d = tmp / case
            d.mkdir()
            wm = dict(idx["weight_map"])
            if case == "missing":
                wm.pop("mtp.fc_hidden.weight")
            else:
                wm["mtp.layers.0.extra.weight"] = wm["mtp.fc_hidden.weight"]
            for f in set(wm.values()):
                os.symlink(snap.dir / f, d / f)
            (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
            try:
                mt.check_names(mt.Snapshot(d))
                raise AssertionError(f"{case}: strict load accepted it")
            except ValueError as e:
                assert ("missing" in str(e)) and (case in str(e) or "extra" in str(e) or "fc_hidden" in str(e))
        try:
            mt.check_names(snap, {"mtp.fc_hidden.weight": torch.zeros(10, 10, dtype=torch.bfloat16)})
            raise AssertionError("bad overlay shape accepted")
        except ValueError:
            pass
        try:
            mt.check_names(snap, {"mtp.nope.weight": torch.zeros(1, dtype=torch.bfloat16)})
            raise AssertionError("unexpected overlay name accepted")
        except ValueError:
            pass
        tensors = {n: torch.zeros(1) for n in mt.EXPECTED if n != "mtp.fc_embedding.weight"}
        try:
            mt.MtpHeadTorch(tensors, torch.zeros(1, 1), torch.zeros(1, 1))
            raise AssertionError("module accepted a missing tensor")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp)
    full = None
    if os.environ.get("MTP_FULL_LOAD") == "1":
        # all 31 tensors whole (5 GB of experts), BF16 experts kept as stored; the vocabulary stays sliced
        fh = mt.MtpHeadTorch.from_snapshot(SNAP, mode="float", compute_dtype=torch.float32, vocab_rows=VOCAB)
        full = {"experts_gu": list(fh.experts_gu.shape), "experts_down": list(fh.experts_down.shape), "router": list(fh.router.shape),
                "indexer_buffers": [n for n in mt.INDEXER if hasattr(fh, mt.key(n))]}
        with torch.no_grad():
            d = fh.chain_all(tokens_of(6), hidden_rows(6), torch.arange(100, 106), 2, draft=False)["d"][-1]
        full["finite"] = bool(torch.isfinite(d).all())
        del fh, d
    record("strict_load", tensors=rep["tensors"], trainable_tensors=rep["trainable"], trainable_params=n_train,
           frozen=rep["frozen"], unused=rep["unused"], expert_range=h.load_report["expert_range"], full_load=full)


# ---- submodule parity against Hibana's plain-torch implementations (fp32) ----
def test_parity_norms_rope():
    skip_without_snapshot()
    from hibana.model.flashnext.hyper import grouped_gemma_rms
    from hibana.model.flashnext.reference import gemma_rms as ref_rms, rope_neox
    from hibana.model.norm import GemmaRMSNorm
    from hibana.model.rope import PartialRotary
    h = head()
    out = {}
    x = F.embedding(tokens_of(64), h.embed).float()
    w = h.param("mtp.pre_fc_norm_embedding.weight").detach()
    out["gemma_rms"] = float((mt.gemma_rms(x, w) - ref_rms(x, w, mt.EPS)).abs().max())
    out["gemma_rms_vs_GemmaRMSNorm"] = float((mt.gemma_rms(x, w) - GemmaRMSNorm(w.to(torch.bfloat16), mt.EPS).torch_forward(x)).abs().max())
    hh = hidden_rows(32)
    wn = h.param("mtp.layers.0.attn_hyper_connection.hc_norm.weight").detach()
    out["grouped_rms"] = float((mt.grouped_rms(hh, wn) - grouped_gemma_rms(hh, wn, mt.EPS, mt.HIDDEN)).abs().max())
    q = torch.randn(32, mt.N_HEADS, mt.HEAD_DIM) * 3
    qw = h.param("mtp.layers.0.self_attn.q_norm.weight").detach()
    pos = torch.randint(0, 200_000, (32,))
    cos, sin = mt.rope_cos_sin(pos, "cpu")
    mine = mt.head_norm_rope(q, qw, cos, sin, torch.float32)
    normed = ref_rms(q, qw, mt.EPS)
    out["rope_vs_rope_neox"] = float((mine - rope_neox(normed, pos, mt.ROTARY_DIM, mt.ROPE_THETA)).abs().max())
    pr = PartialRotary(mt.HEAD_DIM, mt.ROTARY_DIM, mt.ROPE_THETA, 200_001, torch.device("cpu"))
    out["rope_vs_PartialRotary"] = float((mine - pr(pos, normed)).abs().max())
    record("parity_norms_rope", **out)
    assert all(v <= 1e-3 for v in out.values()), out


def _hc_ref(h, site):
    from hibana.model.flashnext.hyper import HyperConnection
    p = mt._HC_SITES[site]
    g = lambda n: h.param(f"{p}.{n}").detach().to(torch.bfloat16)  # noqa: E731
    hc = HyperConnection(g("hc_norm.weight"), g("input_mix_weight_down.weight"), g("input_mix_weight_up.weight"),
                         g("block_inject_weight.weight") if site != "mixer" else None, mt.HC, mt.HIDDEN, mt.EPS, precision="bf16", fused=False)
    # the same matrices in FP32 (the plain path multiplies with whatever dtype the weights have)
    hc.down.weight = hc.down.weight.float()
    hc.up.weight = hc.up.weight.float()
    if hc.inject is not None:
        hc.inject.weight = hc.inject.weight.float()
    hc.norm = hc.norm.float()
    return hc


def test_parity_hyper_connections():
    skip_without_snapshot()
    h = head()
    out = {}
    x = hidden_rows(48, seed=3)
    block = torch.randn(48, mt.HIDDEN)
    for site in ("attn", "mlp", "mixer"):
        hc = _hc_ref(h, site)
        rm, (rh, rn) = hc.mix(x)
        with torch.no_grad():
            mm, mn = h.mix(x, site)
        out[f"{site}.mix"] = float((mm - rm).abs().max())
        out[f"{site}.normed"] = float((mn - rn).abs().max())
        if site != "mixer":
            with torch.no_grad():
                mc = h.combine(block, x, mn, site)
            out[f"{site}.combine"] = float((mc - hc.combine(block, (rh, rn))).abs().max())
    record("parity_hyper_connections", **out)
    assert all(v <= 1e-3 for v in out.values()), out


def test_parity_fuse_attention_moe():
    """fuse with Hibana's GemmaRMSNorm/TrunkLinear (plain paths), the attention block and the MoE against naive loops."""
    skip_without_snapshot()
    from hibana.model.flashnext.linear import TrunkLinear
    from hibana.model.flashnext.reference import gemma_rms as ref_rms, rope_neox
    from hibana.model.norm import GemmaRMSNorm
    h = head()
    P = lambda n: h.param(n).detach()  # noqa: E731
    out = {}
    t = 20
    tok, hid, pos = tokens_of(t, 5), hidden_rows(t, 5), torch.arange(1000, 1000 + t)
    # fuse: FlashNextMtpHead._fuse with the plain norm and FP32 TrunkLinears
    fc_e, fc_h = TrunkLinear(P("mtp.fc_embedding.weight")), TrunkLinear(P("mtp.fc_hidden.weight"))
    ne = GemmaRMSNorm(P("mtp.pre_fc_norm_embedding.weight").to(torch.bfloat16), mt.EPS)
    nh = GemmaRMSNorm(P("mtp.pre_fc_norm_hidden.weight").to(torch.bfloat16), mt.EPS)
    ne.weight_plus_one = 1.0 + P("mtp.pre_fc_norm_embedding.weight").float()
    nh.weight_plus_one = 1.0 + P("mtp.pre_fc_norm_hidden.weight").float()
    e = fc_e(ne.torch_forward(F.embedding(tok, h.embed).float()))
    hh = fc_h(nh.torch_forward(hid).view(-1, mt.HC, mt.HIDDEN))
    ref_u = (hh + e.unsqueeze(-2)).reshape(-1, mt.WIDTH)
    with torch.no_grad():
        u = h.fuse(tok, hid)
    out["fuse"] = float((u - ref_u).abs().max())
    # attention: Hibana's q|gate layout, plain q/k norm + rope_neox, a naive causal softmax per head (GQA 12:1)
    x = torch.randn(t, mt.HIDDEN)
    a = "mtp.layers.0.self_attn"
    qkv = TrunkLinear(torch.cat([P(f"{a}.q_proj.weight"), P(f"{a}.k_proj.weight"), P(f"{a}.v_proj.weight")]))(x)
    qg = qkv[:, :2 * mt.Q_SIZE].view(t, mt.N_HEADS, 2 * mt.HEAD_DIM)
    q = rope_neox(ref_rms(qg[..., :mt.HEAD_DIM], P(f"{a}.q_norm.weight"), mt.EPS), pos, mt.ROTARY_DIM, mt.ROPE_THETA)
    gate = qg[..., mt.HEAD_DIM:]
    k = rope_neox(ref_rms(qkv[:, 2 * mt.Q_SIZE:2 * mt.Q_SIZE + mt.KV_SIZE].view(t, mt.N_KV, mt.HEAD_DIM), P(f"{a}.k_norm.weight"), mt.EPS),
                  pos, mt.ROTARY_DIM, mt.ROPE_THETA)
    v = qkv[:, 2 * mt.Q_SIZE + mt.KV_SIZE:].view(t, mt.N_KV, mt.HEAD_DIM)
    att = torch.empty(t, mt.N_HEADS, mt.HEAD_DIM)
    for i in range(t):
        for hd in range(mt.N_HEADS):
            kv = hd // (mt.N_HEADS // mt.N_KV)
            s = (k[:i + 1, kv] @ q[i, hd]) / mt.HEAD_DIM ** 0.5
            att[i, hd] = torch.softmax(s, 0) @ v[:i + 1, kv]
    ref_a = TrunkLinear(P(f"{a}.o_proj.weight"))((att * torch.sigmoid(gate)).reshape(t, mt.Q_SIZE))
    with torch.no_grad():
        mq, mk, mv, mg = h.qkv(x, pos)
        ma = h.out_proj(h.attend(mq, (mk, mv), torch.arange(t), [(mk, mv)]), mg)
    out["qk_norm_rope_q"] = float((mq - q).abs().max())
    out["attention_block"] = float((ma - ref_a).abs().max())
    # MoE: a per-token loop (softmax top-10 renormalized over the loaded experts, silu-mul, gated shared expert)
    xm = torch.randn(12, mt.HIDDEN)
    m = "mtp.layers.0.mlp"
    ref = torch.empty(12, mt.HIDDEN)
    for i in range(12):
        p = torch.softmax(xm[i] @ h.router.float().T, 0)
        wv, ids = p.topk(mt.TOPK)
        wv = wv / wv.sum()
        acc = torch.zeros(mt.HIDDEN)
        for wi, e in zip(wv, ids):
            gu = h.experts_gu[e].float() @ xm[i]
            acc += wi * (h.experts_down[e].float() @ (F.silu(gu[:mt.INTER]) * gu[mt.INTER:]))
        sgu = torch.cat([P(f"{m}.shared_expert.gate_proj.weight"), P(f"{m}.shared_expert.up_proj.weight")]) @ xm[i]
        sh = P(f"{m}.shared_expert.down_proj.weight") @ (F.silu(sgu[:mt.INTER]) * sgu[mt.INTER:])
        ref[i] = acc + torch.sigmoid(xm[i] @ P(f"{m}.shared_expert_gate.weight").float().reshape(-1)) * sh
    with torch.no_grad():
        mo = h.moe(xm)
    out["moe"] = float((mo - ref).abs().max())
    out["moe_rel"] = float((mo - ref).abs().max() / ref.abs().max())
    record("parity_fuse_attention_moe", **out)
    assert out["fuse"] <= 1e-3 and out["qk_norm_rope_q"] <= 1e-3 and out["attention_block"] <= 1e-3 and out["moe"] <= 1e-3, out


def test_fp8_quantizers_match_hibana():
    """E4M3 weight packing bitwise against TrunkLinear.from_bf16 and FlashNextMtpHead._fp8_experts."""
    skip_without_snapshot()
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from hibana.model.flashnext.linear import TrunkLinear
    from hibana.model.flashnext.mtp import FlashNextMtpHead
    h = head()
    w = h.param("mtp.fc_embedding.weight").detach().to(torch.bfloat16)
    ref = TrunkLinear.from_bf16(w, "fp8")
    q, s = mt.quant_weight(w)
    out = {"trunk_weight_bitwise": bool(torch.equal(q.view(torch.uint8), ref.weight.view(torch.uint8))),
           "trunk_scale_equal": bool(torch.equal(s, ref.scale))}
    tmp = Path(tempfile.mkdtemp())
    try:
        stack = h.experts_gu[:4].contiguous()
        save_file({"x": stack}, str(tmp / "e.safetensors"))
        fake = SimpleNamespace(_map={"x": "e.safetensors"}, snap=tmp)
        rq, rs = FlashNextMtpHead._fp8_experts(fake, "x", torch.device("cpu"))
        mq, ms = mt.MtpHeadTorch._quant_experts(stack)
        out["experts_bitwise"] = bool(torch.equal(mq.view(torch.uint8), rq.view(torch.uint8)))
        out["experts_scale_equal"] = bool(torch.equal(ms, rs))
    finally:
        shutil.rmtree(tmp)
    record("fp8_quantizers", **out)
    assert all(out.values()), out


def test_accept_rule_matches_accept_block():
    from hibana.engine.accept import MAX_STOPS, accept_block
    g = torch.Generator().manual_seed(7)
    b, k, v = 400, 4, 40
    out = {}
    for spec in ("exact", "3:2.0", "2:1.0"):
        rule = mt.parse_rule(spec)
        logits = torch.randn(b, k, v, generator=g) * 1.5
        z = torch.cat([logits.argmax(-1), torch.zeros(b, 1, dtype=torch.long)], dim=1)
        # drafts: often the argmax, often the runner-up, sometimes anything
        top = logits.topk(3, dim=-1).indices
        pick = torch.randint(0, 4, (b, k), generator=g)
        y = torch.where(pick < 3, top.gather(-1, pick.clamp(max=2).unsqueeze(-1)).squeeze(-1), torch.randint(0, v, (b, k), generator=g))
        relax = torch.tensor([[rule[0] if rule else 0] * b, [rule[1] if rule else 0.0] * b], dtype=torch.float32)
        acc, _, _ = accept_block(y, z, logits if rule else None, relax, torch.full((b,), 99), torch.full((b, MAX_STOPS), -1))
        tv, ti = logits.topk(8, dim=-1)
        mine = torch.cumprod(mt.accept_matches(y, ti, tv, rule).long(), dim=1).sum(1)
        out[spec] = {"agree": float((mine == acc).float().mean()), "mean_accepted": float(acc.float().mean())}
    record("accept_rule", **out)
    assert all(o["agree"] == 1.0 for o in out.values()), out


# ---- causality, chained depth, gradients ----
def _chain_outputs(h, tok, hid, pos, depth):
    with torch.no_grad():
        return h.chain_all(tok, hid, pos, depth, feed="teacher", draft=False)["d"]


def test_causality():
    skip_without_snapshot()
    h = head()
    m, depth, t = 20, 4, 11
    tok, hid, pos = tokens_of(m, 1), hidden_rows(m, 1), torch.arange(500, 500 + m)
    base = _chain_outputs(h, tok, hid, pos, depth)
    out = {}
    for what in ("token", "hidden"):
        tok2, hid2 = tok.clone(), hid.clone()
        if what == "token":
            tok2[t] = (tok2[t] + 1) % VOCAB
        else:
            hid2[t] = hid2[t] + torch.randn(mt.WIDTH)
        pert = _chain_outputs(h, tok2, hid2, pos, depth)
        before, after = 0.0, 0.0
        for j in range(depth):
            q = torch.arange(m) + j  # the head position of chain r's step j (its token's row)
            diff = (pert[j] - base[j]).abs().amax(-1)
            before = max(before, float(diff[q < t].max()))
            after = max(after, float(diff[q >= t].max()))
        out[what] = {"max_change_before_t": before, "max_change_at_or_after_t": after}
    record("causality", t=t, **out)
    assert all(o["max_change_before_t"] == 0.0 and o["max_change_at_or_after_t"] > 0 for o in out.values()), out


def _chain_compare(h, tok, hid, pos, m, depth, feed="teacher"):
    with torch.no_grad():
        batched = h.chain_all(tok, hid, pos, depth, feed=feed, draft=feed == "argmax")
        main = h.prefill_kv(tok, hid, pos)
        worst, rel = 0.0, 0.0
        for r in range(m):
            inc = h.chain_subset(tok, hid, pos, torch.tensor([r]), depth, main, feed=feed)
            for j in range(depth):
                dd = float((inc["d"][j][0] - batched["d"][j][r]).abs().max())
                worst, rel = max(worst, dd), max(rel, dd / float(batched["d"][j][r].abs().max()))
    return worst, rel


def test_chained_batched_equals_incremental():
    """Batched K-pass (every row a chain, one call per step) against each chain alone step by step over the prefilled
    KV. Judged in float64 compute (matmuls whose blocking depends on the batch size are then exact to 1e-12; the
    row-wise FP32 parts are batch-independent); the fp32 numbers are reported beside (attention logits are sharp, so
    fp32 BLAS-order noise shows at ~1e-4 relative)."""
    skip_without_snapshot()
    h = head()
    m, depth = 18, 4
    tok, hid, pos = tokens_of(m, 2), hidden_rows(m, 2), torch.arange(77, 77 + m)
    fp32_abs, fp32_rel = _chain_compare(h, tok, hid, pos, m, depth)
    h.compute_dtype = torch.float64
    try:
        f64_abs, f64_rel = _chain_compare(h, tok, hid.double(), pos, m, depth)
        f64_argmax_abs, _ = _chain_compare(h, tok, hid.double(), pos, m, depth, feed="argmax")
    finally:
        h.compute_dtype = torch.float32
    with torch.no_grad():
        batched = h.chain_all(tok, hid, pos, depth, feed="teacher", draft=True)
        main = h.prefill_kv(tok, hid, pos)
        sub = h.chain_subset(tok, hid, pos, torch.tensor([3, 9, 17, 0, 5]), depth, main)
        worst_sub = max(float((sub["d"][j] - batched["d"][j][[3, 9, 17, 0, 5]]).abs().max()) for j in range(depth))
        drafts_equal = all(torch.equal(inc_d, b_d) for inc_d, b_d in zip(sub["draft"], [d[[3, 9, 17, 0, 5]] for d in batched["draft"]]))
    record("chained_depth", rows=m, depth=depth, f64_max_abs_incremental_vs_batched=f64_abs, f64_max_rel=f64_rel,
           fp32_max_abs=fp32_abs, fp32_max_rel=fp32_rel, fp32_max_abs_subset5_vs_batched=worst_sub,
           fp32_drafts_equal=drafts_equal, f64_argmax_feed_max_abs=f64_argmax_abs)
    assert f64_abs <= 1e-4 and f64_argmax_abs <= 1e-4, (f64_abs, f64_argmax_abs)


def test_gradients_finite():
    skip_without_snapshot()
    from mtp_train import window_loss
    h = head()
    h.train()
    n = 24
    tmp = Path(tempfile.mkdtemp())
    try:
        mt.write_synthetic_shard(tmp / "s.safetensors", [{"rows": n, "source": "zeroclaw", "assist_frac": 1.0}], vocab=VOCAB)
        win = mt.CaptureShard(tmp / "s.safetensors").window(0)
    finally:
        shutil.rmtree(tmp)
    v = win.targets(4)[2]
    denom = [int(v[:, j].sum()) for j in range(4)]
    for p in h.trainable_parameters():
        p.grad = None
    loss, _ = window_loss(h, win, 4, [1, .8, .6, .5], 0.3, denom, torch.device("cpu"))
    loss.backward()
    norms = {n_: float(h.param(n_).grad.norm()) if h.param(n_).grad is not None else None for n_ in mt.TRAINABLE}
    finite = all(x is not None and torch.isfinite(h.param(k).grad).all() for k, x in norms.items())
    nonzero = sum(1 for x in norms.values() if x)
    frozen_grad = [b for b in ("router", "experts_gu", "experts_down", "embed", "lm_head") if getattr(h, b).requires_grad]
    record("gradients", loss=float(loss), trainable_tensors=len(norms), finite=finite, nonzero=nonzero,
           min_norm=min(x for x in norms.values() if x is not None), max_norm=max(x for x in norms.values() if x is not None),
           frozen_requiring_grad=frozen_grad)
    for p in h.trainable_parameters():
        p.grad = None
    h.eval()
    assert finite and nonzero == len(mt.TRAINABLE) and not frozen_grad


def test_fp8_mode_close_to_float():
    skip_without_snapshot()
    h = head()
    m, depth = 32, 4
    tok, hid, pos = tokens_of(m, 4), hidden_rows(m, 4).to(torch.bfloat16), torch.arange(2000, 2000 + m)
    with torch.no_grad():
        h.mode = "float"
        f = h.chain_all(tok, hid.float(), pos, depth)
        h.mode = "fp8"
        q = h.chain_all(tok, hid, pos, depth)
        h.mode = "float"
    cs = [mt.cosine_rows(q["d"][j], f["d"][j]) for j in range(depth)]
    agree = [float((q["draft"][j] == f["draft"][j]).float().mean()) for j in range(depth)]
    record("fp8_vs_float", note="random hidden rows (out of distribution); a smoke test of the fp8 path, not a judge",
           mean_cos_per_depth=[round(float(c.mean()), 5) for c in cs], min_cos_per_depth=[round(float(c.min()), 5) for c in cs],
           top1_agreement_per_depth=agree)
    assert float(cs[0].mean()) > 0.98, cs[0]


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}"[:2000], flush=True)
    print(json.dumps(RESULTS, indent=1, default=str))
    sys.exit(1 if fails else 0)
