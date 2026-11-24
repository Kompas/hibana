# SPDX-License-Identifier: Apache-2.0
"""Hibana's own Flash-Next MTP head against MtpHeadTorch (fp8 and float modes) on captured (tokens, h) windows.

Hibana's head is built as the engine builds it (`FlashNextMtpHead` with the real FP8 kernels, the target's embedding,
FP8 lm_head and rope; `profiling/flashnext_draft_step.py`'s loader) and driven as `proposers/mtp.py` drives it: the
window's target-conditioned draft KV through `write_kv`, then batches of chains, one `forward` per step, each chain
attending the window's KV before its start plus its own steps' rows. MtpHeadTorch runs the same chains in the same
batches (`chain_subset` over `prefill_kv`), so the per-call FP8 activation scales see the same rows.

Teacher forcing: step j's input token is the data token, its hidden the implementation's own d of step j-1. Drafts for
the comparison are the full lm_head argmax of each implementation (Hibana: the FP8 lm_head); Hibana's served draft head
(`--draft-head w2`: the 2-bit shortlist rescored top 16) is reported beside it.

Prints one JSON verdict (and writes it to --out). Exit 0: every frozen pass line holds; 2: some fail; 1: error.

    python3 mtp_equiv.py --capture-root /data/mtp/capture --set heldout --snapshot /hf --out equiv.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import mtp_torch as mt  # noqa: E402

PASS_LINES = {  # frozen
    "top1_fp8": 0.990, "top1_float": 0.985, "cos_d_mean": 0.999, "cos_d_min": 0.99, "taps_mean": 0.9995, "accept_pt": 0.3,
}


# ---- Hibana's head ----
class OverlayWeights:
    """`SnapshotWeights` with trained mtp.* tensors in front. The routed experts are always read from the base
    snapshot by `FlashNextMtpHead._fp8_experts` (`_map` / `snap`): they are frozen by construction."""

    def __init__(self, base, overlay: dict | None):
        self.base, self.overlay = base, overlay or {}
        self._map, self.snap = base._map, base.snap

    def get(self, name):
        return self.overlay[name] if name in self.overlay else self.base.get(name)

    def __contains__(self, name):
        return name in self.overlay or name in self.base

    def close(self):
        self.base.close()

    def __getattr__(self, item):
        return getattr(self.base, item)


class HibanaHead:
    """FlashNextMtpHead on CUDA with its own draft KV pool, chains run as the proposer runs them."""

    def __init__(self, snapshot, device, max_rows: int, batch: int, depth: int, overlay=None, draft_head: str = "fp8",
                 rescore: int | None = None):
        from hibana.engine.config import current
        from hibana.model.flashnext.hyper import HyperConnection
        from hibana.model.flashnext.linear import TrunkLinear
        from hibana.model.flashnext.model import PREFIX
        from hibana.model.flashnext.mtp import FlashNextMtpHead
        from hibana.model.rope import PartialRotary
        from hibana.target import FLASHNEXT
        from hibana.weights.safetensors import SnapshotWeights
        t = FLASHNEXT
        snap = mt.resolve_snapshot(snapshot)
        ov = mt.load_overlay(overlay) if isinstance(overlay, (str, os.PathLike)) else overlay
        if ov:
            for n in (mt.EXPERTS_GU, mt.EXPERTS_DOWN):
                if n in ov:
                    base_t = mt.Snapshot(snap).get(n)
                    if not torch.equal(base_t.view(torch.int16), ov[n].view(torch.int16)):
                        raise ValueError(f"{n} differs from the snapshot: Hibana's head reads the experts from the snapshot")
                    del base_t
        w = OverlayWeights(SnapshotWeights(snap), {k: v for k, v in (ov or {}).items() if k not in (mt.EXPERTS_GU, mt.EXPERTS_DOWN)})
        self.precision = current().flashnext_trunk or "fp8"
        self.max_rows, self.batch, self.depth = max_rows, batch, depth
        self.num_pages = 1 + max_rows + batch * depth + 8
        try:
            self.model = SimpleNamespace(
                target=t, precision=self.precision, embed=w.get(f"{PREFIX}.embed_tokens.weight").to(device),
                rope=PartialRotary(t.head_dim, t.rotary_dim, t.rope_theta, t.max_position_embeddings, device),
                lm_head=TrunkLinear.load(w, ["lm_head.weight"], device, self.precision),
                mixer=HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", t.hc_count, t.hidden_size, t.rms_norm_eps, device,
                                           combine=False))
            self.head = FlashNextMtpHead(w, device, self.model, self.num_pages, padding_rows=batch)
        finally:
            w.close()
        self.served = None
        if draft_head != "fp8":
            from hibana.proposers.draft_head import RESCORE, ShortlistHead
            self.served = ShortlistHead(self.head.lm_head, None, draft_head, rescore or RESCORE[draft_head])
        self.device = torch.device(device)

    @torch.inference_mode()
    def argmax_full(self, d):
        return self.head.lm_head(self.head.head_input(d)).float().argmax(-1)

    @torch.inference_mode()
    def argmax_served(self, d):
        return self.served.argmax(self.head.head_input(d)).long() if self.served is not None else None

    @torch.inference_mode()
    def window(self, tokens, hidden, positions):
        """write_kv of every head row (one call, as a prefill chunk's ingestion)."""
        m = tokens.shape[0]
        if m > self.max_rows:
            raise ValueError(f"window of {m} head rows > --max-rows {self.max_rows}")
        self.m = m
        self.base_pages = torch.arange(1, 1 + m, dtype=torch.int32, device=self.device)
        self.head.write_kv(tokens, hidden, positions, self.base_pages)

    @torch.inference_mode()
    def chains(self, tokens, hidden, positions, starts, depth, feed="teacher", taps=False):
        """Chains at head rows `starts` (<= batch). Returns {"d": [..], "draft": [..], "served": [..], "taps": {}}."""
        h = self.head
        b = starts.numel()
        m = tokens.shape[0]
        scratch = (1 + self.max_rows + torch.arange(b * depth, device=self.device).view(b, depth)).to(torch.int32)
        tok, hid = tokens[starts], hidden[starts]
        ds, drafts, served, tap = [], [], [], {}
        for j in range(depth):
            pos = (positions[starts] + j).to(torch.int64)
            write = scratch[:, j].contiguous()
            kv = [torch.cat([self.base_pages[:int(r)], scratch[i, :j + 1]]) for i, r in enumerate(starts.tolist())]
            if taps and j == 0:
                h.plan([1] * b, kv)
                u = h._fuse(tok, hid)
                x, res = h.attn_hc.mix(u)
                a = h._attention(x, pos, write, h._attn)
                u2 = h.attn_hc.combine(a, res)
                x2, res2 = h.mlp_hc.mix(u2)
                d = h.mlp_hc.combine(h._moe(x2), res2)
                tap = {"fuse": u.clone(), "attn": u2.clone(), "moe": d.clone()}
            else:
                d = h.forward(tok, hid, pos, write, [1] * b, kv)
            d = d.clone()
            ds.append(d)
            dr = self.argmax_full(d)
            drafts.append(dr)
            served.append(self.argmax_served(d))
            if feed == "teacher":
                idx = starts + j + 1
                tok = torch.where(idx < m, tokens[idx.clamp(max=m - 1)], torch.zeros_like(idx))
            else:
                tok = dr
            hid = d
        return {"d": ds, "draft": drafts, "served": served, "taps": tap}


class TorchStandIn:
    """HibanaHead's interface over an fp8-mode MtpHeadTorch: `--standin` runs the whole tool on CPU (plumbing tests;
    the verdict then compares the emulation with itself)."""

    def __init__(self, snapshot, device, overlay=None, expert_range=None, vocab_rows=None):
        self.t = mt.MtpHeadTorch.from_snapshot(snapshot, device=device, mode="fp8", overlay=overlay, expert_range=expert_range,
                                               vocab_rows=vocab_rows)
        self.model = SimpleNamespace(embed=self.t.embed)
        self.served = None

    @torch.no_grad()
    def window(self, tokens, hidden, positions):
        self.main = self.t.prefill_kv(tokens, hidden, positions)

    @torch.no_grad()
    def chains(self, tokens, hidden, positions, starts, depth, feed="teacher", taps=False):
        o = self.t.chain_subset(tokens, hidden, positions, starts, depth, self.main, feed=feed, taps=taps)
        return {"d": o["d"], "draft": o["draft"], "served": [None] * depth, "taps": o["taps"]}


def windows_of(capture_root, sets, max_windows, window_rows, min_rows=8):
    n = 0
    for s in sets:
        for path in mt.list_shards(capture_root, s):
            shard = mt.CaptureShard(path)
            for i in range(len(shard)):
                if shard.rows(i) < min_rows:
                    continue
                yield s, shard.window(i, window_rows)
                n += 1
                if max_windows and n >= max_windows:
                    return


def pick_starts(valid0: torch.Tensor, k: int) -> torch.Tensor:
    """Up to k eligible chain starts (assistant rows past the context), evenly spaced."""
    elig = valid0.nonzero().flatten()
    if k and elig.numel() > k:
        elig = elig[torch.linspace(0, elig.numel() - 1, k).round().long()]
    return elig


class Agg:
    def __init__(self):
        self.v = {}

    def add(self, name, x):
        x = x.detach().float().flatten().cpu()
        if x.numel():
            self.v.setdefault(name, []).append(x)

    def cat(self, name):
        return torch.cat(self.v[name]) if name in self.v else torch.empty(0)

    def stats(self, name):
        x = self.cat(name)
        if not x.numel():
            return {"n": 0}
        return {"n": int(x.numel()), "mean": round(float(x.mean()), 6), "min": round(float(x.min()), 6)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-root", required=True, help="dir with <set>/shard-*.safetensors (e.g. /data/mtp/capture)")
    ap.add_argument("--set", action="append", dest="sets", help="capture set(s), repeatable (default heldout)")
    ap.add_argument("--snapshot", required=True, help="HF root or snapshot dir of RadixArk/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--weights", help="trained mtp.* safetensors to compare instead of the snapshot's (both sides)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=24)
    ap.add_argument("--window-rows", type=int, default=4096)
    ap.add_argument("--chains-per-window", type=int, default=256, help="0: every eligible start")
    ap.add_argument("--batch", type=int, default=32, help="chains per Hibana forward (proposer batch)")
    ap.add_argument("--draft-head", default="w2", choices=("fp8", "w3", "w2"), help="Hibana's served draft head, reported beside")
    ap.add_argument("--fast-rule", default="3:2.0", help="K:M relaxed rule of the fast profile for MTP (HIBANA_ACCEPT_DEFAULT)")
    ap.add_argument("--extra-rule", action="append", default=["2:1.0"], help="more K:M rules to report")
    ap.add_argument("--feed", default="teacher", choices=("teacher", "argmax"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write the verdict JSON here too")
    ap.add_argument("--dump-dir", help="save per-window tensors for a CPU post-mortem")
    ap.add_argument("--dump-windows", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="resolve paths and imports, read shard metadata, exit")
    ap.add_argument("--standin", action="store_true", help="CPU plumbing test: an fp8 MtpHeadTorch stands in for Hibana's head")
    ap.add_argument("--expert-range", help="a:b experts only (with --standin, CPU)")
    ap.add_argument("--vocab-rows", type=int, help="first N vocabulary rows only (with --standin, CPU)")
    args = ap.parse_args(argv)
    sets = args.sets or ["heldout"]
    rules = {"exact": None, "fast": mt.parse_rule(args.fast_rule)}
    for r in args.extra_rule:
        if r != args.fast_rule:
            rules[f"relax_{r}"] = mt.parse_rule(r)

    if args.dry_run:
        return dry_run(args, sets, need_hibana=True)

    t0 = time.time()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    er = tuple(int(x) for x in args.expert_range.split(":")) if args.expert_range else None
    if args.standin:
        hib = TorchStandIn(args.snapshot, dev, args.weights, er, args.vocab_rows)
    else:
        hib = HibanaHead(args.snapshot, dev, args.window_rows, args.batch, args.depth, overlay=args.weights, draft_head=args.draft_head)
    # share the target's embedding with MtpHeadTorch; its lm_head is the BF16 checkpoint matrix (float) / re-quantized (fp8)
    tor = mt.MtpHeadTorch.from_snapshot(args.snapshot, device=dev, mode="fp8", overlay=args.weights, embed=hib.model.embed,
                                        expert_range=er, vocab_rows=args.vocab_rows)
    tor.eval()
    load_s = time.time() - t0
    impls = ("hibana", "fp8", "float", "hibana_served")
    stats = {k: mt.AcceptStats(args.depth, rules) for k in impls}
    agg = Agg()
    n_win = n_chain = 0
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
    for set_name, win in windows_of(args.capture_root, sets, args.max_windows, args.window_rows):
        tokens, hidden, pos = (x.to(dev) for x in win.head_inputs(dev))
        tids, tlog, valid = win.targets(args.depth, dev)
        exists = (torch.arange(tokens.shape[0], device=dev)[:, None] + torch.arange(args.depth, device=dev)[None, :]) < tokens.shape[0]
        starts_all = pick_starts(valid[:, 0], args.chains_per_window)
        if not starts_all.numel():
            continue
        with torch.inference_mode():
            hib.window(tokens, hidden, pos)
            mains = {}
            for mode in ("fp8", "float"):
                tor.mode = mode
                mains[mode] = tor.prefill_kv(tokens, hidden, pos)
        res = {k: {"draft": [], "d": []} for k in ("hibana", "fp8", "float")}
        served = []
        dump = {"starts": starts_all.cpu(), "taps": {}} if args.dump_dir and n_win < args.dump_windows else None
        for a in range(0, starts_all.numel(), args.batch):
            st = starts_all[a:a + args.batch]
            first = a == 0
            with torch.inference_mode():
                ho = hib.chains(tokens, hidden, pos, st, args.depth, args.feed, taps=True)
                outs = {"hibana": ho}
                for mode in ("fp8", "float"):
                    tor.mode = mode
                    outs[mode] = tor.chain_subset(tokens, hidden, pos, st, args.depth, mains[mode], feed=args.feed, taps=True)
            ex = exists[st]
            for j in range(args.depth):
                m = ex[:, j]
                for mode in ("fp8", "float"):
                    agg.add(f"top1_{mode}_d{j + 1}", (outs[mode]["draft"][j] == ho["draft"][j])[m])
                    agg.add(f"cos_{mode}_d{j + 1}", mt.cosine_rows(outs[mode]["d"][j], ho["d"][j])[m])
            for mode in ("fp8", "float"):
                for tap in ("fuse", "attn", "moe"):
                    agg.add(f"tap_{mode}_{tap}", mt.cosine_rows(outs[mode]["taps"][tap], ho["taps"][tap]))
            for k in ("hibana", "fp8", "float"):
                res[k]["draft"].append(torch.stack(outs[k]["draft"], 1))
                if dump is not None:
                    res[k]["d"].append(torch.stack([x.to(torch.bfloat16) for x in outs[k]["d"]], 1).cpu())
            if ho["served"][0] is not None:
                served.append(torch.stack(ho["served"], 1))
            if dump is not None and first:
                dump["taps"] = {k: {t: v.cpu() for t, v in outs[k]["taps"].items()} for k in ("hibana", "fp8", "float")}
        key = f"{set_name}:{win.seq_id}"
        for k in ("hibana", "fp8", "float"):
            dr = torch.cat(res[k]["draft"])
            stats[k].add(key, win.source, dr, tids[starts_all], tlog[starts_all], valid[starts_all])
        if served:
            stats["hibana_served"].add(key, win.source, torch.cat(served), tids[starts_all], tlog[starts_all], valid[starts_all])
        if dump is not None:
            dump.update({"tokens": tokens.cpu(), "positions": pos.cpu(),
                         "drafts": {k: torch.cat(res[k]["draft"]).cpu() for k in res},
                         "d": {k: torch.cat(res[k]["d"]) for k in res}})
            torch.save(dump, Path(args.dump_dir) / f"window-{n_win:03d}.pt")
        n_win += 1
        n_chain += int(starts_all.numel())
        print(json.dumps({"window": n_win, "chains": int(starts_all.numel()),
                          "top1_fp8_d1": agg.stats("top1_fp8_d1").get("mean"), "cos_fp8_d1": agg.stats("cos_fp8_d1").get("mean")}),
              file=sys.stderr, flush=True)

    verdict = build_verdict(args, agg, stats, rules, n_win, n_chain, load_s, time.time() - t0)
    verdict["hibana_trunk_precision"] = getattr(hib, "precision", "standin")  # the emulation assumes fp8 (the default)
    if torch.cuda.is_available() and dev.type == "cuda":
        verdict["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    text = json.dumps(verdict, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0 if verdict["pass"] else 2


def pooled(agg, prefix, depth):
    xs = [agg.cat(f"{prefix}_d{j + 1}") for j in range(depth)]
    x = torch.cat(xs) if xs else torch.empty(0)
    return x


def build_verdict(args, agg, stats, rules, n_win, n_chain, load_s, total_s) -> dict:
    depth = args.depth
    summ = {k: s.summary(bootstrap=args.bootstrap) for k, s in stats.items()}
    out = {"standin": bool(args.standin), "windows": n_win, "chains": n_chain, "depth": depth, "feed": args.feed,
           "rules": {k: v for k, v in rules.items()},
           "weights": args.weights or "snapshot", "load_s": round(load_s, 1), "elapsed_s": round(total_s, 1)}
    top1 = {mode: float(pooled(agg, f"top1_{mode}", depth).mean()) if n_chain else float("nan") for mode in ("fp8", "float")}
    cos = {mode: pooled(agg, f"cos_{mode}", depth) for mode in ("fp8", "float")}
    out["top1_agreement"] = {mode: {"pooled": round(top1[mode], 6), **{f"d{j + 1}": agg.stats(f"top1_{mode}_d{j + 1}").get("mean")
                                                                         for j in range(depth)}} for mode in ("fp8", "float")}
    out["cos_d"] = {mode: {"mean": round(float(cos[mode].mean()), 6) if cos[mode].numel() else None,
                           "min": round(float(cos[mode].min()), 6) if cos[mode].numel() else None,
                           **{f"d{j + 1}": agg.stats(f"cos_{mode}_d{j + 1}") for j in range(depth)}} for mode in ("fp8", "float")}
    out["taps"] = {mode: {t: agg.stats(f"tap_{mode}_{t}") for t in ("fuse", "attn", "moe")} for mode in ("fp8", "float")}
    out["acceptance"] = {k: summ[k].get("all") for k in summ}
    out["acceptance_by_source"] = {k: {s: v for s, v in summ[k].items() if s != "all"} for k in summ}
    checks = {}
    checks["top1_fp8"] = {"value": round(top1["fp8"], 6), "min": PASS_LINES["top1_fp8"], "pass": top1["fp8"] >= PASS_LINES["top1_fp8"]}
    checks["top1_float"] = {"value": round(top1["float"], 6), "min": PASS_LINES["top1_float"], "pass": top1["float"] >= PASS_LINES["top1_float"]}
    cm, cn = out["cos_d"]["fp8"]["mean"], out["cos_d"]["fp8"]["min"]
    checks["cos_d_mean_fp8"] = {"value": cm, "min": PASS_LINES["cos_d_mean"], "pass": cm is not None and cm >= PASS_LINES["cos_d_mean"]}
    checks["cos_d_min_fp8"] = {"value": cn, "min": PASS_LINES["cos_d_min"], "pass": cn is not None and cn >= PASS_LINES["cos_d_min"]}
    for t in ("fuse", "attn", "moe"):
        v = out["taps"]["fp8"][t].get("mean")
        checks[f"tap_{t}_fp8"] = {"value": v, "min": PASS_LINES["taps_mean"], "pass": v is not None and v >= PASS_LINES["taps_mean"]}
    ha, ta = summ["hibana"].get("all", {}), summ["fp8"].get("all", {})
    for r in rules:
        if r in ha and r in ta:
            diffs = [round(100 * abs(x - y), 4) for x, y in zip(ha[r]["prefix_accept"], ta[r]["prefix_accept"])]
            checks[f"accept_{r}_pt"] = {"value": diffs, "max": PASS_LINES["accept_pt"], "pass": max(diffs) <= PASS_LINES["accept_pt"]}
        else:
            checks[f"accept_{r}_pt"] = {"value": None, "pass": False}
    out["checks"] = checks
    out["pass"] = bool(n_chain) and all(c["pass"] for c in checks.values())
    out["pass_lines"] = PASS_LINES
    return out


def dry_run(args, sets, need_hibana: bool) -> int:
    report = {"snapshot": None, "sets": {}, "imports": {}}
    snap = mt.Snapshot(args.snapshot)
    report["snapshot"] = str(snap.dir)
    report["load_check"] = mt.check_names(snap, mt.load_overlay(getattr(args, "weights", None)))
    report["target_head"] = {n: list(snap.shape(n)) for n in (mt.EMBED_NAME, mt.LM_HEAD_NAME)}
    if need_hibana:
        for mod in ("hibana.model.flashnext.mtp", "hibana.model.flashnext.linear", "hibana.model.flashnext.hyper", "hibana.model.rope",
                    "hibana.proposers.draft_head", "hibana.weights.safetensors", "hibana.target", "hibana.engine.config"):
            try:
                __import__(mod)
                report["imports"][mod] = "ok"
            except Exception as e:  # noqa: BLE001
                report["imports"][mod] = f"FAIL {e!r}"[:300]
        report["cuda"] = torch.cuda.is_available()
    for s in sets:
        shards = mt.list_shards(args.capture_root, s)
        entry = {"shards": len(shards)}
        if shards:
            sh = mt.CaptureShard(shards[0])
            entry.update({"first_windows": len(sh), "context_rows": sh.context_rows, "sources": sorted(set(sh.sources))})
        report["sets"][s] = entry
    report["ok"] = all(v == "ok" for v in report["imports"].values())
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(json.dumps({"pass": False, "error": repr(e)[:500]}))
        sys.exit(1)
