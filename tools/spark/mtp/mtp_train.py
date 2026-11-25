# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the Flash-Next MTP draft head on captured target hyper streams (MtpHeadTorch, float mode).

Trainable (~88M): fc_embedding, fc_hidden, both pre-fc norms, the three hyper-connections, attention q/k/v/o and q/k
norms, the shared expert and its gate. Frozen: routed experts, router, indexer, embedding, lm_head.

Loss on assistant rows (a chain's start at or past the window's context rows): chained depth K (default 4, weights
1, .8, .6, .5), step j fed the teacher token and its own d as hidden; per step CE against the target's argmax plus
--kl-weight x KL(target top-8 || draft restricted to those 8 tokens). Windows up to --window-rows rows with their true
positions; context rows are inputs only.

Data: 70% agent rows (the `train` set, sources containing "zeroclaw" x5) and 30% `general`, sampled with replacement
per step (seeded by (seed, step), so a resumed run sees the same windows). FP32 master weights, BF16 compute, AdamW,
linear warmup then cosine. Held-out teacher-forced acceptance (depth 1..K, exact and fast rule; mtp_tf_accept.py) at
step 0 and every --eval-every steps; the best by --select-metric is kept. A hard wall-clock limit (--max-minutes)
ends the run with a final eval and the export: the best weights as the 31 mtp.* tensors (BF16, checkpoint names,
frozen ones copied byte for byte from the snapshot) to <out-dir>/mtp_trained.safetensors and a JSON report.
Checkpoints (<out-dir>/ckpt/latest.pt) every --ckpt-every steps and at the end; --resume continues from one.

    python3 mtp_train.py --capture-root /data/mtp/capture --snapshot /hf --out-dir /data/mtp/train-1 --max-minutes 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import mtp_torch as mt  # noqa: E402
from mtp_tf_accept import rules_of, torch_accept  # noqa: E402


# ---- data ----
class Pool:
    """Every window of the training sets with its sampling weight."""

    def __init__(self, capture_root, train_set, general_set, agent_frac, zeroclaw_boost, window_rows, min_rows=8, include_ids=None):
        self.entries = []  # (shard path, window index, rows, source, kind)
        self.shards = {}
        for kind, name in (("agent", train_set), ("general", general_set)):
            if not name:
                continue
            for path in mt.list_shards(capture_root, name):
                sh = mt.CaptureShard(path)
                self.shards[path] = sh
                for i in range(len(sh)):
                    n = min(sh.rows(i), window_rows)
                    if kind == "agent" and include_ids is not None and sh.seq_ids[i] not in include_ids:
                        continue
                    if n >= min_rows:
                        self.entries.append((path, i, n, sh.sources[i], kind))
        if not self.entries:
            raise FileNotFoundError(f"no capture windows under {capture_root}/{train_set}")
        rows = np.array([e[2] for e in self.entries], dtype=np.float64)
        boost = np.array([zeroclaw_boost if "zeroclaw" in e[3].lower() else 1.0 for e in self.entries])
        kind = np.array([e[4] for e in self.entries])
        w = rows * boost
        p = np.zeros_like(w)
        has_general = bool((kind == "general").any())
        for k, frac in (("agent", agent_frac if has_general else 1.0), ("general", 1.0 - agent_frac)):
            m = kind == k
            if m.any():
                p[m] = frac * w[m] / w[m].sum()
        self.p = p / p.sum()
        self.window_rows = window_rows
        self.total_rows = int(rows.sum())
        self.summary = {"windows": len(self.entries), "rows": self.total_rows,
                        "agent_windows": int((kind == "agent").sum()), "general_windows": int((kind == "general").sum()),
                        "zeroclaw_windows": int((boost > 1).sum()), "general_present": has_general}

    def batch(self, seed: int, step: int, tokens_per_step: int) -> list[int]:
        rng = np.random.default_rng([seed, step])
        out, rows = [], 0
        while rows < tokens_per_step:
            i = int(rng.choice(len(self.entries), p=self.p))
            out.append(i)
            rows += self.entries[i][2]
        return out

    def window(self, i: int) -> mt.Window:
        path, w, _, _, _ = self.entries[i]
        return self.shards[path].window(w, self.window_rows)


class LazyWindows:
    """The first n held-out windows, read from their shards on every pass (64 resident windows would be ~5 GB)."""

    def __init__(self, capture_root, heldout_set, n, window_rows):
        self.refs, self.window_rows = [], window_rows
        for path in mt.list_shards(capture_root, heldout_set):
            sh = mt.CaptureShard(path)
            for i in range(len(sh)):
                if len(self.refs) >= n:
                    return
                if sh.rows(i) >= 8:
                    self.refs.append((sh, i))

    def __len__(self):
        return len(self.refs)

    def __iter__(self):
        for sh, i in self.refs:
            yield sh.window(i, self.window_rows)


def eval_windows(capture_root, heldout_set, n, window_rows):
    return LazyWindows(capture_root, heldout_set, n, window_rows) if n else []


# ---- loss ----
def _ce_kl(x, lm_head, tgt, ids8, lg8, cd):
    logits = F.linear(x.to(cd), lm_head.to(cd)).float()
    ce = F.cross_entropy(logits, tgt, reduction="sum")
    lq = torch.log_softmax(logits.gather(1, ids8), dim=-1)
    lp = torch.log_softmax(lg8.float(), dim=-1)
    kl = (lp.exp() * (lp - lq)).sum()
    return ce, kl


def window_loss(head: mt.MtpHeadTorch, win: mt.Window, depth: int, weights, kl_weight: float, denom, device, chunk: int = 1024):
    """Sum over the window's valid rows of w_j (CE_j + kl_weight KL_j) / denom[j]; returns (loss, sums per depth)."""
    tokens, hidden, pos = win.head_inputs(device)
    tids, tlog, valid = win.targets(depth, device)
    cd = head.compute_dtype
    out = head.chain_all(tokens, hidden.to(cd if cd == torch.float32 else torch.bfloat16), pos, depth, feed="teacher", draft=False)
    loss = torch.zeros((), device=device)
    sums = []
    for j in range(depth):
        sel = valid[:, j].nonzero().flatten()
        if not sel.numel() or not denom[j]:
            sums.append((0.0, 0.0, 0))
            continue
        x = head.head_input(out["d"][j][sel])
        ce_t = kl_t = 0.0
        step_loss = torch.zeros((), device=device)
        for a in range(0, sel.numel(), chunk):
            s = slice(a, a + chunk)
            args = (x[s], head.lm_head, tids[sel[s], j, 0], tids[sel[s], j], tlog[sel[s], j], cd)
            ce, kl = checkpoint(_ce_kl, *args, use_reentrant=False) if torch.is_grad_enabled() else _ce_kl(*args)
            step_loss = step_loss + ce + kl_weight * kl
            ce_t += float(ce.detach())
            kl_t += float(kl.detach())
        loss = loss + weights[j] * step_loss / denom[j]
        sums.append((ce_t, kl_t, int(sel.numel())))
    return loss, sums


def lr_at(step: int, lr: float, warmup: int, horizon: int, min_frac: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, horizon - warmup))
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


# ---- eval / select ----
def evaluate(head, windows, depth, rules, mode):
    prev = head.mode
    head.mode = mode
    head.refresh_fp8()
    was = head.training
    head.eval()
    try:
        st = torch_accept(head, windows, depth, rules)
        summ = st.summary(bootstrap=0)
    finally:
        head.mode = prev
        head.train(was)
    return summ


def metric_of(summary: dict, name: str) -> float:
    """`<rule>_tpr<K>` (tokens per round at depth K) or `<rule>_acc<j>` (prefix acceptance at depth j), on `all`."""
    a = summary.get("all", {})
    rule, _, what = name.rpartition("_")
    if rule not in a:
        return float("-inf")
    if what.startswith("tpr"):
        return float(a[rule]["tokens_per_round"][int(what[3:]) - 1])
    if what.startswith("acc"):
        return float(a[rule]["prefix_accept"][int(what[3:]) - 1])
    raise ValueError(f"unknown metric {name}")


# ---- checkpoints ----
def state_of(head) -> dict:
    return {n: head.param(n).detach().float().cpu().clone() for n in mt.TRAINABLE}


def load_state(head, state: dict) -> None:
    with torch.no_grad():
        for n in mt.TRAINABLE:
            head.param(n).copy_(state[n].to(head.param(n).device, head.param(n).dtype))
    head.refresh_fp8()


def save_ckpt(path: Path, **obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def export(head, snap_root, state: dict, out: Path, meta: dict) -> dict:
    """The 31 mtp.* tensors: trained BF16 from `state`, frozen ones byte-copied from the snapshot."""
    snap = mt.Snapshot(snap_root)
    entries = {}
    for n in mt.EXPECTED:
        if n in mt.TRAINABLE:
            entries[n] = state[n].to(torch.bfloat16).contiguous()
        else:
            entries[n] = mt.raw_entry(snap, n)
    mt.write_safetensors(out, entries, metadata={k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()})
    return {"path": str(out), "tensors": len(entries), "frozen_copied": [n for n in mt.EXPECTED if n not in mt.TRAINABLE]}


# ---- main ----
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-weights", help="start from a trained mtp.* safetensors instead of the snapshot's head")
    ap.add_argument("--train-set", default="train")
    ap.add_argument("--general-set", default="general", help="'' to train on the agent set only")
    ap.add_argument("--heldout-set", default="heldout")
    ap.add_argument("--agent-frac", type=float, default=0.7)
    ap.add_argument("--include-ids", help="file with one train seq_id per line: only these agent windows are used")
    ap.add_argument("--zeroclaw-boost", type=float, default=5.0)
    ap.add_argument("--window-rows", type=int, default=4096)
    ap.add_argument("--tokens-per-step", type=int, default=8192, help="rows per optimizer step (whole windows, accumulated)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help="cap on optimizer steps (0: epochs decide)")
    ap.add_argument("--max-minutes", type=float, default=300.0, help="hard wall-clock limit, export included")
    ap.add_argument("--reserve-minutes", type=float, default=8.0, help="kept for the final eval and export")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--betas", default="0.9,0.95")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--depth-weights", default="1,0.8,0.6,0.5")
    ap.add_argument("--kl-weight", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--eval-mode", choices=("fp8", "float"), default="fp8")
    ap.add_argument("--eval-act-scale", choices=("row", "tensor"), default="row", help="fp8 eval: activation scale per row or per call")
    ap.add_argument("--fast-rule", default="3:2.0")
    ap.add_argument("--extra-rule", action="append", default=["2:1.0"])
    ap.add_argument("--select-metric", default="fast_tpr4", help="<rule>_tpr<K> or <rule>_acc<j> on the held-out set")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/ckpt/latest.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compute-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--expert-storage", choices=("bf16", "fp8"), default="bf16", help="fp8: E4M3 experts dequantized per call (-2.6 GB)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="keep every step's activations (more memory)")
    ap.add_argument("--attn-chunk", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=10)
    # test knobs
    ap.add_argument("--expert-range", help="a:b: load only these experts (CPU tests)")
    ap.add_argument("--vocab-rows", type=int, help="load only the first N vocabulary rows (CPU tests)")
    ap.add_argument("--overfit-one-batch", action="store_true", help="every step reuses step 0's windows")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    t_start = time.time()
    deadline = t_start + 60 * args.max_minutes
    out_dir = Path(args.out_dir)
    if args.threads:
        torch.set_num_threads(args.threads)
    rules = rules_of(args.fast_rule, args.extra_rule)
    weights = [float(x) for x in args.depth_weights.split(",")]
    if len(weights) != args.depth:
        raise ValueError("--depth-weights needs one weight per depth")
    try:
        inc = set(open(args.include_ids).read().split()) if args.include_ids else None
        pool = Pool(args.capture_root, args.train_set, args.general_set, args.agent_frac, args.zeroclaw_boost, args.window_rows, include_ids=inc)
    except FileNotFoundError as e:
        if not args.dry_run:
            raise
        print(json.dumps({"dry_run": True, "ok": False, "error": str(e), "snapshot": str(mt.Snapshot(args.snapshot).dir),
                          "load_check": mt.check_names(mt.Snapshot(args.snapshot), mt.load_overlay(args.init_weights))}, indent=1))
        return 1
    heldout = eval_windows(args.capture_root, args.heldout_set, args.eval_windows, args.window_rows)
    steps_per_epoch = max(1, math.ceil(pool.total_rows / args.tokens_per_step))
    total_steps = int(math.ceil(args.epochs * steps_per_epoch))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.dry_run:
        snap = mt.Snapshot(args.snapshot)
        print(json.dumps({"snapshot": str(snap.dir), "load_check": mt.check_names(snap, mt.load_overlay(args.init_weights)),
                          "pool": pool.summary, "heldout_windows": len(heldout), "steps_per_epoch": steps_per_epoch,
                          "total_steps": total_steps, "out_dir": str(out_dir), "device": args.device,
                          "cuda": torch.cuda.is_available()}, indent=1))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    er = tuple(int(x) for x in args.expert_range.split(":")) if args.expert_range else None
    cd = torch.float32 if args.compute_dtype == "fp32" else torch.bfloat16
    head = mt.MtpHeadTorch.from_snapshot(args.snapshot, device=dev, mode="float", compute_dtype=cd, overlay=args.init_weights,
                                         expert_range=er, vocab_rows=args.vocab_rows, expert_storage=args.expert_storage,
                                         grad_ckpt=not args.no_grad_ckpt, attn_chunk=args.attn_chunk, act_scale=args.eval_act_scale)
    head.train()
    params = head.trainable_parameters()
    n_params = sum(p.numel() for p in params)
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    b1, b2 = (float(x) for x in args.betas.split(","))
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(b1, b2), eps=1e-8)
    ckpt_path = out_dir / "ckpt" / "latest.pt"
    best_path = out_dir / "ckpt" / "best.pt"
    start_step, history, evals, horizon, elapsed_before = 0, [], [], None, 0.0
    best = {"metric": float("-inf"), "step": None}
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_state(head, ck["params"])
        opt.load_state_dict(ck["opt"])
        start_step, history, evals, horizon = ck["step"], ck["history"], ck["evals"], ck["horizon"]
        best, elapsed_before = ck["best"], ck.get("elapsed", 0.0)
        print(json.dumps({"resumed": str(ckpt_path), "step": start_step}), flush=True)
    if start_step == 0 and not evals and heldout:
        ev = evaluate(head, heldout, args.depth, rules, args.eval_mode)
        m = metric_of(ev, args.select_metric)
        evals.append({"step": 0, "metric": m, "summary": ev.get("all")})
        best = {"metric": m, "step": 0}
        save_ckpt(best_path, params=state_of(head), step=0, metric=m)
        print(json.dumps({"eval": 0, args.select_metric: m}), flush=True)
    if start_step == 0 and not best_path.exists():
        save_ckpt(best_path, params=state_of(head), step=0, metric=best["metric"])

    def checkpoint_now(step):
        save_ckpt(ckpt_path, params=state_of(head), opt=opt.state_dict(), step=step, history=history, evals=evals, horizon=horizon,
                  best=best, elapsed=elapsed_before + time.time() - t_start, args=vars(args))

    reserve = 60 * args.reserve_minutes
    step_times, oom_events = [], []
    first_batch = pool.batch(args.seed, 0, args.tokens_per_step)
    step = start_step
    stop_reason = "steps"
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        now = time.time()
        est = float(np.median(step_times)) if step_times else 0.0
        if now + est + reserve > deadline:
            stop_reason = "wall-clock"
            break
        if horizon is None and len(step_times) >= 20:
            left = (deadline - reserve - now) / max(est, 1e-6)
            horizon = int(min(total_steps, step + 0.95 * left))
        t0 = time.time()
        idx = first_batch if args.overfit_one_batch else pool.batch(args.seed, step, args.tokens_per_step)
        wins = [pool.window(i) for i in idx]
        denom = [0] * args.depth
        for w in wins:
            v = w.targets(args.depth)[2]
            for j in range(args.depth):
                denom[j] += int(v[:, j].sum())
        lr = lr_at(step, args.lr, args.warmup, horizon or total_steps, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot, per_depth = 0.0, [[0.0, 0.0, 0] for _ in range(args.depth)]
        try:
            for w in wins:
                loss, sums = window_loss(head, w, args.depth, weights, args.kl_weight, denom, dev)
                if loss.requires_grad:
                    loss.backward()
                tot += float(loss.detach())
                for j, (c, k, n) in enumerate(sums):
                    per_depth[j][0] += c
                    per_depth[j][1] += k
                    per_depth[j][2] += n
        except torch.cuda.OutOfMemoryError:
            # unattended: retry the step with windows half as long (their tails are dropped) instead of dying
            opt.zero_grad(set_to_none=True)
            del wins
            torch.cuda.empty_cache()
            if pool.window_rows <= 512:
                raise
            pool.window_rows //= 2
            oom_events.append({"step": step, "window_rows": pool.window_rows})
            print(json.dumps({"oom": step, "window_rows": pool.window_rows}), flush=True)
            continue
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip)) if args.grad_clip else float("nan")
        opt.step()
        step += 1
        dt = time.time() - t0
        step_times.append(dt)
        rec = {"step": step, "loss": round(tot, 6), "lr": lr, "gnorm": round(gnorm, 4), "rows": sum(w.tokens.numel() for w in wins),
               "s": round(dt, 3), "ce": [round(c / n, 4) if n else None for c, _, n in per_depth],
               "kl": [round(k / n, 4) if n else None for _, k, n in per_depth]}
        history.append(rec)
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(rec), flush=True)
        if heldout and args.eval_every and step % args.eval_every == 0:
            ev = evaluate(head, heldout, args.depth, rules, args.eval_mode)
            m = metric_of(ev, args.select_metric)
            evals.append({"step": step, "metric": m, "summary": ev.get("all")})
            print(json.dumps({"eval": step, args.select_metric: m}), flush=True)
            if m > best["metric"]:
                best = {"metric": m, "step": step}
                save_ckpt(best_path, params=state_of(head), step=step, metric=m)
        if args.ckpt_every and step % args.ckpt_every == 0:
            checkpoint_now(step)
    checkpoint_now(step)
    # final eval of the last weights when the run ended between evals
    if heldout and (not evals or evals[-1]["step"] != step):
        ev = evaluate(head, heldout, args.depth, rules, args.eval_mode)
        m = metric_of(ev, args.select_metric)
        evals.append({"step": step, "metric": m, "summary": ev.get("all")})
        if m > best["metric"]:
            best = {"metric": m, "step": step}
            save_ckpt(best_path, params=state_of(head), step=step, metric=m)
        checkpoint_now(step)
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)["params"] if best_path.exists() else state_of(head)
    if not heldout:
        best_state, best = state_of(head), {"metric": None, "step": step}
    meta = {"source": "mtp_train.py", "best_step": best["step"], "metric": args.select_metric}
    exported = export(head, args.snapshot, best_state, out_dir / "mtp_trained.safetensors", meta)
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None
    report = {"stop_reason": stop_reason, "steps": step, "total_steps": total_steps, "horizon": horizon, "steps_per_epoch": steps_per_epoch,
              "trainable_params": n_params, "pool": pool.summary, "heldout_windows": len(heldout), "best": best, "evals": evals,
              "baseline": evals[0] if evals else None, "improved": bool(best["step"]), "exported": exported,
              "median_step_s": float(np.median(step_times)) if step_times else None,
              "peak_gpu_gib": round(peak, 2) if peak else None, "oom_events": oom_events, "final_window_rows": pool.window_rows, "elapsed_s": round(elapsed_before + time.time() - t_start, 1),
              "args": vars(args), "load_report": head.load_report, "history_tail": history[-20:]}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history))
    print(json.dumps({k: report[k] for k in ("stop_reason", "steps", "best", "improved", "peak_gpu_gib", "elapsed_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
