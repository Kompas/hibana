#!/usr/bin/env python3
"""Capture Flash-Next's hidden stream and top-8 next-token logits over prepared token windows (MTP-head training data).

For each sequence {"id", "session", "source", "model", "tokens", "assist", "window": [a, b]} the target, built in
process as `hibana serve flashnext --<profile>` builds it (the same resolution as tools/lossy_gate.py's in-process
arm), prefills tokens[:b] (the true context) and keeps rows a .. b-1: per row i the hyper stream `forward()["norm"]`
after token i and the full lm_head's top-8 logits for token i + 1. The context tokens[:a] prefill in the engine's
solo chunk (8192); the window rows in `--window-chunk` rows per forward (default 64: the decode path, where the 3-bit
trunk and head copies serve, as for the rows a served request generates; routing then is the full top-10, not the
decode budget). Rows go to capfmt shards under <out>/<set>/ (capfmt.py documents the format).

The run is resumable (finished seq_ids are skipped), stops cleanly at --max-minutes or --max-bytes or on SIGTERM
(the current sequence finishes and the buffer is written), and never prints token text: logs carry ids and counts.

    python mtp_capture.py --set train --seqs /mtp/data/seqs-train.jsonl --out /data/mtp/capture \\
        --hf-root /hf --w3-checkpoint /data/w3-checkpoints/flashnext-edp-recal --max-minutes 120 --max-bytes 200e9
    python mtp_capture.py --set general --general /data/step3/inputs.json --out /data/mtp/capture ...
    python mtp_capture.py ... --dry-run     # resolve settings, paths and inputs, estimate bytes and time (no GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capfmt  # noqa: E402

TOPK = capfmt.TOPK
MAX_WINDOW = 4096
MAX_TOKENS = 32768


def hibana_repo() -> Path:
    """The Hibana checkout to import: $HIBANA_REPO, this file's checkout, or the image's /opt/hibana."""
    for cand in (os.environ.get("HIBANA_REPO"), HERE.parents[2] if len(HERE.parents) > 2 else None, "/opt/hibana"):
        if cand and (Path(cand) / "hibana" / "__init__.py").is_file() and (Path(cand) / "tools" / "lossy_gate.py").is_file():
            return Path(cand)
    raise SystemExit("no Hibana checkout found (set HIBANA_REPO)")


def use_repo() -> Path:
    repo = hibana_repo()
    for p in (str(repo / "tools"), str(repo)):
        if p not in sys.path:
            sys.path.insert(1, p)
    return repo


def env_checkpoint() -> str | None:
    """$HIBANA_W3_CHECKPOINT when it names a directory (the recipe's compose sets it, `auto` by default)."""
    v = os.environ.get("HIBANA_W3_CHECKPOINT", "")
    return v if v and v != "auto" else None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- the arm (settings as the gate resolves them)

def parse_env(items) -> dict:
    """--env KEY=VALUE overrides (HIBANA_* settings, as tools/lossy_gate.py --env)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.startswith("HIBANA_"):
            raise SystemExit(f"--env takes HIBANA_KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def arm(profile: str, hf_root: str | None, w3_checkpoint: str | None, target: str = "flashnext", extra: dict | None = None) -> dict:
    """The engine environment and executor settings of `hibana serve <target> --<profile>` with these overrides
    (tools/lossy_gate.py resolve + arm_plan: the config file, the target's settings, the profile's defaults)."""
    use_repo()
    import lossy_gate
    overrides = dict(extra or {})
    if hf_root:
        overrides["HIBANA_HF_CACHE"] = overrides["HIBANA_FLASHNEXT_CACHE"] = str(hf_root)
    if w3_checkpoint:
        overrides["HIBANA_W3_CHECKPOINT"] = str(w3_checkpoint)
    s = lossy_gate.resolve(target, profile, overrides)
    plan = lossy_gate.arm_plan(s)
    plan["env"].update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "HIBANA_SSD": "0", "HIBANA_TRACES": "off"})
    plan["overrides"] = overrides
    plan["profile"] = profile
    return plan


def apply_env(plan: dict) -> None:
    """Install the arm's environment in this process (before torch or the engine config is imported)."""
    os.environ.update(plan["env"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def check_snapshot(plan: dict) -> dict:
    """The pinned snapshot under the arm's HF root, its mtp.* tensors, and the W3 checkpoint's manifest check (the
    same `CalibratedWeights` validation the model load runs). CPU only."""
    from hibana import target as targets
    from hibana.engine import config
    config.install(config.EngineConfig.from_env())
    t = targets.get(plan["target"])
    snap = t.snapshot_dir()
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    mtp = {k: v for k, v in index.items() if k.startswith("mtp.")}
    missing = sorted({v for v in index.values()} - {p.name for p in snap.iterdir()})
    out = {"snapshot": str(snap), "index_entries": len(index), "mtp_tensors": len(mtp), "mtp_files": sorted(set(mtp.values())),
           "missing_files": missing[:5], "w3_checkpoint": plan["env"].get("HIBANA_W3_CHECKPOINT")}
    if out["w3_checkpoint"]:
        from hibana.weights.w3_checkpoint import CalibratedWeights
        cw = CalibratedWeights(out["w3_checkpoint"], revision=t.model_revision)
        out["w3_weights"] = len(cw.weights)
        out["w3_identity"] = cw.identity[:16]
    if missing:
        raise SystemExit(f"snapshot {snap} lacks files its index names: {missing[:3]}")
    return out


# ---------------------------------------------------------------- inputs

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{n}: not JSON ({error.msg})") from None
    return out


def check_sequence(seq: dict, max_window: int = MAX_WINDOW, max_tokens: int = MAX_TOKENS) -> str | None:
    """Why a sequence cannot be captured, or None."""
    toks, win = seq.get("tokens"), seq.get("window")
    if not isinstance(seq.get("id"), (str, int)):
        return "no id"
    if not isinstance(toks, list) or not toks or not all(isinstance(t, int) and t >= 0 for t in toks):
        return "tokens"
    if not (isinstance(win, list) and len(win) == 2 and all(isinstance(x, int) for x in win)):
        return "window"
    a, b = win
    if not 0 <= a < b <= len(toks):
        return f"window [{a}, {b}] outside {len(toks)} tokens"
    if b - a > max_window:
        return f"window of {b - a} rows > {max_window}"
    if b > max_tokens:
        return f"context of {b} tokens > {max_tokens}"
    for span in seq.get("assist") or []:
        if not (isinstance(span, list) and len(span) == 2 and 0 <= span[0] <= span[1] <= len(toks)):
            return "assist span"
    return None


def assist_mask(spans, a: int, b: int, total: int) -> np.ndarray:
    """Row i of the window (token a + i) -> 1 when token a + i + 1 is assistant-produced. Spans are [s, e) over
    token indices."""
    mark = np.zeros(total + 1, dtype=np.uint8)
    for s, e in spans or []:
        mark[s:e] = 1
    return mark[a + 1:b + 1].copy()


def assistant_spans(ids: list[int], opener: list[int], closer: int) -> list[list[int]]:
    """[s, e) spans of assistant turns in a chat-templated sequence: after each `opener` (the tokens of
    "<|im_start|>assistant\\n") through its closing <|im_end|> (the model produces it), or to the end."""
    spans, n, k, i = [], len(ids), len(opener), 0
    while i + k <= n:
        if ids[i:i + k] == opener:
            s = i + k
            e = s
            while e < n and ids[e] != closer:
                e += 1
            e = min(n, e + 1)
            spans.append([s, e])
            i = e
        else:
            i += 1
    return spans


def general_sequences(path: Path, opener: list[int] | None, closer: int | None, window: int = MAX_WINDOW) -> list[dict]:
    """The calibration inputs (calibration_inputs build: {"samples": [{"source", "input_ids"}]}) as capture
    sequences: the last `window` tokens of each sample, assistant spans from the chat template's markers."""
    data = json.loads(Path(path).read_text())
    out = []
    for i, sample in enumerate(data["samples"]):
        ids = [int(t) for t in sample["input_ids"]]
        spans = assistant_spans(ids, opener, closer) if opener else []
        out.append({"id": f"general-{i:04d}", "session": f"general-{i:04d}", "source": "general", "model": "",
                    "tokens": ids, "assist": spans, "window": [max(0, len(ids) - window), len(ids)]})
    return out


def interleave(seqs: list[dict]) -> list[dict]:
    """Round robin over sources (file order within a source), so a cap cuts every source alike."""
    by: OrderedDict[str, list] = OrderedDict()
    for s in seqs:
        by.setdefault(str(s.get("source", "")), []).append(s)
    queues = [list(reversed(v)) for v in by.values()]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def context_rows_for(n: int, cap: int) -> int:
    """The window's first rows used as context only: `cap`, or half the window when it is shorter than 2 x cap."""
    return min(cap, n // 2)


def prepare(args, tokenizer=None) -> tuple[list[dict], dict]:
    """The set's sequences (validated, ordered) and counts."""
    if args.general:
        opener = closer = None
        if tokenizer is not None:
            opener = tokenizer.encode("<|im_start|>assistant\n")
            closer = tokenizer.encode("<|im_end|>")
            closer = closer[0] if len(closer) == 1 else None
            if closer is None:
                opener = None
        seqs = general_sequences(Path(args.general), opener, closer, args.max_window)
    else:
        seqs = load_jsonl(Path(args.seqs))
    bad: dict[str, int] = {}
    good = []
    seen = set()
    for s in seqs:
        why = check_sequence(s, args.max_window, args.max_tokens)
        if why is None and str(s["id"]) in seen:
            why = "duplicate id"
        if why:
            bad[why.split(" ")[0]] = bad.get(why.split(" ")[0], 0) + 1
            continue
        s["id"] = str(s["id"])
        seen.add(s["id"])
        good.append(s)
    if args.order == "interleave":
        good = interleave(good)
    if args.limit:
        good = good[:args.limit]
    rows = sum(s["window"][1] - s["window"][0] for s in good)
    counts = {"sequences": len(good), "rejected": bad, "window_rows": rows, "prefill_tokens": sum(s["window"][1] for s in good),
              "assist_rows": int(sum(assist_mask(s.get("assist"), *s["window"], len(s["tokens"])).sum() for s in good)),
              "sources": {}}
    for s in good:
        src = counts["sources"].setdefault(str(s.get("source", "")), {"sequences": 0, "rows": 0})
        src["sequences"] += 1
        src["rows"] += s["window"][1] - s["window"][0]
    counts["bytes"] = rows * capfmt.row_bytes()
    return good, counts


# ---------------------------------------------------------------- the engine

class EngineCapture:
    """The target in process (no drafter): prefills tokens[:b] and returns rows a..b-1's hidden stream and top-k."""

    def __init__(self, plan: dict, max_context: int, window_chunk: int, prefix_cache: bool, logf=print):
        import torch
        from hibana import target as targets
        from hibana.engine import config
        from hibana.engine.executor import Executor
        self.torch = torch
        cfg = config.EngineConfig.from_env().replace(spec="0")  # the target only: nothing drafts while capturing
        config.install(cfg)
        served = targets.get(cfg.target)
        snap = served.snapshot_dir()
        started = time.perf_counter()
        self.model = targets.load_model(served, snap, log=logf)
        self.ex = Executor(self.model, log=logf, max_context=max_context)
        self.ex.capture_graphs(log=logf)
        self.ex.warmup(log=logf)
        prof = self.ex.profile
        self.context_chunk = int(getattr(prof, "solo_chunk_size", 0) or getattr(prof, "chunk_size", 0) or served.prefill_chunk)
        self.window_chunk = window_chunk
        self.prefix_cache = prefix_cache
        self.info = {"target": served.name, "model": f"{served.model_id}@{served.model_revision}", "snapshot": str(snap),
                     "config": cfg.describe(), "w3_identity": getattr(self.model, "w3_identity", None),
                     "device": torch.cuda.get_device_name(), "load_s": round(time.perf_counter() - started, 1),
                     "context_chunk": self.context_chunk, "window_chunk": window_chunk, "prefix_cache": prefix_cache}

    def __call__(self, seq: dict) -> dict:
        from hibana.engine.sampler import SamplingParams
        torch, ex, model = self.torch, self.ex, self.model
        a, b = seq["window"]
        ids = list(seq["tokens"][:b])
        params = SamplingParams(max_new_tokens=1, trace=False)
        if not self.prefix_cache:
            ex.flush_cache()
        req = ex.admit(ids, params)
        if req.prefill_pos > a:  # the cache holds part of the window: recompute it
            req.cacheable = False
            ex.release(req)
            ex.flush_cache()
            req = ex.admit(ids, params)
        cached = req.prefill_pos
        hs, top_i, top_v = [], [], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_window = None
        try:
            while req.prefill_pos < req.prefill_len:
                pos = req.prefill_pos
                if pos < a:
                    chunk = min(self.context_chunk, a - pos)
                else:
                    if t_window is None:
                        torch.cuda.synchronize()
                        t_window = time.perf_counter()
                    chunk = self.window_chunk
                batch = ex._prepare_single_prefill(req, chunk)
                out = model.forward(batch.ids, batch.context)
                if pos >= a:
                    h = out["norm"]
                    v, i = model.logits(h).float().topk(TOPK, dim=-1)
                    hs.append(h.to(torch.bfloat16).cpu())
                    top_v.append(v.cpu())
                    top_i.append(i.to(torch.int32).cpu())
                ex._finish_single_prefill(batch, out)
            torch.cuda.synchronize()
        except BaseException:
            ex.state.discard_private(req)
            raise
        finally:
            if not self.prefix_cache:
                req.cacheable = False
            ex.release(req)
        t1 = time.perf_counter()
        h = torch.cat(hs)
        n = b - a
        if h.shape[0] != n:
            raise RuntimeError(f"captured {h.shape[0]} rows, expected {n}")
        t_window = t_window or t0
        return {"h": h.view(torch.int16).numpy().view(np.uint16), "topk_ids": torch.cat(top_i).numpy(),
                "topk_logits": torch.cat(top_v).numpy().astype(np.float32),
                "timing": {"cached": cached, "context_tokens": a - min(cached, a), "context_s": round(t_window - t0, 3),
                           "window_s": round(t1 - t_window, 3), "seconds": round(t1 - t0, 3)}}


# ---------------------------------------------------------------- the loop (engine-agnostic, tested on CPU)

def run_capture(seqs: list[dict], writer: capfmt.ShardWriter, capture, *, deadline: float, max_bytes: float,
                context_rows: int, stop=lambda: False, progress: Path | None = None, logf=log) -> dict:
    """Capture every sequence the writer has not finished, until a limit; always flushes the buffer."""
    done = skipped = 0
    reason = "complete"
    tokens = seconds = 0.0
    per_seq, agree_all = [], []
    started = time.monotonic()
    row_b = capfmt.row_bytes(writer.hidden, writer.topk)
    try:
        for k, seq in enumerate(seqs):
            if seq["id"] in writer.finished:
                skipped += 1
                continue
            if stop():
                reason = "signal"
                break
            now = time.monotonic()
            expect = 1.5 * (sum(per_seq[-20:]) / len(per_seq[-20:])) if per_seq else 0.0
            if now + expect >= deadline:
                reason = "max-minutes"
                break
            a, b = seq["window"]
            n = b - a
            if writer.total_bytes + n * row_b > max_bytes:
                reason = "max-bytes"
                break
            got = capture(seq)
            t = got["timing"]
            positions = np.arange(a, b, dtype=np.int32)
            assist = assist_mask(seq.get("assist"), a, b, len(seq["tokens"]))
            # sanity: the target's top-1 at row i against token i + 1, on assistant rows (its own outputs: high)
            nxt = np.asarray(seq["tokens"][a + 1:b + 1], dtype=np.int64)
            top1 = np.asarray(got["topk_ids"])[:len(nxt), 0]
            on = assist[:len(nxt)].astype(bool)
            t["top1_agree_assist"] = round(float((top1[on] == nxt[on]).mean()), 4) if on.any() else None
            extra = {"session": str(seq.get("session", "")), "model": str(seq.get("model", "")), **t}
            writer.add(seq["id"], str(seq.get("source", "")), context_rows_for(n, context_rows),
                       np.asarray(seq["tokens"][a:b], dtype=np.int32), positions, got["h"],
                       np.asarray(got["topk_ids"], dtype=np.int32), np.asarray(got["topk_logits"], dtype=np.float32),
                       assist, extra=extra)
            done += 1
            per_seq.append(t["seconds"])
            agree_all.append(t["top1_agree_assist"])
            tokens += b - t.get("cached", 0)
            seconds += t["seconds"]
            rec = {"i": k, "id": seq["id"], "rows": n, "context": a, **t,
                   "prefill_tps": round((b - t.get("cached", 0)) / t["seconds"], 1) if t["seconds"] else None,
                   "bytes": writer.total_bytes}
            if progress is not None:
                with open(progress, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if done % 10 == 1:
                logf(f"seq {k} ({done} new): {n} rows after {a} context tokens, {t['seconds']:.1f}s "
                     f"({rec['prefill_tps']} tok/s), top-1 = next token on assistant rows {t['top1_agree_assist']}, "
                     f"{writer.total_bytes / 2**30:.1f} GiB")
    finally:
        writer.flush()
    agree = [x for x in agree_all if x is not None]
    return {"reason": reason, "captured": done, "top1_agree_assist_mean": round(sum(agree) / len(agree), 4) if agree else None, "skipped_finished": skipped, "remaining": len(seqs) - done - skipped,
            "prefill_tokens": int(tokens), "prefill_tps": round(tokens / seconds, 1) if seconds else None,
            "seconds": round(time.monotonic() - started, 1), "bytes_on_disk": writer.disk_bytes}


# ---------------------------------------------------------------- driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seqs", help="sequences JSONL (seqs-{train,heldout}.jsonl)")
    src.add_argument("--general", help="calibration inputs JSON (calibration_inputs build), captured as set `general`")
    ap.add_argument("--set", required=True, help="capture set name (train, heldout, general)")
    ap.add_argument("--out", default="/data/mtp/capture", help="capture root: shards go to OUT/SET/")
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--hf-root", default=None, help="HF cache root holding the pinned snapshot (default: the config's)")
    ap.add_argument("--w3-checkpoint", default=env_checkpoint(), help="3-bit checkpoint directory (default: $HIBANA_W3_CHECKPOINT)")
    ap.add_argument("--env", action="append", default=[], help="HIBANA_KEY=VALUE setting override (repeatable)")
    ap.add_argument("--window-chunk", type=int, default=64, help="rows per forward over the window (64: the decode path)")
    ap.add_argument("--context-rows", type=int, default=1024, help="context-only rows at each window's start (capped at half the window)")
    ap.add_argument("--shard-bytes", type=float, default=1 << 30)
    ap.add_argument("--max-bytes", type=float, default=float("inf"), help="stop before the set exceeds this many bytes on disk")
    ap.add_argument("--max-minutes", type=float, default=float("inf"))
    ap.add_argument("--max-window", type=int, default=MAX_WINDOW)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--order", choices=["interleave", "file"], default="interleave")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-prefix-cache", action="store_true", help="flush the prefix cache before every sequence")
    ap.add_argument("--dry-run", action="store_true", help="resolve everything and estimate; no model load, no GPU")
    ap.add_argument("--assume-prefill-tps", type=float, default=4000.0, help="dry-run estimate: context prefill tok/s")
    ap.add_argument("--assume-window-rps", type=float, default=500.0, help="dry-run estimate: window rows/s at the chunk")
    args = ap.parse_args(argv)
    t_start = time.monotonic()
    deadline = t_start + 60 * args.max_minutes

    plan = arm(args.profile, args.hf_root, args.w3_checkpoint, extra=parse_env(args.env))
    apply_env(plan)
    set_dir = Path(args.out) / args.set
    tokenizer = None
    if args.general:
        from hibana import target as targets
        from hibana.engine import config
        from hibana.tokenizer import SnapshotTokenizer
        config.install(config.EngineConfig.from_env())
        tokenizer = SnapshotTokenizer(targets.get(plan["target"]).snapshot_dir())
    seqs, counts = prepare(args, tokenizer)
    if not seqs:
        raise SystemExit("no sequences to capture")
    max_context = max(s["window"][1] for s in seqs) + 64

    if args.dry_run:
        import torch  # noqa: F401 - the image's engine imports resolve
        import hibana.engine.executor  # noqa: F401
        import hibana.engine.sampler  # noqa: F401
        snap = check_snapshot(plan)
        writer = capfmt.ShardWriter(set_dir, args.set, shard_bytes=int(args.shard_bytes), context_cap=args.context_rows, log=None)
        todo = [s for s in seqs if s["id"] not in writer.finished]
        ctx = sum(s["window"][0] for s in todo)
        rows = sum(s["window"][1] - s["window"][0] for s in todo)
        st = os.statvfs(set_dir)
        est = {"todo_sequences": len(todo), "already_captured": len(seqs) - len(todo), "todo_bytes": rows * capfmt.row_bytes(),
               "todo_hours": round((ctx / args.assume_prefill_tps + rows / args.assume_window_rps) / 3600, 2),
               "assumed": {"prefill_tps": args.assume_prefill_tps, "window_rows_per_s": args.assume_window_rps},
               "disk_free_bytes": st.f_bavail * st.f_frsize}
        print(json.dumps({"dry_run": True, "set": args.set, "set_dir": str(set_dir), "profile": args.profile,
                          "arm": {k: plan[k] for k in ("target", "proposer", "spec", "accept", "overrides")},
                          "engine_env": {k: v for k, v in sorted(plan["env"].items()) if k.startswith("HIBANA_")},
                          "snapshot": snap, "inputs": counts, "max_context": max_context, "estimate": est}, indent=1))
        return 0

    writer = capfmt.ShardWriter(set_dir, args.set, shard_bytes=int(args.shard_bytes), context_cap=args.context_rows, log=log)
    todo = sum(1 for s in seqs if s["id"] not in writer.finished)
    log(f"set {args.set}: {len(seqs)} sequences, {todo} to capture, {writer.disk_bytes / 2**30:.1f} GiB already on disk")
    if not todo:
        log("nothing to do")
        return 0
    stopping = {"flag": False}

    def on_signal(signum, frame):
        stopping["flag"] = True
        log(f"signal {signum}: finishing the current sequence, then writing the buffer")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    engine = EngineCapture(plan, max_context, args.window_chunk, not args.no_prefix_cache)
    (set_dir / "arm.json").write_text(json.dumps({**engine.info, "profile": args.profile, "overrides": plan["overrides"],
                                                  "env": plan["env"]}, indent=1) + "\n")
    result = run_capture(seqs, writer, engine, deadline=deadline, max_bytes=args.max_bytes, context_rows=args.context_rows,
                         stop=lambda: stopping["flag"], progress=set_dir / "progress.jsonl")
    result.update(set=args.set, total_sequences=len(seqs), load_s=engine.info["load_s"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (set_dir / f"run-{stamp}.json").write_text(json.dumps(result, indent=1) + "\n")
    log("done: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
