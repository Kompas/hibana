# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced draft acceptance of the Flash-Next MTP head over capture sets, at depth 1..K, per rule.

A chain starts at every assistant row of a window (past its context rows): step 1 reads (x_i, h_{i-1}); step j's token
is the data token x_{i+j-1} (teacher forcing) and its hidden the head's own d of step j-1. Draft j is checked against
the target's top-8 at row i+j-1 (`topk_ids` / `topk_logits`, the capture's `fast`-profile target) under
- `exact`: the target's argmax (greedy verification),
- `fast` (`--fast-rule`, default 3:2.0 = HIBANA_ACCEPT_DEFAULT of the fast profile with MTP): the target ranks the
  draft top-K and within M logits of its top (engine/accept.accept_block), and any `--extra-rule`.
Only chains whose K steps are all assistant rows inside the window count. Reported per depth j: P(first j drafts
accepted) with a 95% bootstrap CI over sequences, the conditional acceptance, and expected tokens per round at depth
j (1 + the sum of the prefix acceptances), overall and per source.

Heads: `--impl hibana` (Hibana's FlashNextMtpHead with its FP8 kernels, on CUDA; `--draft-head w2` drafts through the
served 2-bit shortlist head) or `--impl torch` (MtpHeadTorch, `--mode fp8|float`), on the snapshot's weights or a
trained `--weights` file (mtp_train.py's output).

    python3 mtp_tf_accept.py --capture-root /data/mtp/capture --set heldout --set general --impl torch --mode fp8 \
        --snapshot /hf --weights out/mtp_trained.safetensors --out tf.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import mtp_torch as mt  # noqa: E402


def rules_of(fast_rule: str, extra: list[str]) -> dict:
    rules = {"exact": None, "fast": mt.parse_rule(fast_rule)}
    for r in extra or []:
        if r != fast_rule:
            rules[f"relax_{r}"] = mt.parse_rule(r)
    return rules


@torch.no_grad()
def torch_accept(head: mt.MtpHeadTorch, windows, depth: int, rules: dict, feed: str = "teacher", device=None,
                 stats: mt.AcceptStats | None = None, set_name: str = "", draft_head=None) -> mt.AcceptStats:
    """Every head row's chain in one batched K-pass per window (chain_all)."""
    stats = stats or mt.AcceptStats(depth, rules)
    device = device or next(iter(head.buffers())).device
    hd = torch.float32 if head.mode == "float" and head.compute_dtype == torch.float32 else torch.bfloat16
    for win in windows:
        tokens, hidden, pos = win.head_inputs(device)
        tids, tlog, valid = win.targets(depth, device)
        if not bool(valid[:, :depth].all(-1).any()):
            continue
        out = head.chain_all(tokens, hidden.to(hd), pos, depth, feed=feed, draft_head=draft_head)
        drafts = torch.stack(out["draft"], dim=1)
        stats.add(f"{set_name}:{win.seq_id}", win.source, drafts, tids, tlog, valid)
    return stats


def hibana_accept(hib, windows, depth: int, rules: dict, feed: str, chains_per_window: int, batch: int, device,
                  stats_full: mt.AcceptStats, stats_served: mt.AcceptStats | None, set_name: str = ""):
    from mtp_equiv import pick_starts
    for win in windows:
        tokens, hidden, pos = win.head_inputs(device)
        tids, tlog, valid = win.targets(depth, device)
        full = valid[:, :depth].all(-1)
        starts = pick_starts(full, chains_per_window)
        if not starts.numel():
            continue
        hib.window(tokens, hidden, pos)
        dr, sv = [], []
        for a in range(0, starts.numel(), batch):
            o = hib.chains(tokens, hidden, pos, starts[a:a + batch], depth, feed)
            dr.append(torch.stack(o["draft"], 1))
            if o["served"][0] is not None:
                sv.append(torch.stack(o["served"], 1))
        key = f"{set_name}:{win.seq_id}"
        stats_full.add(key, win.source, torch.cat(dr), tids[starts], tlog[starts], valid[starts])
        if stats_served is not None and sv:
            stats_served.add(key, win.source, torch.cat(sv), tids[starts], tlog[starts], valid[starts])


def iter_windows(capture_root, set_name, max_windows, window_rows):
    n = 0
    for path in mt.list_shards(capture_root, set_name):
        shard = mt.CaptureShard(path)
        for i in range(len(shard)):
            yield shard.window(i, window_rows)
            n += 1
            if max_windows and n >= max_windows:
                return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--set", action="append", dest="sets", help="capture set(s), repeatable (default heldout)")
    ap.add_argument("--impl", choices=("torch", "hibana"), default="torch")
    ap.add_argument("--mode", choices=("fp8", "float"), default="fp8", help="MtpHeadTorch mode (--impl torch)")
    ap.add_argument("--snapshot", required=True, help="HF root or snapshot dir")
    ap.add_argument("--weights", help="trained mtp.* safetensors (default: the snapshot's head)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fast-rule", default="3:2.0")
    ap.add_argument("--extra-rule", action="append", default=["2:1.0"])
    ap.add_argument("--feed", choices=("teacher", "argmax"), default="teacher")
    ap.add_argument("--max-windows", type=int, default=0, help="per set; 0 = all")
    ap.add_argument("--window-rows", type=int, default=4096)
    ap.add_argument("--draft-head", default="w2", choices=("fp8", "w3", "w2"), help="--impl hibana: also report the served head")
    ap.add_argument("--chains-per-window", type=int, default=0, help="--impl hibana: 0 = every eligible chain")
    ap.add_argument("--batch", type=int, default=64, help="--impl hibana: chains per forward")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--expert-range", help="a:b, CPU tests only")
    ap.add_argument("--vocab-rows", type=int, help="CPU tests only")
    ap.add_argument("--compute-dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--act-scale", default="row", choices=("row", "tensor"),
                    help="--impl torch --mode fp8: FP8 activation scale per row (a serving draft step: one row a request) or per "
                         "call over the whole window (prefill-like)")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    sets = args.sets or ["heldout"]
    rules = rules_of(args.fast_rule, args.extra_rule)
    if args.dry_run:
        from mtp_equiv import dry_run
        return dry_run(args, sets, need_hibana=args.impl == "hibana")
    t0 = time.time()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"impl": args.impl, "mode": args.mode if args.impl == "torch" else "hibana-fp8", "act_scale": args.act_scale,
              "weights": args.weights or "snapshot",
              "depth": args.depth, "feed": args.feed, "rules": rules, "sets": {}}
    if args.impl == "torch":
        er = tuple(int(x) for x in args.expert_range.split(":")) if args.expert_range else None
        head = mt.MtpHeadTorch.from_snapshot(args.snapshot, device=dev, mode=args.mode, overlay=args.weights, expert_range=er,
                                             vocab_rows=args.vocab_rows, act_scale=args.act_scale,
                                             compute_dtype=torch.float32 if args.compute_dtype == "fp32" else torch.bfloat16)
        head.eval()
        for s in sets:
            st = torch_accept(head, iter_windows(args.capture_root, s, args.max_windows, args.window_rows), args.depth, rules,
                              args.feed, dev, set_name=s)
            result["sets"][s] = st.summary(bootstrap=args.bootstrap)
    else:
        from mtp_equiv import HibanaHead
        hib = HibanaHead(args.snapshot, dev, args.window_rows, args.batch, args.depth, overlay=args.weights, draft_head=args.draft_head)
        for s in sets:
            full = mt.AcceptStats(args.depth, rules)
            served = mt.AcceptStats(args.depth, rules) if args.draft_head != "fp8" else None
            with torch.inference_mode():
                hibana_accept(hib, iter_windows(args.capture_root, s, args.max_windows, args.window_rows), args.depth, rules, args.feed,
                              args.chains_per_window, args.batch, dev, full, served, s)
            result["sets"][s] = full.summary(bootstrap=args.bootstrap)
            if served is not None:
                result["sets"][f"{s}@served-{args.draft_head}"] = served.summary(bootstrap=args.bootstrap)
    result["elapsed_s"] = round(time.time() - t0, 1)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
