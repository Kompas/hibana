#!/usr/bin/env python3
"""Speculative-decoding speed of Flash-Next's MTP head on held-out agent prompts, for one model snapshot (one arm).

The engine is built in process exactly as tools/lossy_gate.py builds its in-process arm (`hibana serve flashnext
--<profile>` settings: proposer, depth, acceptance rule, 3-bit copies), from the snapshot under `--hf-root` (a stock
or a tuned MTP head: only `mtp.*` differ) with `--w3-checkpoint`. Each prompt ({"id", "session", "source", "model",
"prompt_tokens"}, already templated) runs alone with a cold prefix cache, greedy, up to --max-new tokens, with the
profile's greedy acceptance rule (fast: relaxed top-3 within 2.0; `--accept 1:0` for exact). Per prompt: tokens
emitted, verify rounds, tokens per round, drafts proposed and accepted, relaxed acceptances, the accepted-length
histogram of its rounds and decode tok/s. Summary per source, `pi` split by model (qwen-trial117 / other).

    python agent_eval.py --prompts /mtp/data/prompts-heldout.jsonl --out /data/mtp/eval/stock.json --hf-root /hf
    python agent_eval.py --prompts ... --out /data/mtp/eval/tuned.json --hf-root /hf-tuned
    python agent_eval.py --compare /data/mtp/eval/stock.json /data/mtp/eval/tuned.json [--out cmp.json]

--compare pairs prompts by id and reports tuned/stock ratios per group with 95% bootstrap intervals over prompts
(tools/lossy_gate.py `paired_log`: the mean log ratio, resampled by prompt). Runs are resumable (OUT.partial.jsonl;
a rerun of the same arm skips finished prompts) and stop at --max-minutes. Logs and files carry no prompt text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mtp_capture import apply_env, arm, check_snapshot, env_checkpoint, load_jsonl, log, parse_env, use_repo  # noqa: E402

SPLIT_MODEL = "qwen-trial117"
SPLIT_SOURCES = ("pi",)
KEYS = ("tok_per_round", "decode_tps", "accept_rate", "round_ms")


# ---------------------------------------------------------------- grouping and statistics (CPU, tested)

def groups_of(row: dict, split_sources=SPLIT_SOURCES, split_model: str = SPLIT_MODEL) -> list[str]:
    """The summary groups a prompt counts in: all, its source, and for split sources source/model."""
    src = str(row.get("source", ""))
    out = ["all", src]
    if src in split_sources:
        out.append(f"{src}/{split_model if row.get('model') == split_model else 'other'}")
    return out


def summarize(rows: list[dict], split_sources=SPLIT_SOURCES, split_model: str = SPLIT_MODEL) -> dict:
    """Per group: pooled tokens per round ((emitted - 1) / rounds, as the gate counts), pooled acceptance and relaxed
    share, pooled and mean decode tok/s."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        for g in groups_of(r, split_sources, split_model):
            by.setdefault(g, []).append(r)
    out = {}
    for g, rs in sorted(by.items()):
        n_out = sum(r["output_tokens"] for r in rs)
        rounds = sum(r["rounds"] for r in rs)
        dec = sum(r["decode_s"] for r in rs)
        tps = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        tpr = [r["tok_per_round"] for r in rs if r.get("tok_per_round")]
        hist: dict[str, int] = {}
        for r in rs:
            for k, v in (r.get("accepted_hist") or {}).items():
                hist[k] = hist.get(k, 0) + v
        out[g] = {"prompts": len(rs), "prompt_tokens_mean": sum(r["prompt_tokens"] for r in rs) / len(rs),
                  "output_tokens": n_out, "rounds": rounds,
                  "tok_per_round": sum(r["output_tokens"] - 1 for r in rs) / rounds if rounds else None,
                  "tok_per_round_mean": sum(tpr) / len(tpr) if tpr else None,
                  "accept_rate": sum(r["accepted"] for r in rs) / max(1, sum(r["proposed"] for r in rs)),
                  "relaxed_share": sum(r["relaxed"] for r in rs) / max(1, n_out),
                  "decode_tps": sum(r["output_tokens"] - 1 for r in rs) / dec if dec else None,
                  "decode_tps_mean": sum(tps) / len(tps) if tps else None,
                  "accepted_hist": dict(sorted(hist.items(), key=lambda kv: int(kv[0])))}
    return out


def compare(base: dict, cand: dict, split_sources=SPLIT_SOURCES, split_model: str = SPLIT_MODEL) -> dict:
    """Paired tuned/stock (cand/base) ratios per group, with lossy_gate's paired bootstrap over prompts."""
    use_repo()
    from lossy_gate import paired_log
    b_rows = {r["id"]: r for r in base["prompts"]}
    c_rows = {r["id"]: r for r in cand["prompts"]}
    ids = [i for i in b_rows if i in c_rows]  # the base's order
    warnings = []
    if len(ids) < max(len(b_rows), len(c_rows)):
        warnings.append(f"{len(b_rows)} / {len(c_rows)} prompts, {len(ids)} paired")
    for key in ("prompts_sha256", "accept", "max_new", "w3_identity", "profile"):
        if base["arm"].get(key) != cand["arm"].get(key):
            warnings.append(f"arms differ in {key}: {base['arm'].get(key)} / {cand['arm'].get(key)}")
    if base["arm"].get("mtp_identity") == cand["arm"].get("mtp_identity"):
        warnings.append("both arms read the same mtp.* index entries (same head?)")
    by: dict[str, tuple[list, list]] = {}
    for i in ids:
        for g in groups_of(b_rows[i], split_sources, split_model):
            pair = by.setdefault(g, ([], []))
            pair[0].append(b_rows[i])
            pair[1].append(c_rows[i])
    out = {}
    for g, (bs, cs) in sorted(by.items()):
        row = {k: paired_log(bs, cs, k) for k in KEYS}
        sb, sc = summarize(bs, (), "")["all"], summarize(cs, (), "")["all"]
        row["pooled"] = {k: (sc[k] / sb[k] if sb.get(k) and sc.get(k) else None) for k in ("tok_per_round", "decode_tps", "accept_rate")}
        row["same_output"] = sum(b["output_sha256"] == c["output_sha256"] for b, c in zip(bs, cs))
        row["prompts"] = len(bs)
        out[g] = row
    return {"base": base.get("label"), "cand": cand.get("label"), "paired": len(ids), "warnings": warnings, "groups": out}


def fmt_ratio(r: dict) -> str:
    if not r or not r.get("n"):
        return "-"
    lo, hi = r["ci95"]
    return f"{r['ratio']:.3f} [{lo:.3f}, {hi:.3f}]"


def compare_table(cmp: dict) -> str:
    lines = [f"tuned/stock, {cmp['paired']} paired prompts (95% bootstrap by prompt)", "",
             "| group | n | tokens/round | decode tok/s | acceptance | same output |", "|---|---|---|---|---|---|"]
    for g, r in cmp["groups"].items():
        lines.append(f"| {g} | {r['prompts']} | {fmt_ratio(r['tok_per_round'])} | {fmt_ratio(r['decode_tps'])} | "
                     f"{fmt_ratio(r['accept_rate'])} | {r['same_output']} |")
    lines += [f"warning: {w}" for w in cmp["warnings"]]
    return "\n".join(lines)


def prompt_row(p: dict, r, decode_s: float, prefill_s: float, rounds_log: list[tuple[int, int]]) -> dict:
    """One prompt's measurements from a finished engine Request (no text: the output is kept as a hash)."""
    n = len(r.output_ids)
    rounds = r.spec_rounds
    hist: dict[str, int] = {}
    depths: dict[str, int] = {}
    for depth, acc in rounds_log:
        hist[str(acc)] = hist.get(str(acc), 0) + 1
        depths[str(depth)] = depths.get(str(depth), 0) + 1
    return {"id": p["id"], "source": str(p.get("source", "")), "model": str(p.get("model", "")),
            "prompt_tokens": len(p["prompt_tokens"]), "output_tokens": n, "finish": getattr(r, "finish_reason", None),
            "rounds": rounds, "proposed": r.spec_proposed, "accepted": r.spec_accepted, "emitted": r.spec_emitted,
            "relaxed": r.spec_relaxed, "tok_per_round": (n - 1) / rounds if rounds else None,
            "accept_rate": r.spec_accepted / r.spec_proposed if r.spec_proposed else None,
            "relaxed_share": r.spec_relaxed / n if n else None,
            "prefill_s": round(prefill_s, 4), "decode_s": round(decode_s, 4),
            "decode_tps": (n - 1) / decode_s if n > 1 and decode_s > 0 else None,
            "round_ms": 1000 * decode_s / rounds if rounds else None,
            "accepted_hist": hist, "depth_hist": depths, "observed_rounds": len(rounds_log),
            "output_sha256": hashlib.sha256(json.dumps(list(r.output_ids)).encode()).hexdigest()[:16]}


# ---------------------------------------------------------------- the arm's identity

def mtp_identity(snapshot: Path) -> str:
    """A hash of the index's mtp.* entries and the size and mtime of their files: tells a tuned head from the stock."""
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    mtp = sorted((k, v) for k, v in index.items() if k.startswith("mtp."))
    files = sorted({v for _, v in mtp})
    stats = [(f, (snapshot / f).stat().st_size, int((snapshot / f).stat().st_mtime)) for f in files]
    return hashlib.sha256(json.dumps([mtp, stats]).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the engine

class Runner:
    def __init__(self, plan: dict, max_context: int):
        import torch
        use_repo()
        from gate.lossy_gate import load  # the gate's in-process arm: model, executor (spec, graphs, warmup), tokenizer
        self.torch = torch
        ns = argparse.Namespace(proposer=plan["proposer"], spec=plan["spec"], verify_state=plan["verify_state"],
                                draft_precision=plan["draft_precision"], draft=plan["draft"], max_context=max_context)
        self.model, self.ex, _, self.info = load(ns)
        self.chunk = int(getattr(self.ex.profile, "solo_chunk_size", 0) or getattr(self.ex.profile, "chunk_size", 0) or 8192)
        self.rounds: list[tuple[int, int]] = []
        policy = self.ex.spec.depth_policy
        original = getattr(policy, "observe", None)

        def observe(request, depth, accepted, *rest, **kw):  # every verify round, host or device acceptance path
            self.rounds.append((int(depth), int(accepted)))
            if original is not None:
                return original(request, depth, accepted, *rest, **kw)
        policy.observe = observe

    def run(self, ids: list[int], params):
        """One request alone: cold cache, prefill in the engine's solo chunk, timed decode (as generate_batch)."""
        torch, ex = self.torch, self.ex
        ex.flush_cache()
        ex.spec.reset_stats()
        self.rounds = []
        ex.validate_request(ids, params)
        r = ex.admit(ids, params)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while r.prefill_pos < r.prefill_len:
            ex.prefill(r, self.chunk)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        while not r.finished:
            ex.decode_step([r])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ex.release(r)
        return r, t2 - t1, t1 - t0, list(self.rounds)


# ---------------------------------------------------------------- driver

def accept_tuple(text: str) -> tuple[int, float]:
    k, m = text.split(":")
    return int(k), float(m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", help="prompts JSONL (prompts-heldout.jsonl)")
    ap.add_argument("--out", help="this arm's JSON (or the comparison's, with --compare)")
    ap.add_argument("--compare", nargs=2, metavar=("STOCK", "TUNED"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--hf-root", default=None, help="HF cache root holding the pinned snapshot (stock or tuned head)")
    ap.add_argument("--w3-checkpoint", default=env_checkpoint())
    ap.add_argument("--env", action="append", default=[], help="HIBANA_KEY=VALUE setting override (repeatable)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--accept", default=None, help="greedy K:M acceptance (default: the profile's; 1:0 exact)")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=float("inf"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split-sources", default=",".join(SPLIT_SOURCES))
    ap.add_argument("--split-model", default=SPLIT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="resolve settings, paths and prompts; no model load, no GPU")
    args = ap.parse_args(argv)
    split = tuple(s for s in args.split_sources.split(",") if s)

    if args.compare:
        base, cand = (json.loads(Path(p).read_text()) for p in args.compare)
        cmp = compare(base, cand, split, args.split_model)
        if args.out:
            Path(args.out).write_text(json.dumps(cmp, indent=1) + "\n")
        print(compare_table(cmp))
        return 0
    if not args.prompts or not args.out:
        ap.error("--prompts and --out are required (or --compare)")

    deadline = time.monotonic() + 60 * args.max_minutes
    plan = arm(args.profile, args.hf_root, args.w3_checkpoint, extra=parse_env(args.env))
    apply_env(plan)
    accept = accept_tuple(args.accept or plan["accept"])
    prompts_path = Path(args.prompts)
    prompts = load_jsonl(prompts_path)
    bad = [p for p in prompts if not (isinstance(p.get("prompt_tokens"), list) and p["prompt_tokens"] and "id" in p)]
    if bad:
        raise SystemExit(f"{len(bad)} prompts lack id or prompt_tokens")
    for p in prompts:
        p["id"] = str(p["id"])
    if len({p["id"] for p in prompts}) != len(prompts):
        raise SystemExit("duplicate prompt ids")
    if args.limit:
        prompts = prompts[:args.limit]
    max_context = max(len(p["prompt_tokens"]) for p in prompts) + args.max_new + 64
    snap = check_snapshot(plan)
    arm_info = {"label": args.label or Path(args.out).stem, "profile": args.profile, "hf_root": args.hf_root,
                "w3_checkpoint": plan["env"].get("HIBANA_W3_CHECKPOINT"), "w3_identity": snap.get("w3_identity"),
                "snapshot": snap["snapshot"], "mtp_identity": mtp_identity(Path(snap["snapshot"])),
                "proposer": plan["proposer"], "spec": plan["spec"], "accept": list(accept), "max_new": args.max_new,
                "ignore_eos": args.ignore_eos, "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest()[:16]}
    groups: dict[str, int] = {}
    for p in prompts:
        for g in groups_of(p, split, args.split_model):
            groups[g] = groups.get(g, 0) + 1

    if args.dry_run:
        import torch  # noqa: F401
        import hibana.engine.executor  # noqa: F401
        from hibana.engine.sampler import SamplingParams
        SamplingParams(max_new_tokens=args.max_new, accept_top_k=accept[0], accept_margin=accept[1], trace=False)
        use_repo()
        import gate.lossy_gate  # noqa: F401
        import lossy_gate  # noqa: F401
        tokens = sum(len(p["prompt_tokens"]) for p in prompts)
        print(json.dumps({"dry_run": True, "arm": arm_info, "engine_env": {k: v for k, v in sorted(plan["env"].items()) if k.startswith("HIBANA_")},
                          "snapshot": snap, "prompts": len(prompts), "prompt_tokens": tokens,
                          "prompt_tokens_max": max(len(p["prompt_tokens"]) for p in prompts), "groups": groups,
                          "max_context": max_context,
                          "estimate_minutes_per_arm": round((90 + tokens / 4000 + len(prompts) * args.max_new / 110) / 60, 1)}, indent=1))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial.jsonl")
    arm_info["env_overrides"] = plan["overrides"]
    key = {"env": plan["overrides"], **{k: arm_info[k] for k in ("hf_root", "w3_checkpoint", "mtp_identity", "accept", "max_new", "ignore_eos", "profile", "prompts_sha256")}}
    rows: list[dict] = []
    if partial.exists():
        lines = [json.loads(x) for x in partial.read_text().splitlines() if x.strip()]
        if lines and lines[0].get("arm_key") != key:
            raise SystemExit(f"{partial} belongs to another arm; move it away")
        rows = [x for x in lines[1:] if "id" in x]
    else:
        partial.write_text(json.dumps({"arm_key": key}) + "\n")
    done = {r["id"] for r in rows}
    log(f"{arm_info['label']}: {len(prompts)} prompts, {len(done)} done, groups {groups}")

    from hibana.engine.sampler import SamplingParams
    runner = Runner(plan, max_context)
    arm_info.update({k: runner.info.get(k) for k in ("model", "config", "w3_identity", "engine_source_sha256", "device", "load_s")})
    params = SamplingParams(max_new_tokens=args.max_new, accept_top_k=accept[0], accept_margin=accept[1], trace=False,
                            ignore_eos=args.ignore_eos)
    runner.run(prompts[0]["prompt_tokens"], SamplingParams(max_new_tokens=64, trace=False))  # first-request effects
    reason = "complete"
    per = []
    for k, p in enumerate(prompts):
        if p["id"] in done:
            continue
        expect = 1.5 * sum(per[-20:]) / len(per[-20:]) if per else 0.0
        if time.monotonic() + expect >= deadline:
            reason = "max-minutes"
            break
        t = time.monotonic()
        r, decode_s, prefill_s, rounds_log = runner.run(p["prompt_tokens"], params)
        row = prompt_row(p, r, decode_s, prefill_s, rounds_log)
        per.append(time.monotonic() - t)
        rows.append(row)
        with open(partial, "a") as f:
            f.write(json.dumps(row) + "\n")
        if row["observed_rounds"] != row["rounds"]:
            log(f"prompt {k}: the depth policy saw {row['observed_rounds']} rounds, the request counted {row['rounds']}")
        if len(per) % 10 == 1:
            log(f"prompt {k}: {row['prompt_tokens']} -> {row['output_tokens']} tokens, {row['tok_per_round'] or 0:.2f} tok/round, "
                f"{row['decode_tps'] or 0:.1f} tok/s")
    order = {p["id"]: i for i, p in enumerate(prompts)}
    rows.sort(key=lambda r: order.get(r["id"], math.inf))
    result = {"label": arm_info["label"], "arm": arm_info, "complete": len(rows) == len(prompts), "stopped": reason,
              "summary": summarize(rows, split, args.split_model), "prompts": rows}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, out)
    s = result["summary"].get("all", {})
    log(f"done ({reason}): {len(rows)}/{len(prompts)} prompts, {s.get('tok_per_round') or 0:.3f} tok/round, "
        f"{s.get('decode_tps') or 0:.1f} tok/s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
