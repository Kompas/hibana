#!/usr/bin/env python3
"""Render exported trace segments to token ids exactly as the served model sees them, and build the datasets.

Input: `conversations.jsonl` from trace_export.py. Output (in --out-dir)::

    seqs-train.jsonl, seqs-heldout.jsonl   {"id","session","source","model","tokens","assist","window",...}
    prompts-heldout.jsonl                  {"id","session","source","model","prompt_tokens",...}
    stats.json                             counts only

Rendering. The chat template is the one vLLM serves (`--chat-template`), compiled the way transformers does for
vLLM (ImmutableSandboxedEnvironment, trim_blocks/lstrip_blocks, loopcontrols, and a `tojson` that keeps non-ASCII
characters and key order). Messages are given in the shape vLLM hands the template: tool-call arguments parsed to
dicts, `reasoning_content` carrying the assistant's reasoning. (A request that names the field `reasoning_content`
is rendered with an empty think block by vLLM; pi sends it as `reasoning`, which vLLM maps onto the template's
`reasoning_content`, see `vllm_request`.) The template renders every past assistant turn's reasoning
(`preserve_thinking` is undefined) and the generation prompt ends in `<think>\\n`; so the loss mask of an
assistant turn starts right after `<think>\\n` and ends after `<|im_end|>` (the trailing newline is the template's).

Token ids come from the snapshot's tokenizer.json (`tokenizers`, no special tokens added), the same file vLLM
serves. `assist` ranges are derived from character offsets of each assistant fragment, never from searching tokens.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jinja2
import jinja2.ext
import jinja2.sandbox
from tokenizers import Tokenizer

SEED = 20261124
MAX_TOKENS = 32768
WINDOW = 4096
MAX_PROMPT = 16384
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n"
DUMMY_USER = {"role": "user", "content": "x"}
USER_FRAGMENT = "<|im_start|>user\nx<|im_end|>\n"
# chat_template_kwargs the traces were served with (pi sets reasoning_effort=medium in its model config; the
# ZeroClaw config sends none, so the template default applies).
RENDER_KWARGS: dict[str, dict[str, Any]] = {"pi": {"reasoning_effort": "medium"}, "zeroclaw": {}}


# ----------------------------------------------------------------------------------------------------------------
# template + tokenizer


def _hf_tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None, separators: Any = None, sort_keys: bool = False) -> str:
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


class Renderer:
    def __init__(self, template_path: str | Path):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = _hf_tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: ""
        self.template = env.from_string(Path(template_path).read_text())

    def render(self, messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool, **kwargs: Any) -> str:
        return self.template.render(messages=[template_message(m) for m in messages], tools=tools or None, add_generation_prompt=add_generation_prompt, **kwargs)


def template_message(m: dict) -> dict:
    """A normalized message in the form the template sees (what vLLM passes)."""
    out: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant":
        out["reasoning_content"] = m.get("reasoning_content") or ""
        calls = m.get("tool_calls") or []
        if calls:
            out["tool_calls"] = [{"id": c.get("id", ""), "type": "function", "function": {"name": c["function"]["name"], "arguments": parse_arguments(c["function"].get("arguments"))}} for c in calls]
    return out


def parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw) if raw else {}
    except Exception:
        return {}
    return v if isinstance(v, dict) else {}


class Tok:
    def __init__(self, tokenizer_json: str | Path):
        self.tok = Tokenizer.from_file(str(tokenizer_json))

    def encode(self, text: str) -> tuple[list[int], list[int], list[int]]:
        enc = self.tok.encode(text, add_special_tokens=False)
        return enc.ids, [o[0] for o in enc.offsets], [o[1] for o in enc.offsets]

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ----------------------------------------------------------------------------------------------------------------
# rendering a segment with per-message spans


@dataclass
class Rendered:
    text: str
    ids: list[int]
    starts: list[int]  # char offset of each token
    spans: dict[int, tuple[int, int]]  # assistant message index -> (char start of fragment, char end incl. trailing "\n")
    assist: dict[int, tuple[int, int]]  # assistant message index -> token range [s, e) produced by the model
    header_end: dict[int, int]  # assistant message index -> char offset where generation starts (after <think>\n)


def render_kwargs(conv: dict) -> dict:
    return conv.get("render") or RENDER_KWARGS.get(conv["source"], {})


def _fragment(r: Renderer, msg: dict, kwargs: dict) -> str:
    base = r.render([DUMMY_USER], None, False, **kwargs)
    full = r.render([DUMMY_USER, msg], None, False, **kwargs)
    assert full.startswith(base)
    return full[len(base) :]


def render_segment(conv: dict, r: Renderer, tok: Tok, spot_checks: int = 4, rng: random.Random | None = None) -> Rendered:
    """Render all messages once; locate each assistant fragment by its own rendering and verify by prefix renders."""
    msgs, tools, kw = conv["messages"], conv.get("tools") or None, render_kwargs(conv)
    text = r.render(msgs, tools, False, **kw)
    spans: dict[int, tuple[int, int]] = {}
    sys_prefix = 1 if msgs and msgs[0]["role"] == "system" else 0
    prelude = r.render(msgs[:sys_prefix] + [DUMMY_USER], tools, False, **kw)
    if not prelude.endswith(USER_FRAGMENT):
        raise AssertionError("unexpected user fragment")
    prelude = prelude[: -len(USER_FRAGMENT)]  # system block (+ tools + reasoning instructions), if any
    if not text.startswith(prelude):
        raise AssertionError("prelude mismatch")
    cursor = len(prelude)
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        frag = _fragment(r, m, kw)
        pos = text.find(frag, cursor)
        if pos < 0:
            raise AssertionError("assistant fragment not found")
        spans[j] = (pos, pos + len(frag))
        cursor = pos + len(frag)
    # verify: the text before a sampled assistant fragment is exactly the prefix render
    rng = rng or random.Random(SEED)
    if spans and spot_checks:
        for j in rng.sample(sorted(spans), min(spot_checks, len(spans))):
            prefix = r.render(msgs[:j], tools, False, **kw)
            if not (len(prefix) == spans[j][0] and text.startswith(prefix)):
                spans = exact_spans(msgs, tools, kw, r, text)
                break
    ids, starts, _ = tok.encode(text)
    assist: dict[int, tuple[int, int]] = {}
    header_end: dict[int, int] = {}
    for j, (cs, ce) in spans.items():
        if not text.startswith(ASSISTANT_HEADER, cs) or not text.endswith("<|im_end|>\n", cs, ce):
            raise AssertionError("unexpected assistant fragment shape")
        gs, ge = cs + len(ASSISTANT_HEADER), ce - 1  # generation: after "<think>\n", through "<|im_end|>"
        header_end[j] = gs
        assist[j] = (bisect.bisect_left(starts, gs), bisect.bisect_left(starts, ge))
    return Rendered(text, ids, starts, spans, assist, header_end)


def exact_spans(msgs: list[dict], tools: list[dict] | None, kw: dict, r: Renderer, text: str) -> dict[int, tuple[int, int]]:
    """Slow, exact fragment spans from prefix renders (used when the fast path fails its verification)."""
    spans: dict[int, tuple[int, int]] = {}
    for j, m in enumerate(msgs):
        if m["role"] == "assistant":
            a = len(r.render(msgs[:j], tools, False, **kw)) if j else 0
            b = len(r.render(msgs[: j + 1], tools, False, **kw))
            spans[j] = (a, b)
    return spans


# ----------------------------------------------------------------------------------------------------------------
# splitting, sample and prompt selection


def split_sessions(convs: list[dict], seed: int = SEED, frac: float = 0.15) -> dict[str, str]:
    """session -> "train" | "heldout": 85/15 by session, fixed seed.

    Stratified by source and, within a source, by whether the served model (qwen-trial117) produced the session;
    inside a stratum the sessions are ordered by size (assistant turns) and cut into groups of round(1/frac), one
    random session of each group is held out, so the held-out set spans small and large sessions instead of
    whatever the shuffle gives (a plain shuffle left 7 pi sessions and almost no assistant text).
    """
    size: dict[str, int] = collections.defaultdict(int)
    stratum: dict[str, tuple[str, bool]] = {}
    for c in convs:
        sid = c["session"]
        served = any(m.get("model") == "qwen-trial117" for m in c["messages"] if m["role"] == "assistant")
        size[sid] += len(new_assistant_indexes(c))
        old = stratum.get(sid)
        stratum[sid] = (c["source"], served or (old[1] if old else False))
    strata: dict[tuple[str, bool], list[str]] = collections.defaultdict(list)
    for sid, key in stratum.items():
        strata[key].append(sid)
    group = max(2, int(round(1 / frac)))
    out: dict[str, str] = {}
    rng = random.Random(seed)
    for key in sorted(strata):
        sids = sorted(strata[key], key=lambda x: (size[x], x))
        for g in range(0, len(sids), group):
            chunk = sids[g : g + group]
            if len(chunk) >= group - 3 or (len(chunk) >= 2 and rng.random() < len(chunk) / group):
                held = {rng.choice(chunk)}
            else:
                held = set()
            for sid in chunk:
                out[sid] = "heldout" if sid in held else "train"
    return out


def new_assistant_indexes(conv: dict) -> list[int]:
    return [j for j, m in enumerate(conv["messages"]) if m["role"] == "assistant" and j >= conv.get("carried", 0)]


def contains_dropped_image(conv: dict, upto: int) -> bool:
    return any(m.get("has_dropped_image") for m in conv["messages"][:upto])


def select_windows(rd: Rendered, new_idx: list[int], max_tokens: int = MAX_TOKENS, window: int = WINDOW, min_gain: int = 64, min_frac: float = 0.5, cap: int = 200) -> list[int]:
    """Greedy cover of the segment's new assistant tokens by `window`-token windows that end at an assistant turn.

    A turn is a candidate if its end is within `max_tokens`. The next pick is the candidate with the most
    not-yet-covered assistant tokens in its window, provided that is at least `min_gain` and at least `min_frac`
    of the window's new assistant tokens (so windows overlap by at most half of their trainable text).
    """
    horizon = min(len(rd.ids), max_tokens)
    mine = [0] * (horizon + 1)  # 1 where a not-yet-covered, new assistant token sits
    for j in new_idx:
        s, e = rd.assist[j]
        for t in range(s, min(e, horizon)):
            mine[t] = 1
    total_prefix = [0] * (horizon + 1)
    for t in range(horizon):
        total_prefix[t + 1] = total_prefix[t] + mine[t]
    cands = [j for j in new_idx if rd.assist[j][1] <= horizon and rd.assist[j][1] > rd.assist[j][0]]
    picked: list[int] = []
    unc = mine[:]
    while len(picked) < cap:
        pre = [0] * (horizon + 1)
        for t in range(horizon):
            pre[t + 1] = pre[t] + unc[t]
        best, best_gain = None, 0
        for j in cands:
            if j in picked:
                continue
            b = rd.assist[j][1]
            a = max(0, b - window)
            gain = pre[b] - pre[a]
            full = total_prefix[b] - total_prefix[a]
            if gain >= min_gain and gain >= min_frac * full and gain > best_gain:
                best, best_gain = j, gain
        if best is None:
            break
        picked.append(best)
        b = rd.assist[best][1]
        for t in range(max(0, b - window), b):
            unc[t] = 0
    return sorted(picked)


def make_sample(conv: dict, rd: Rendered, j: int, tok: Tok) -> dict:
    """The sample that ends at assistant message j.

    `tokens` is the rendered conversation up to and including `<|im_end|>` of message j. Earlier turns are
    tokenized as part of the whole rendered text (that is what the served model is sent for its history). The
    last turn is tokenized the way it was generated: the prompt (ending in `<think>\n`) and the completion
    separately. They are the same tokens unless the completion starts with a newline (an empty think block), where
    one tokenization of the whole text would fuse the prompt's `\n` with it into a token the model never saw; then
    `boundary_merged` is true and the prompt/completion tokenization is used.
    """
    s, e = rd.assist[j]
    gs = rd.header_end[j]
    merged = not (s < len(rd.starts) and rd.starts[s] == gs)
    if merged:
        prompt_ids = tok.ids(rd.text[:gs])
        completion_ids = tok.ids(rd.text[gs : rd.spans[j][1] - 1])
        tokens = prompt_ids + completion_ids
        s, e = len(prompt_ids), len(tokens)
    else:
        tokens = rd.ids[:e]
    assist, models = [], []
    for k, (a, b) in sorted(rd.assist.items()):
        if k < j and b > a:
            assist.append([a, b])
            models.append(conv["messages"][k].get("model") or "")
    assist.append([s, e])
    models.append(conv["messages"][j].get("model") or "")
    return {
        "id": f"{conv['session']}:{conv['segment']}:{j}",
        "session": conv["session"],
        "segment": conv["segment"],
        "source": conv["source"],
        "model": conv["messages"][j].get("model") or "",
        "tokens": tokens,
        "assist": assist,
        "assist_model": models,
        "window": [max(0, e - WINDOW), e],
        "gen_start": s,
        "boundary_merged": merged,
        "system_source": conv.get("system_source", ""),
    }


def make_prompt(conv: dict, rd: Rendered, tok: Tok, j: int) -> dict | None:
    """Context up to (not including) assistant message j, ending with the generation prompt."""
    cs = rd.spans[j][0]
    text = rd.text[:cs] + ASSISTANT_HEADER
    ids = tok.ids(text)
    if len(ids) > MAX_PROMPT:
        return None
    return {
        "id": f"{conv['session']}:{conv['segment']}:{j}",
        "session": conv["session"],
        "segment": conv["segment"],
        "source": conv["source"],
        "model": conv["messages"][j].get("model") or "",
        "prompt_tokens": ids,
    }


# ----------------------------------------------------------------------------------------------------------------
# oracle (vLLM /tokenize)


def vllm_request(conv: dict, upto: int, add_generation_prompt: bool) -> dict:
    """The /tokenize body for messages[:upto]: what a client would send vLLM (reasoning named `reasoning`)."""
    msgs = []
    for m in conv["messages"][:upto]:
        o: dict[str, Any] = {"role": m["role"], "content": m.get("content")}
        if m["role"] == "assistant":
            if m.get("reasoning_content"):
                o["reasoning"] = m["reasoning_content"]
            if m.get("tool_calls"):
                o["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool":
            o["tool_call_id"] = m["tool_call_id"]
        msgs.append(o)
    body: dict[str, Any] = {"model": "qwen-trial117", "messages": msgs, "add_generation_prompt": add_generation_prompt}
    if conv.get("tools"):
        body["tools"] = conv["tools"]
    kw = render_kwargs(conv)
    if kw:
        body["chat_template_kwargs"] = kw
    return body


def vllm_tokenize(url: str, body: dict, timeout: int = 600) -> list[int]:
    req = urllib.request.Request(url.rstrip("/") + "/tokenize", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["tokens"]


# ----------------------------------------------------------------------------------------------------------------
# driver


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build(convs: list[dict], r: Renderer, tok: Tok, out_dir: Path, n_pi_prompts: int = 60, seed: int = SEED, max_tokens: int = MAX_TOKENS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    split = split_sessions(convs, seed)
    rng = random.Random(seed)
    hdr_n = len(tok.ids(ASSISTANT_HEADER))
    files = {k: open(out_dir / name, "w", encoding="utf-8") for k, name in (("train", "seqs-train.jsonl"), ("heldout", "seqs-heldout.jsonl"), ("prompts", "prompts-heldout.jsonl"))}
    for name in ("seqs-train.jsonl", "seqs-heldout.jsonl", "prompts-heldout.jsonl"):
        os.chmod(out_dir / name, 0o600)
    stats: dict[str, Any] = {
        "split_sessions": collections.Counter(),
        "segments": collections.Counter(),
        "samples": collections.Counter(),
        "samples_with_prompt_completion_tokenization": collections.Counter(),
        "window_tokens": collections.Counter(),
        "assistant_tokens_in_window": collections.Counter(),
        "assistant_tokens_in_seqs": collections.Counter(),
        "tokens_in_seqs": collections.Counter(),
        "assistant_tokens_total": collections.Counter(),
        "assistant_tokens_new_total": collections.Counter(),
        "assistant_tokens_beyond_max_tokens": collections.Counter(),
        "new_assistant_tokens_by_turn_end": collections.Counter(),
        "excluded": collections.Counter(),
        "prompts": collections.Counter(),
        "render_errors": 0,
    }
    seen_sess: dict[str, str] = {}
    for c in convs:
        seen_sess.setdefault(c["session"], c["source"])
    for sid, src in seen_sess.items():
        stats["split_sessions"][f"{split[sid]}/{src}"] += 1

    prompt_cands: dict[str, list[tuple[dict, Rendered, int]]] = collections.defaultdict(list)  # per session
    kept_convs: list[dict] = []
    for ci, conv in enumerate(convs):
        sp = split[conv["session"]]
        key = f"{sp}/{conv['source']}"
        try:
            rd = render_segment(conv, r, tok, rng=rng)
        except Exception as exc:  # a segment the template rejects (e.g. no user query)
            stats["render_errors"] += 1
            stats["excluded"][f"render_error:{type(exc).__name__}"] += 1
            continue
        stats["segments"][key] += 1
        new_idx = new_assistant_indexes(conv)
        for j in new_idx:
            s, e = rd.assist[j]
            model = conv["messages"][j].get("model") or ""
            stats["assistant_tokens_new_total"][f"{key}/{model}"] += e - s
            if e > max_tokens:
                stats["assistant_tokens_beyond_max_tokens"][f"{key}/{model}"] += e - s
            bucket = next(b for b in (8192, 16384, 32768, 65536, 131072, 262144, 10**9) if e <= b)
            stats["new_assistant_tokens_by_turn_end"][f"{conv['source']}/<= {bucket}"] += e - s
        for j in [k for k, m in enumerate(conv["messages"]) if m["role"] == "assistant"]:
            s, e = rd.assist[j]
            stats["assistant_tokens_total"][key] += e - s
        reachable = [j for j in new_idx if rd.assist[j][1] <= max_tokens]
        cand = [j for j in reachable if not contains_dropped_image(conv, j + 1)]
        stats["excluded"]["turn_with_dropped_image_in_context(within_max_tokens)"] += len(reachable) - len(cand)
        picks = select_windows(rd, cand, max_tokens=max_tokens)
        for j in picks:
            smp = make_sample(conv, rd, j, tok)
            files[sp].write(json.dumps(smp) + "\n")
            a, b = smp["window"]
            in_win = sum(max(0, min(y, b) - max(x, a)) for x, y in smp["assist"])
            model = smp["model"]
            if smp["boundary_merged"]:
                stats["samples_with_prompt_completion_tokenization"][f"{key}/{model}"] += 1
            for k in (key, f"{key}/{model}"):
                stats["samples"][k] += 1
                stats["window_tokens"][k] += b - a
                stats["assistant_tokens_in_window"][k] += in_win
                stats["assistant_tokens_in_seqs"][k] += sum(y - x for x, y in smp["assist"])
                stats["tokens_in_seqs"][k] += len(smp["tokens"])
        if sp == "heldout" or conv["source"] == "zeroclaw":
            for j in new_idx:
                if contains_dropped_image(conv, j):
                    stats["excluded"]["prompt_with_dropped_image_in_context"] += 1
                elif bisect.bisect_left(rd.starts, rd.spans[j][0]) + hdr_n > MAX_PROMPT:
                    stats["excluded"]["prompt_longer_than_16384"] += 1
                else:
                    prompt_cands[conv["session"]].append((conv, rd, j))
        kept_convs.append(conv)
    # prompts: all usable ZeroClaw turns, ~n_pi_prompts pi turns spread round-robin over heldout sessions
    pi_pool: dict[str, list] = {sid: v for sid, v in prompt_cands.items() if seen_sess[sid] == "pi" and split[sid] == "heldout"}
    for v in pi_pool.values():
        rng.shuffle(v)
        # pop() takes from the end: turns the served model produced come first, then the others
        v.sort(key=lambda t: t[0]["messages"][t[2]].get("model") == "qwen-trial117")
    chosen: list[tuple[dict, Rendered, int]] = []
    order = sorted(pi_pool)
    while len(chosen) < n_pi_prompts and any(pi_pool.values()):
        for sid in order:
            if pi_pool[sid] and len(chosen) < n_pi_prompts:
                chosen.append(pi_pool[sid].pop())
    for sid, v in prompt_cands.items():
        if seen_sess[sid] == "zeroclaw":
            chosen.extend(v)
    prompts = []
    for conv, rd, j in chosen:
        p = make_prompt(conv, rd, tok, j)
        if p is None:
            stats["excluded"]["prompt_longer_than_16384"] += 1
            continue
        p["in_train"] = split[conv["session"]] == "train"
        prompts.append(p)
        files["prompts"].write(json.dumps(p) + "\n")
        stats["prompts"][f"{conv['source']}/{p['model']}"] += 1
        stats["prompts"][f"{conv['source']}/in_train={p['in_train']}"] += 1
    for f in files.values():
        f.close()
    # split disjointness
    sess = {"train": set(), "heldout": set()}
    for name, sp in (("seqs-train.jsonl", "train"), ("seqs-heldout.jsonl", "heldout")):
        for row in read_jsonl(out_dir / name):
            sess[sp].add(row["session"])
    stats["session_overlap_train_heldout"] = sorted(sess["train"] & sess["heldout"])
    stats["sessions_in_train_seqs"], stats["sessions_in_heldout_seqs"] = len(sess["train"]), len(sess["heldout"])
    for k, v in list(stats.items()):
        if isinstance(v, collections.Counter):
            stats[k] = dict(sorted(v.items()))
    stats["_convs"] = kept_convs
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True, help="dir with tokenizer.json (and chat_template.jinja)")
    ap.add_argument("--template", default=None, help="chat template to render with (default: <snapshot>/chat_template.jinja); use the one vLLM serves")
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--export-stats", nargs="*", default=[], help="export-stats JSON files from trace_export.py (merged under stats[\"export\"])")
    ap.add_argument("--oracle-url", default=None, help="vLLM base URL; compares token ids with /tokenize on random samples")
    ap.add_argument("--oracle-n", type=int, default=20)
    ap.add_argument("--n-pi-prompts", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="longest sequence (tokens) a sample may have")
    args = ap.parse_args(argv)

    snap = Path(args.snapshot)
    r = Renderer(args.template or snap / "chat_template.jinja")
    tok = Tok(snap / "tokenizer.json")
    convs = list(read_jsonl(args.conversations))
    out_dir = Path(args.out_dir)
    stats = build(convs, r, tok, out_dir, n_pi_prompts=args.n_pi_prompts, max_tokens=args.max_tokens)
    kept = stats.pop("_convs")
    result: dict[str, Any] = {}
    if args.oracle_url:
        result = oracle_check(out_dir, kept, r, tok, args.oracle_url, args.oracle_n)
        stats["oracle"] = result
    stats["export"] = {Path(f).stem: json.load(open(f)) for f in args.export_stats if os.path.exists(f)}
    with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1, sort_keys=True)
    os.chmod(out_dir / "stats.json", 0o600)
    print(json.dumps({k: v for k, v in stats.items() if k in ("split_sessions", "samples", "prompts", "session_overlap_train_heldout", "oracle")}, indent=1, sort_keys=True))
    return 0


def oracle_check(out_dir: Path, convs: list[dict], r: Renderer, tok: Tok, url: str, n: int, seed: int = SEED) -> dict:
    """Compare our ids with vLLM /tokenize for `n` random train/heldout samples, and as many random prompts."""
    index = {(c["session"], c["segment"]): c for c in convs}
    rng = random.Random(seed + 1)
    rows = [row for name in ("seqs-train.jsonl", "seqs-heldout.jsonl") for row in read_jsonl(out_dir / name)]
    rng.shuffle(rows)
    # sample across sources and models, shortest-first would bias; take a random but stratified-ish draw
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_kind[f"{row['source']}/{'qwen' if row['model'] == 'qwen-trial117' else 'other'}"].append(row)
    picked: list[dict] = []
    kinds = sorted(by_kind)
    while len(picked) < n and any(by_kind.values()):
        for k in kinds:
            if by_kind[k] and len(picked) < n:
                picked.append(by_kind[k].pop())
    results = []
    nl = tok.ids("\n")
    for row in picked:
        conv = index[(row["session"], row["segment"])]
        j = int(row["id"].rsplit(":", 1)[1])
        # (a) the prompt of the last turn, as the server tokenizes it with the generation prompt
        prompt_ok = vllm_tokenize(url, vllm_request(conv, j, True)) == row["tokens"][: row["gen_start"]]
        # (b) the whole sample, as the server tokenizes the conversation (identical unless the boundary is merged)
        full_ok = vllm_tokenize(url, vllm_request(conv, j + 1, False)) == row["tokens"] + nl
        results.append({"id_hash": _h(row["id"]), "kind": f"{row['source']}/{row['model']}", "n_tokens": len(row["tokens"]), "boundary_merged": row["boundary_merged"], "prompt_match": prompt_ok, "full_match": full_ok, "match": prompt_ok and (full_ok or row["boundary_merged"])})
    prompt_rows = list(read_jsonl(out_dir / "prompts-heldout.jsonl"))
    rng.shuffle(prompt_rows)
    presults = []
    for row in prompt_rows[:n]:
        conv = index[(row["session"], row["segment"])]
        j = int(row["id"].rsplit(":", 1)[1])
        theirs = vllm_tokenize(url, vllm_request(conv, j, True))
        presults.append({"id_hash": _h(row["id"]), "n_tokens": len(row["prompt_tokens"]), "match": theirs == row["prompt_tokens"]})
    return {
        "samples_checked": len(results),
        "samples_match": sum(x["match"] for x in results),
        "samples_prompt_match": sum(x["prompt_match"] for x in results),
        "samples_full_match": sum(x["full_match"] for x in results),
        "samples_boundary_merged": sum(x["boundary_merged"] for x in results),
        "prompts_checked": len(presults),
        "prompts_match": sum(x["match"] for x in presults),
        "detail": results + [{"prompt": True, **x} for x in presults],
    }


def _h(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode()).hexdigest()[:10]


if __name__ == "__main__":
    raise SystemExit(main())
