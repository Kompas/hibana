#!/usr/bin/env python3
"""Export agent traces (pi coding agent sessions, ZeroClaw session DBs) as normalized conversation segments.

Output (`conversations.jsonl`, one JSON object per segment)::

    {"session", "segment", "source": "pi"|"zeroclaw", "model", "trigger",
     "messages": [OpenAI-style: system / user / assistant(tool_calls, reasoning_content, model) / tool],
     "tools": [{"type": "function", "function": {...}}], "carried": int, ...}

`carried` is the number of leading messages that were already part of the previous segment's context (they are
context only: samples are drawn from the messages after them, so no assistant text is trained on twice).

What a segment is. pi sessions are append-only logs, but three entry kinds change what the model sees afterwards
and therefore start a new segment: `compaction` (the context becomes system + summary + kept tail), `context_edit`
(an earlier message is omitted or replaced) and a mid-conversation `system` message (system prompt sections / tool
set change). The context at every point is derived with the same projection pi uses (`buildSessionContext`), and
the OpenAI messages are derived with the same transforms pi applies for an openai-completions model
(`transformMessages` + `convertMessages`), so the segments hold what the served model was sent.

Privacy: this module reads private traces. It prints counts only and writes message content only to the output.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
KNOWN_ENTRY_TYPES = {"session", "model_change", "thinking_level_change", "message", "compaction", "context_edit", "custom"}
KNOWN_ROLES = {"system", "user", "assistant", "toolResult"}
# Reasoning fields pi replays on an openai-completions assistant message (everything else is dropped by pi).
REASONING_FIELDS = {"reasoning", "reasoning_content", "reasoning_text"}

# ----------------------------------------------------------------------------------------------------------------
# pi: loading


@dataclass
class PiSession:
    path: str
    session_id: str
    cwd: str
    timestamp: str
    entries: list[dict]  # every entry except the header, in file order
    lines: int = 0
    unparsed: int = 0
    type_counts: collections.Counter = field(default_factory=collections.Counter)
    off_path: set[str] = field(default_factory=set)  # entry ids not on the active branch


def load_pi_session(path: str | Path) -> PiSession:
    """Parse one pi session file. A line that is not valid JSON, has no known `type`, or has an unknown message
    role counts as unparsed (and is not used)."""
    header: dict | None = None
    entries: list[dict] = []
    lines = unparsed = 0
    counts: collections.Counter = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            lines += 1
            try:
                d = json.loads(raw)
                t = d["type"]
                if t not in KNOWN_ENTRY_TYPES:
                    raise ValueError(t)
                if t == "message" and d["message"]["role"] not in KNOWN_ROLES:
                    raise ValueError(d["message"]["role"])
            except Exception:
                unparsed += 1
                continue
            counts[t] += 1
            if t == "session":
                header = d
            else:
                entries.append(d)
    if header is None:
        header = {"id": hashlib.sha1(str(path).encode()).hexdigest()[:16], "cwd": "", "timestamp": ""}
    sess = PiSession(str(path), header.get("id", ""), header.get("cwd", ""), header.get("timestamp", ""), entries, lines, unparsed, counts)
    sess.off_path = _off_path_ids(entries)
    return sess


def _off_path_ids(entries: list[dict]) -> set[str]:
    """Entries that are not ancestors of the last entry (an abandoned branch). Empty for a linear session."""
    by_id = {e["id"]: e for e in entries if "id" in e}
    if not entries:
        return set()
    leaf = entries[-1].get("id")
    on_path: set[str] = set()
    cur = leaf
    while cur is not None and cur in by_id and cur not in on_path:
        on_path.add(cur)
        cur = by_id[cur].get("parentId")
    return {i for i in by_id if i not in on_path}


# ----------------------------------------------------------------------------------------------------------------
# pi: context projection (mirrors pi's buildContextEntries / projectContextEntry / sessionEntryToContextMessages)


def _entry_messages(entry: dict) -> list[dict]:
    t = entry["type"]
    if t == "message":
        m = entry["message"]
        if m["role"] == "system" and m.get("content") is None:
            return [{**m, "content": ""}]
        if m["role"] in ("user", "assistant", "toolResult") and m.get("content") is None:
            return [{**m, "content": []}]
        return [m]
    if t == "compaction":
        summary = {"role": "compactionSummary", "summary": entry.get("summary", ""), "tokensBefore": entry.get("tokensBefore")}
        sysm = entry.get("systemMessage")
        return [sysm, summary] if sysm else [summary]
    return []


def _project_entry(entry: dict, edit: dict | None) -> list[dict]:
    msgs = _entry_messages(entry)
    if edit is None:
        return msgs
    repl = edit.get("replacement")
    if repl is None:
        return []
    out = []
    for m in msgs:
        if m["role"] not in ("user", "assistant", "toolResult", "custom"):
            out.append(m)
            continue
        content = repl.get("content")
        if m["role"] in ("assistant", "toolResult") and isinstance(content, str):
            content = [{"type": "text", "text": content}]
        out.append({**m, "content": content})
    return out


def project_context(path: list[dict]) -> list[tuple[str, dict]]:
    """The context pi would send after `path` (a prefix of the session): a list of (entry id, message)."""
    comp_idx = None
    for i, e in enumerate(path):
        if e["type"] == "compaction":
            comp_idx = i
    if comp_idx is None:
        ctx_entries = list(path)
    else:
        comp = path[comp_idx]
        ctx_entries = [comp]
        found = False
        first_kept = comp.get("firstKeptEntryId")
        for e in path[:comp_idx]:
            if e.get("id") == first_kept:
                found = True
            if found and not (e["type"] == "message" and e["message"]["role"] == "system"):
                ctx_entries.append(e)
        ctx_entries.extend(path[comp_idx + 1 :])
    edits = {e["targetId"]: e for e in ctx_entries if e["type"] == "context_edit"}
    items: list[tuple[str, dict]] = []
    for idx, e in enumerate(ctx_entries):
        msgs = [] if (e["type"] == "compaction" and idx > 0) else _project_entry(e, edits.get(e.get("id")))
        items.extend((e.get("id", ""), m) for m in msgs)
    return items


# ----------------------------------------------------------------------------------------------------------------
# pi: system prompt / tools (mirrors getSystemMessageText / getCurrentSystemMessage / getCurrentTools)


def merged_system(system_msgs: list[dict]) -> tuple[str, list[dict]]:
    """(system prompt text, tool definitions) after applying every system message in order."""
    content: list[str] = []
    sections: dict[str, str] = {}
    tools: dict[str, dict] = {}
    for m in system_msgs:
        text = _content_text(m.get("content"))
        if text:
            content.append(text)
        for name, value in (m.get("sections") or {}).items():
            if value is None:
                sections.pop(name, None)
            else:
                sections[name] = value
        for t in m.get("toolsRemoved") or []:
            tools.pop(t["name"], None)
        for t in m.get("toolsAdded") or []:
            tools[t["name"]] = t
    parts = [part for part in ["\n\n".join(content), *sections.values()] if part]
    return "\n\n".join(parts), list(tools.values())


def _content_text(content: Any, sep: str = "\n") -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return sep.join(b.get("text", "") for b in content if b.get("type") == "text")


def openai_tool(tool: dict) -> dict:
    """pi's convertTools for this provider (no strict mode, no grammar tools)."""
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("parameters", {"type": "object", "properties": {}})}}


# ----------------------------------------------------------------------------------------------------------------
# pi: transformMessages + convertMessages (openai-completions), keeping thinking for every assistant turn


def _jstrip_empty(s: str) -> bool:
    return len(s.strip()) == 0


def to_openai_messages(items: list[tuple[str, dict]], n_carried_items: int = 0) -> tuple[list[dict], list[str | None], list[bool], dict]:
    """Convert projected non-system context items into OpenAI messages.

    Returns (messages, source entry ids, `new` flags, info). `new` is False for messages made from the first
    `n_carried_items` items. info counts what was dropped and why (image blocks, errored assistants, ...).
    """
    info: collections.Counter = collections.Counter()
    out: list[dict] = []
    src: list[str | None] = []
    new: list[bool] = []
    pending: list[dict] = []  # tool calls of the last kept assistant that have no result yet
    seen_results: set[str] = set()

    def emit(msg: dict, entry_id: str | None, is_new: bool) -> None:
        out.append(msg)
        src.append(entry_id)
        new.append(is_new)

    def close_pending(is_new: bool) -> None:
        nonlocal pending, seen_results
        for tc in pending:
            if tc["id"] not in seen_results:
                emit({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": "No result provided"}, None, is_new)
                info["synthetic_tool_result"] += 1
        pending = []
        seen_results = set()

    for pos, (entry_id, m) in enumerate(items):
        is_new = pos >= n_carried_items
        role = m["role"]
        if role == "assistant":
            close_pending(is_new)
            if m.get("stopReason") in ("error", "aborted"):
                info["assistant_error_or_aborted"] += 1
                continue
            blocks = m.get("content") or []
            texts = [b["text"] for b in blocks if b.get("type") == "text" and not _jstrip_empty(b.get("text", ""))]
            thinks = [b["thinking"] for b in blocks if b.get("type") == "thinking" and not _jstrip_empty(b.get("thinking", ""))]
            calls = [b for b in blocks if b.get("type") == "toolCall"]
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None, "model": m.get("model")}
            if thinks:
                msg["reasoning_content"] = "\n".join(thinks)
            if calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))}}
                    for c in calls
                ]
            if not texts and not calls:
                # pi's convertMessages drops an assistant message with neither text nor tool calls.
                info["assistant_empty"] += 1
                continue
            pending = [{"id": c["id"], "name": c["name"]} for c in calls]
            seen_results = set()
            emit(msg, entry_id, is_new)
        elif role == "toolResult":
            seen_results.add(m.get("toolCallId"))
            blocks = m.get("content") or []
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            n_img = sum(1 for b in blocks if b.get("type") == "image")
            if n_img:
                info["images_dropped"] += n_img
            if text.strip():
                body = text
            elif n_img:
                body = "(see attached image)"
            else:
                body = "(no tool output)"
            emit({"role": "tool", "tool_call_id": m.get("toolCallId"), "name": m.get("toolName"), "content": body, **({"has_dropped_image": True} if n_img else {})}, entry_id, is_new)
        elif role in ("user", "compactionSummary"):
            close_pending(is_new)
            if role == "compactionSummary":
                body = COMPACTION_SUMMARY_PREFIX + m.get("summary", "") + COMPACTION_SUMMARY_SUFFIX
            else:
                c = m.get("content")
                if isinstance(c, str):
                    body = c
                else:
                    body = "".join(b.get("text", "") for b in c if b.get("type") == "text" and b.get("text"))
                    if any(b.get("type") == "image" for b in c):
                        info["user_images_dropped"] += 1
                    if not body:
                        continue
            emit({"role": "user", "content": body}, entry_id, is_new)
        else:  # system handled by the caller
            continue
    close_pending(True)
    return out, src, new, dict(info)


# ----------------------------------------------------------------------------------------------------------------
# pi: replay a session into segments


@dataclass
class Segment:
    trigger: str
    messages: list[dict]
    src_ids: list[str | None]
    carried: int
    tools: list[dict]
    system_source: str
    info: dict
    model: str
    first_item_ids: list[str]


def _ctx_key(items: list[tuple[str, dict]]) -> list[str]:
    return [i + "|" + hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()[:12] for i, m in items]


def build_segment(items: list[tuple[str, dict]], n_carried_items: int, trigger: str, fallback_system: tuple[str, list[dict], str] | None) -> Segment | None:
    sys_msgs = [m for _, m in items if m["role"] == "system"]
    rest = [(i, m) for i, m in items if m["role"] != "system"]
    # carried count in `rest` coordinates
    carried_rest = sum(1 for pos, (_, m) in enumerate(items) if pos < n_carried_items and m["role"] != "system")
    if sys_msgs:
        text, tools = merged_system(sys_msgs)
        system_source = "stored"
    elif fallback_system is not None:
        text, tools, system_source = fallback_system
    else:
        text, tools, system_source = "", [], "none"
    msgs, src, new, info = to_openai_messages(rest, carried_rest)
    # the chat template needs a user query first: drop leading non-user messages
    lead = 0
    while lead < len(msgs) and msgs[lead]["role"] != "user":
        lead += 1
    if lead:
        info["leading_non_user_dropped"] = lead
        msgs, src, new = msgs[lead:], src[lead:], new[lead:]
    if not msgs:
        return None
    carried = 0
    while carried < len(new) and not new[carried]:
        carried += 1
    if any(not n for n in new[carried:]):  # carried flags must be a prefix
        raise AssertionError("carried messages are not a prefix")
    prefix: list[dict] = []
    if text:
        prefix = [{"role": "system", "content": text}]
    model = next((m["model"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("model")), "")
    return Segment(trigger, prefix + msgs, [None] * len(prefix) + src, carried + len(prefix), [openai_tool(t) for t in tools], system_source, info, model, [i for i, _ in rest])


def replay_pi_session(sess: PiSession, fallback_system: tuple[str, list[dict], str] | None = None) -> list[Segment]:
    path = [e for e in sess.entries if e.get("id") not in sess.off_path]
    segments: list[Segment] = []
    cur_items: list[tuple[str, dict]] = []
    seg_start_items = 0  # number of leading items in cur_items that were carried into the open segment
    trigger = "start"

    def close() -> None:
        seg = build_segment(cur_items, seg_start_items, trigger, fallback_system)
        if seg is not None and len(seg.messages) > seg.carried:
            segments.append(seg)

    for t, e in enumerate(path):
        if e["type"] == "message" and e["message"]["role"] != "system":
            cur_items.extend((e["id"], m) for m in _entry_messages(e))
            continue
        if e["type"] not in ("compaction", "context_edit") and not (e["type"] == "message" and e["message"]["role"] == "system"):
            continue
        new_items = project_context(path[: t + 1])
        old_keys, new_keys = _ctx_key(cur_items), _ctx_key(new_items)
        if new_keys == old_keys:
            continue
        is_system = e["type"] == "message"
        only_system_so_far = all(m["role"] == "system" for _, m in cur_items)
        pure_append = new_keys[: len(old_keys)] == old_keys
        if is_system and only_system_so_far:
            cur_items = new_items  # the initial system prompt: nothing to split yet
            continue
        if pure_append and not is_system:
            cur_items = new_items
            continue
        close()
        trigger = "system_update" if is_system else e["type"]
        cur_items = new_items
        seg_start_items = len(new_items)
    close()
    return segments


# ----------------------------------------------------------------------------------------------------------------
# pi: message accounting


def account_pi_messages(sess: PiSession, segments: list[Segment]) -> dict[str, str]:
    """Disposition of every `message` entry: used / system_prompt / system_update / skipped:<reason>."""
    used_ids: set[str] = set()
    for s in segments:
        used_ids.update(i for i in s.src_ids if i)
    out: dict[str, str] = {}
    first_system_seen = False
    for e in sess.entries:
        if e["type"] != "message":
            continue
        m, eid = e["message"], e["id"]
        if eid in sess.off_path:
            out[eid] = "skipped:off_active_branch"
        elif m["role"] == "system":
            out[eid] = "system_update" if first_system_seen else "system_prompt"
            first_system_seen = True
        elif eid in used_ids:
            out[eid] = "used"
        elif m["role"] == "assistant" and m.get("stopReason") in ("error", "aborted"):
            out[eid] = "skipped:assistant_error_or_aborted"
        elif m["role"] == "assistant" and not any(b.get("type") in ("text", "toolCall") and (b.get("type") == "toolCall" or b.get("text", "").strip()) for b in (m.get("content") or [])):
            out[eid] = "skipped:assistant_empty"
        else:
            out[eid] = "skipped:not_in_any_segment"
    return out


# ----------------------------------------------------------------------------------------------------------------
# pi: reconstruction of the system prompt for sessions that did not store one (old pi versions)


def reconstruct_system(sess: PiSession, donors: list[tuple[str, dict, str]]) -> tuple[str, list[dict], str] | None:
    """The system prompt + tools of the stored-prompt session closest in time, with this session's cwd.

    `donors`: (timestamp, first system message, session id). Sections that describe the donor's project
    (`project_context`) are dropped; `cwd` is rewritten. Returns None if there is no donor.
    """
    if not donors:
        return None
    ts = sess.timestamp
    best = min(donors, key=lambda d: (abs(_ts_seconds(d[0]) - _ts_seconds(ts)), d[0]))
    msg = copy.deepcopy(best[1])
    sections = {k: v for k, v in (msg.get("sections") or {}).items() if k != "project_context"}
    if "cwd" in sections:
        sections["cwd"] = f"<cwd>\n{sess.cwd}\n</cwd>"
    msg["sections"] = sections
    text, tools = merged_system([msg])
    return text, tools, "reconstructed"


def _ts_seconds(ts: str) -> float:
    import datetime as dt

    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


# ----------------------------------------------------------------------------------------------------------------
# ZeroClaw


def zeroclaw_segments(db_path: str, label: str) -> tuple[list[dict], dict]:
    """One conversation per session_key from a ZeroClaw `sessions.db` (opened read-only).

    ZeroClaw stores neither its system prompt (only a hash in the runtime trace) nor the tool schemas, so the
    conversations carry no system message and no tools.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("select id, session_key, role, content from sessions order by id").fetchall()
    con.close()
    by_key: dict[str, list[tuple[int, str, str]]] = collections.OrderedDict()
    for rid, key, role, content in rows:
        by_key.setdefault(key, []).append((rid, role, content))
    stats = {"rows_read": len(rows), "sessions": len(by_key), "unparsed": 0, "used": 0, "skipped": collections.Counter()}
    convs = []
    for key, rs in by_key.items():
        msgs: list[dict] = []
        for rid, role, content in rs:
            msg = _zc_message(role, content)
            if msg is None:
                stats["unparsed"] += 1
                continue
            msgs.append(msg)
        lead = 0
        while lead < len(msgs) and msgs[lead]["role"] != "user":
            lead += 1
        if lead:
            stats["skipped"]["leading_non_user"] += lead
        msgs = msgs[lead:]
        stats["used"] += len(msgs)
        if not any(m["role"] == "assistant" for m in msgs):
            stats["skipped"]["session_without_assistant_turn"] += len(msgs)
            stats["used"] -= len(msgs)
            continue
        sid = f"zc-{label}-" + hashlib.sha1(key.encode()).hexdigest()[:10]
        convs.append(
            {
                "session": sid,
                "segment": 0,
                "source": "zeroclaw",
                "model": "qwen-trial117",
                "trigger": "start",
                "messages": msgs,
                "tools": [],
                "carried": 0,
                "system_source": "none",
                "info": {},
            }
        )
    stats["skipped"] = dict(stats["skipped"])
    return convs, stats


def _zc_message(role: str, content: str) -> dict | None:
    if role not in ("user", "assistant", "tool", "system"):
        return None
    try:
        j = json.loads(content)
    except Exception:
        j = None
    if role == "assistant":
        if isinstance(j, dict) and ("tool_calls" in j or "content" in j or "reasoning_content" in j):
            msg: dict[str, Any] = {"role": "assistant", "content": j.get("content") or None, "model": "qwen-trial117"}
            if j.get("reasoning_content"):
                msg["reasoning_content"] = j["reasoning_content"]
            calls = j.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {"id": c.get("id", ""), "type": "function", "function": {"name": c.get("name", ""), "arguments": c.get("arguments") if isinstance(c.get("arguments"), str) else json.dumps(c.get("arguments", {}), ensure_ascii=False)}}
                    for c in calls
                ]
            return msg
        return {"role": "assistant", "content": content, "model": "qwen-trial117"}
    if role == "tool":
        if isinstance(j, dict) and "tool_call_id" in j:
            return {"role": "tool", "tool_call_id": j["tool_call_id"], "content": j.get("content") if isinstance(j.get("content"), str) else json.dumps(j.get("content"), ensure_ascii=False)}
        return {"role": "tool", "tool_call_id": "", "content": content}
    return {"role": role, "content": content}


# ----------------------------------------------------------------------------------------------------------------
# driver


def iter_pi_files(pi_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(os.path.expanduser(pi_dir)):
        out.extend(os.path.join(root, f) for f in files if f.endswith(".jsonl"))
    return sorted(out)


def export_pi(pi_dir: str) -> tuple[list[dict], dict]:
    sessions = [load_pi_session(p) for p in iter_pi_files(pi_dir)]
    donors = []
    for s in sessions:
        first = next((e["message"] for e in s.entries if e["type"] == "message" and e["message"]["role"] == "system"), None)
        if first is not None:
            donors.append((s.timestamp, first, s.session_id))
    convs: list[dict] = []
    stats: dict[str, Any] = {
        "files": len(sessions),
        "lines": sum(s.lines for s in sessions),
        "unparsed": sum(s.unparsed for s in sessions),
        "entry_types": dict(sum((s.type_counts for s in sessions), collections.Counter())),
        "messages_read": 0,
        "disposition": collections.Counter(),
        "dropped_in_conversion": collections.Counter(),
        "segments": 0,
        "sessions_with_reconstructed_system": 0,
        "segment_triggers": collections.Counter(),
    }
    for s in sessions:
        fb = None
        has_sys = any(e["type"] == "message" and e["message"]["role"] == "system" for e in s.entries)
        if not has_sys:
            fb = reconstruct_system(s, donors)
            if fb:
                stats["sessions_with_reconstructed_system"] += 1
        segs = replay_pi_session(s, fb)
        disp = account_pi_messages(s, segs)
        stats["messages_read"] += len(disp)
        stats["disposition"].update(disp.values())
        for k, seg in enumerate(segs):
            stats["segment_triggers"][seg.trigger] += 1
            stats["dropped_in_conversion"].update(seg.info)
            convs.append(
                {
                    "session": s.session_id,
                    "segment": k,
                    "source": "pi",
                    "model": seg.model,
                    "trigger": seg.trigger,
                    "messages": seg.messages,
                    "tools": seg.tools,
                    "carried": seg.carried,
                    "system_source": seg.system_source,
                    "info": seg.info,
                    "src_ids": seg.src_ids,
                }
            )
        stats["segments"] += len(segs)
    for k in ("disposition", "dropped_in_conversion", "segment_triggers"):
        stats[k] = dict(stats[k])
    return convs, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pi-dir", default="~/.pi/agent/sessions")
    ap.add_argument("--zc", action="append", default=[], metavar="LABEL=PATH", help="ZeroClaw sessions.db (repeatable)")
    ap.add_argument("--out", default="-", help="conversations.jsonl path, or - for stdout (pipe it straight to the Spark)")
    ap.add_argument("--stats-out", default=None, help="write export stats JSON (counts only)")
    args = ap.parse_args(argv)

    convs: list[dict] = []
    stats: dict[str, Any] = {}
    pi_convs, stats["pi"] = export_pi(args.pi_dir) if args.pi_dir else ([], {})
    convs += pi_convs
    stats["zeroclaw"] = {}
    for spec in args.zc:
        label, path = spec.split("=", 1)
        zc, zs = zeroclaw_segments(path, label)
        convs += zc
        stats["zeroclaw"][label] = zs
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    if args.out != "-":
        os.chmod(args.out, 0o600)
    for c in convs:
        out.write(json.dumps(c, ensure_ascii=False) + "\n")
    if out is not sys.stdout:
        out.close()
    if args.stats_out:
        with open(args.stats_out, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1, sort_keys=True)
    print(json.dumps({"segments": len(convs), "pi_messages_read": stats["pi"].get("messages_read"), "pi_unparsed": stats["pi"].get("unparsed")}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
