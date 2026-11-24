"""trace_export / trace_render on synthetic traces (no private data): every pi line type, compaction and
context_edit splits, loss-mask ranges, split disjointness.

Rendering tests use a small Qwen-style template and a one-token-per-character tokenizer so that every token range
can be checked by decoding it. Set MTP_SNAPSHOT to a model snapshot dir (tokenizer.json + chat_template.jinja) to
also run the structure checks with the real template and tokenizer.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trace_export as te  # noqa: E402
import trace_render as tr  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# synthetic pi sessions


class Builder:
    """Writes a pi-style session file; ids are sequential and parent-linked like pi's."""

    def __init__(self, path: Path, sid: str = "sess-1", cwd: str = "/work/proj"):
        self.path, self.n, self.parent = path, 0, None
        self.lines: list[str] = []
        self.raw({"type": "session", "version": 3, "id": sid, "timestamp": "2026-11-19T20:00:00.000Z", "cwd": cwd})

    def raw(self, d: dict) -> None:
        self.lines.append(json.dumps(d))

    def add(self, d: dict) -> str:
        self.n += 1
        eid = f"e{self.n:03d}"
        self.raw({**d, "id": eid, "parentId": self.parent, "timestamp": "2026-11-19T20:00:00.000Z"})
        self.parent = eid
        return eid

    def system(self, sections: dict | None = None, tools: list | None = None, removed: list | None = None, content: str = "") -> str:
        m: dict = {"role": "system", "content": content, "sections": sections or {"preamble": "You are a test agent.", "cwd": "<cwd>\n/work\n</cwd>"}}
        m["toolsAdded"] = tools if tools is not None else [TOOL_READ]
        if removed:
            m["toolsRemoved"] = removed
        return self.add({"type": "message", "message": m})

    def user(self, text: str) -> str:
        return self.add({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}], "timestamp": 1}})

    def assistant(self, text: str = "", thinking: str = "", calls: list | None = None, model: str = "qwen-trial117", stop: str | None = None) -> str:
        blocks: list = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking, "thinkingSignature": "reasoning"})
        if text:
            blocks.append({"type": "text", "text": text})
        for cid, name, args in calls or []:
            blocks.append({"type": "toolCall", "id": cid, "name": name, "arguments": args})
        stop = stop or ("toolUse" if calls else "stop")
        return self.add({"type": "message", "message": {"role": "assistant", "content": blocks, "api": "openai-completions", "provider": "spark-abl", "model": model, "stopReason": stop, "usage": {}, "timestamp": 1}})

    def tool_result(self, cid: str, text: str, name: str = "read", image: bool = False) -> str:
        blocks: list = [{"type": "text", "text": text}] if text else []
        if image:
            blocks.append({"type": "image", "data": "AAAA", "mimeType": "image/png"})
        return self.add({"type": "message", "message": {"role": "toolResult", "toolCallId": cid, "toolName": name, "content": blocks, "isError": False, "timestamp": 1}})

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n")
        return self.path


TOOL_READ = {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, "constrainedSampling": {"type": "json_schema", "strict": "prefer"}}
TOOL_BASH = {"name": "bash", "description": "Run a command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}


def simple_session(path: Path, sid: str = "s1", turns: int = 3, model: str = "qwen-trial117") -> Path:
    b = Builder(path, sid)
    b.add({"type": "model_change", "provider": "spark-abl", "modelId": model})
    b.add({"type": "thinking_level_change", "thinkingLevel": "medium"})
    b.system()
    b.user("hello there")
    for i in range(turns):
        b.assistant(text=f"looking {i}", thinking=f"think {i}", calls=[(f"c{i}", "read", {"path": f"/f{i}"})], model=model)
        b.tool_result(f"c{i}", f"content {i}")
    b.assistant(text="all done", thinking="final thought", model=model)
    return b.write()


# ---------------------------------------------------------------------------------------------------------------
# toy renderer: a Qwen-style template and a char-level tokenizer


MINI_TEMPLATE = """\
{%- if tools %}
{{- '<|im_start|>system\\n# Tools\\n' }}
{%- for t in tools %}{{- t | tojson }}{{- '\\n' }}{%- endfor %}
{%- if messages[0].role == 'system' %}{{- messages[0].content }}{%- endif %}
{{- '<|im_end|>\\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant\\n<think>\\n' + (message.reasoning_content | trim) + '\\n</think>\\n\\n' + message.content }}
{%- for tc in message.tool_calls or [] %}{{- '<tool_call>' + tc.function.name + (tc.function.arguments | tojson) + '</tool_call>' }}{%- endfor %}
{{- '<|im_end|>\\n' }}
{%- elif message.role == 'tool' %}
{%- if loop.previtem and loop.previtem.role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}
{{- '\\n<tool_response>\\n' + message.content + '\\n</tool_response>' }}
{%- if loop.last or loop.nextitem.role != 'tool' %}{{- '<|im_end|>\\n' }}{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}
"""

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    (d / "chat_template.jinja").write_text(MINI_TEMPLATE)
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern=Regex("[\\s\\S]"), behavior="isolated")
    tk.add_special_tokens(SPECIALS)
    tk.save(str(d / "tokenizer.json"))
    return tr.Renderer(d / "chat_template.jinja"), tr.Tok(d / "tokenizer.json")


def conv_from(messages: list[dict], tools: list[dict] | None = None, session: str = "s", carried: int = 0, source: str = "pi") -> dict:
    return {"session": session, "segment": 0, "source": source, "model": "qwen-trial117", "trigger": "start", "messages": messages, "tools": tools or [], "carried": carried, "system_source": "stored", "render": {}}


def asst(content=None, reasoning=None, calls=None, model="qwen-trial117") -> dict:
    m: dict = {"role": "assistant", "content": content, "model": model}
    if reasoning:
        m["reasoning_content"] = reasoning
    if calls:
        m["tool_calls"] = [{"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]
    return m


# ---------------------------------------------------------------------------------------------------------------
# parsing


def test_every_line_type_parses_and_bad_lines_are_counted(tmp_path):
    b = Builder(tmp_path / "a.jsonl")
    b.add({"type": "model_change", "provider": "p", "modelId": "m"})
    b.add({"type": "thinking_level_change", "thinkingLevel": "high"})
    b.system()
    b.user("hi")
    b.assistant(text="x", thinking="t", calls=[("c1", "read", {"path": "/a"})])
    b.tool_result("c1", "r", image=True)
    b.add({"type": "custom", "customType": "store", "data": {"set": {}}})
    b.add({"type": "compaction", "summary": "S", "firstKeptEntryId": "e004", "tokensBefore": 10, "details": {}, "fromHook": False, "usage": {},
           "systemMessage": {"role": "system", "content": "", "sections": {"preamble": "P"}, "toolsAdded": [TOOL_READ]}})
    b.add({"type": "context_edit", "targetId": "e005", "replacement": None})
    b.lines.append("{not json")
    b.lines.append(json.dumps({"type": "mystery", "id": "zz"}))
    b.lines.append(json.dumps({"type": "message", "id": "q", "message": {"role": "bashExecution"}}))
    s = te.load_pi_session(b.write())
    assert s.lines == 13 and s.unparsed == 3
    assert dict(s.type_counts) == {"session": 1, "model_change": 1, "thinking_level_change": 1, "message": 4, "custom": 1, "compaction": 1, "context_edit": 1}
    assert len(s.entries) == 9  # everything parsed except the header


def test_conversion_matches_pi_for_openai_completions(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    b.user("q")
    b.assistant(text="t1", thinking="a", calls=[("c1", "read", {"path": "/é", "n": {"b": 1, "a": [1, 2]}})])
    b.tool_result("c1", "")  # empty -> "(no tool output)"
    b.tool_result("c2", "", image=True)  # orphan result with image -> "(see attached image)", image dropped
    b.assistant(text="  ", thinking="  ", calls=None)  # dropped: no text, no tool calls
    b.assistant(text="partial", stop="aborted")  # dropped by transformMessages
    b.assistant(text="r", thinking="b1")
    segs = te.replay_pi_session(te.load_pi_session(b.write()))
    assert len(segs) == 1
    m = segs[0].messages
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert m[0]["content"] == "You are a test agent.\n\n<cwd>\n/work\n</cwd>"
    a = m[2]
    assert a["content"] == "t1" and a["reasoning_content"] == "a" and a["model"] == "qwen-trial117"
    assert a["tool_calls"][0]["function"]["arguments"] == '{"path":"/é","n":{"b":1,"a":[1,2]}}'  # JS JSON.stringify shape, key order kept
    assert m[3]["content"] == "(no tool output)" and m[4]["content"] == "(see attached image)" and m[4]["has_dropped_image"]
    assert segs[0].tools == [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": TOOL_READ["parameters"]}}]
    assert segs[0].info["assistant_error_or_aborted"] == 1 and segs[0].info["assistant_empty"] == 1


def test_missing_tool_result_is_synthesized_like_pi(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    b.user("q")
    b.assistant(text="t", calls=[("c1", "read", {"path": "/a"}), ("c2", "read", {"path": "/b"})])
    b.tool_result("c1", "only one")
    b.user("next")
    b.assistant(text="end")
    msgs = te.replay_pi_session(te.load_pi_session(b.write()))[0].messages
    assert [x["role"] for x in msgs] == ["system", "user", "assistant", "tool", "tool", "user", "assistant"]
    assert msgs[4]["tool_call_id"] == "c2" and msgs[4]["content"] == "No result provided"


# ---------------------------------------------------------------------------------------------------------------
# segmentation


def test_compaction_splits_into_system_summary_kept_tail(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    b.user("u1")  # e002
    b.assistant(text="a1")  # e003
    b.user("u2")  # e004  <- first kept
    b.assistant(text="a2")  # e005
    b.add({"type": "compaction", "summary": "SUM", "firstKeptEntryId": "e004", "tokensBefore": 5, "details": {}, "fromHook": False, "usage": {},
           "systemMessage": {"role": "system", "content": "", "sections": {"preamble": "NEW SYSTEM"}, "toolsAdded": [TOOL_BASH]}})
    b.user("u3")
    b.assistant(text="a3")
    s = te.load_pi_session(b.write())
    segs = te.replay_pi_session(s)
    assert [x.trigger for x in segs] == ["start", "compaction"]
    first, second = segs
    assert [m["content"] for m in first.messages] == ["You are a test agent.\n\n<cwd>\n/work\n</cwd>", "u1", "a1", "u2", "a2"]
    texts = [m["content"] for m in second.messages]
    assert texts[0] == "NEW SYSTEM"
    assert texts[1] == te.COMPACTION_SUMMARY_PREFIX + "SUM" + te.COMPACTION_SUMMARY_SUFFIX
    assert texts[2:] == ["u2", "a2", "u3", "a3"]
    assert second.carried == 4  # system, summary, u2, a2 are context only
    assert [t["function"]["name"] for t in second.tools] == ["bash"]
    disp = te.account_pi_messages(s, segs)
    assert sorted(disp.values()).count("used") == 6 and list(disp.values()).count("system_prompt") == 1


def test_context_edit_omit_and_replace_split_segments(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    b.user("u1")  # e002
    b.assistant(text="a1", calls=[("c1", "read", {"path": "/a"})])  # e003
    big = b.tool_result("c1", "HUGE OUTPUT")  # e004
    b.assistant(text="a2")  # e005
    b.add({"type": "context_edit", "targetId": big, "replacement": {"content": "short"}})
    b.user("u2")
    b.assistant(text="a3")
    b.add({"type": "context_edit", "targetId": "e003", "replacement": None})  # omit the tool-calling assistant
    b.user("u3")
    b.assistant(text="a4")
    s = te.load_pi_session(b.write())
    segs = te.replay_pi_session(s)
    assert [x.trigger for x in segs] == ["start", "context_edit", "context_edit"]
    assert [m["content"] for m in segs[1].messages if m["role"] == "tool"] == ["short"]
    # the omitted assistant is gone from the last segment (so is nothing else but it)
    assert "a1" not in [m["content"] for m in segs[2].messages]
    assert segs[2].carried == len(segs[2].messages) - 2  # only u3 and a4 are new
    disp = te.account_pi_messages(s, segs)
    assert all(v in ("used", "system_prompt") for v in disp.values())


def test_context_edit_on_a_message_outside_the_context_does_not_split(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    b.user("u1")
    b.assistant(text="a1")
    b.add({"type": "context_edit", "targetId": "does-not-exist", "replacement": None})
    b.user("u2")
    b.assistant(text="a2")
    assert len(te.replay_pi_session(te.load_pi_session(b.write()))) == 1


def test_mid_conversation_system_message_merges_sections_and_tools(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system(sections={"preamble": "P", "rules": "R1", "docs": "D"}, tools=[TOOL_READ, TOOL_BASH])
    b.user("u1")
    b.assistant(text="a1")
    b.system(sections={"rules": "R2", "docs": None, "mcp": "M"}, tools=[], removed=[{"name": "read"}])
    b.user("u2")
    b.assistant(text="a2")
    segs = te.replay_pi_session(te.load_pi_session(b.write()))
    assert [s.trigger for s in segs] == ["start", "system_update"]
    assert segs[0].messages[0]["content"] == "P\n\nR1\n\nD"
    assert segs[1].messages[0]["content"] == "P\n\nR2\n\nM"  # R1 replaced in place, D removed, M appended
    assert [t["function"]["name"] for t in segs[1].tools] == ["bash"]
    assert segs[1].carried == 3  # system, u1, a1


def test_session_without_stored_system_prompt_is_reconstructed(tmp_path):
    donor = tmp_path / "donor.jsonl"
    b = Builder(donor, sid="donor")
    b.system(sections={"preamble": "P", "project_context": "SECRET PROJECT", "cwd": "<cwd>\n/donor\n</cwd>"})
    b.user("u")
    b.assistant(text="a")
    b.write()
    old = tmp_path / "old.jsonl"
    b2 = Builder(old, sid="old", cwd="/old/cwd")
    b2.user("u")
    b2.assistant(text="a")
    b2.write()
    convs, stats = te.export_pi(str(tmp_path))
    o = next(c for c in convs if c["session"] == "old")
    assert o["system_source"] == "reconstructed"
    assert o["messages"][0]["content"] == "P\n\n<cwd>\n/old/cwd\n</cwd>"
    assert [t["function"]["name"] for t in o["tools"]] == ["read"]
    assert stats["sessions_with_reconstructed_system"] == 1


def test_export_accounts_for_every_message(tmp_path):
    simple_session(tmp_path / "a.jsonl", "a")
    p = tmp_path / "b.jsonl"
    b = Builder(p, "b")
    b.system()
    b.user("u")
    b.assistant(text="boom", stop="error")
    b.assistant(text="")  # empty
    b.assistant(text="ok")
    b.write()
    convs, stats = te.export_pi(str(tmp_path))
    assert stats["unparsed"] == 0
    assert sum(stats["disposition"].values()) == stats["messages_read"] == 8 + 6
    assert stats["disposition"]["skipped:assistant_error_or_aborted"] == 1 and stats["disposition"]["skipped:assistant_empty"] == 1


def test_off_branch_entries_are_skipped(tmp_path):
    p = tmp_path / "a.jsonl"
    b = Builder(p)
    b.system()
    u = b.user("u1")
    b.assistant(text="abandoned")
    b.parent = u  # a branch: the next entry hangs off u1 again
    b.assistant(text="kept")
    s = te.load_pi_session(b.write())
    assert len(s.off_path) == 1
    segs = te.replay_pi_session(s)
    assert [m["content"] for m in segs[0].messages][-1] == "kept"
    assert "skipped:off_active_branch" in te.account_pi_messages(s, segs).values()


# ---------------------------------------------------------------------------------------------------------------
# ZeroClaw


def make_zc_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("create table sessions (id integer primary key autoincrement, session_key text not null, role text not null, content text not null, created_at text not null)")
    rows = [
        ("k1", "assistant", json.dumps({"content": "orphan", "reasoning_content": "r", "tool_calls": []})),  # leading assistant: dropped
        ("k1", "user", "plain user text"),
        ("k1", "assistant", json.dumps({"content": "", "reasoning_content": "thinking", "tool_calls": [{"id": "t1", "name": "shell", "arguments": '{"cmd":"ls"}'}]})),
        ("k1", "tool", json.dumps({"content": "out", "tool_call_id": "t1", "attachments": []})),
        ("k1", "assistant", "final plain text"),
        ("k2", "user", "only a user"),
    ]
    for k, r, c in rows:
        con.execute("insert into sessions (session_key, role, content, created_at) values (?,?,?,?)", (k, r, c, "2026-11-15T00:00:00Z"))
    con.commit()
    con.close()


def test_zeroclaw_sessions(tmp_path):
    db = tmp_path / "s.db"
    make_zc_db(db)
    convs, stats = te.zeroclaw_segments(str(db), "ct")
    assert stats["rows_read"] == 6 and stats["unparsed"] == 0 and stats["skipped"]["leading_non_user"] == 1
    assert stats["skipped"]["session_without_assistant_turn"] == 1
    assert len(convs) == 1
    m = convs[0]["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "tool", "assistant"]
    assert m[1]["tool_calls"][0]["function"] == {"name": "shell", "arguments": '{"cmd":"ls"}'} and m[1]["reasoning_content"] == "thinking"
    assert m[2]["tool_call_id"] == "t1" and convs[0]["tools"] == [] and convs[0]["source"] == "zeroclaw"
    assert "k1" not in convs[0]["session"]  # the session key (it contains an account id) is hashed


# ---------------------------------------------------------------------------------------------------------------
# rendering: loss-mask ranges


def dec(tok, ids) -> str:
    """Toy tokenizer: one token per character (or special token), so decoding is a join."""
    return "".join(tok.tok.id_to_token(i) for i in ids)


def assist_text(tok, rd, j) -> str:
    s, e = rd.assist[j]
    return dec(tok, rd.ids[s:e])


def test_assist_ranges_cover_exactly_what_the_model_generates(toy):
    r, tok = toy
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do it"},
        asst("looking", "I should read", [("c1", "read", {"path": "/a"})]),
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        {"role": "tool", "tool_call_id": "c2", "content": "second"},
        asst(None, "now answer", None),
        {"role": "user", "content": "thanks"},
        asst("welcome", None, None),
    ]
    msgs[5]["content"] = "42"
    conv = conv_from(msgs, [{"type": "function", "function": {"name": "read", "parameters": {}}}])
    rd = tr.render_segment(conv, r, tok)
    assert assist_text(tok, rd, 2) == 'I should read\n</think>\n\nlooking<tool_call>read{"path": "/a"}</tool_call><|im_end|>'
    assert assist_text(tok, rd, 5) == "now answer\n</think>\n\n42<|im_end|>"
    assert assist_text(tok, rd, 7) == "\n</think>\n\nwelcome<|im_end|>"  # no reasoning: the template's empty think block
    # ids are exactly the tokenization of the rendered text
    assert dec(tok, rd.ids) == rd.text
    # the span before each assistant fragment is the prefix render (what the model was prompted with)
    for j, (cs, ce) in rd.spans.items():
        prompt = r.render(msgs[:j], conv["tools"], True)
        assert rd.text[:cs] + tr.ASSISTANT_HEADER == prompt
    # nothing outside an assistant turn's generated part is in the loss mask: the role header, the generation
    # prompt (`<think>\n`), the trailing newline and every user/tool token
    inside = {t for s, e in rd.assist.values() for t in range(s, e)}
    for j, (cs, ce) in rd.spans.items():
        s, e = rd.assist[j]
        assert dec(tok, rd.ids[s - 1 : s]) == "\n"  # the prompt side ends in the think newline
        assert rd.text[cs:ce].startswith(tr.ASSISTANT_HEADER) and rd.text[ce - 1] == "\n"
        assert (e not in inside) or any(s2 == e for s2, _ in rd.assist.values())
    assert 0 not in inside and len(rd.ids) - 1 not in inside


def test_fast_spans_equal_exact_spans_even_when_a_tool_output_repeats_a_fragment(toy):
    r, tok = toy
    a1 = asst("same", "same thought", None)
    frag = tr._fragment(r, a1, {})
    msgs = [{"role": "user", "content": "q"}, asst("first", "t", [("c1", "read", {"path": "/p"})]), {"role": "tool", "tool_call_id": "c1", "content": "cat " + frag}, a1, {"role": "user", "content": "again"}, asst("last", "z", None)]
    conv = conv_from(msgs)
    rd = tr.render_segment(conv, r, tok, spot_checks=10)  # verify every assistant: falls back to exact spans if needed
    text = r.render(msgs, None, False)
    assert rd.spans == tr.exact_spans(msgs, None, {}, r, text)


@pytest.mark.skipif(not os.environ.get("MTP_SNAPSHOT"), reason="set MTP_SNAPSHOT to the model snapshot dir")
def test_real_template_and_tokenizer_assist_ranges():
    snap = Path(os.environ["MTP_SNAPSHOT"])
    r, tok = tr.Renderer(os.environ.get("MTP_TEMPLATE") or snap / "chat_template.jinja"), tr.Tok(snap / "tokenizer.json")
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do it"},
        asst("looking é", "I should read", [("c1", "read", {"path": "/ä", "n": {"b": 1, "a": [1]}})]),
        {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        asst("done", "ok", None),
    ]
    conv = conv_from(msgs, [{"type": "function", "function": {"name": "read", "description": "d", "parameters": {"type": "object"}}}])
    conv["render"] = {"reasoning_effort": "medium"}
    rd = tr.render_segment(conv, r, tok)

    def at(j):
        a, b = rd.assist[j]
        return tok.decode(rd.ids[a:b])

    assert at(2).startswith("I should read\n</think>\n\nlooking é\n\n<tool_call>\n<function=read>")
    assert at(2).endswith("</function>\n</tool_call><|im_end|>")
    assert at(4) == "ok\n</think>\n\ndone<|im_end|>"
    assert '<parameter=n>\n{"b": 1, "a": [1]}\n</parameter>' in rd.text and "/ä" in rd.text  # non-ASCII kept, key order kept


def test_vllm_request_names_the_reasoning_field_reasoning():
    conv = conv_from([{"role": "user", "content": "q"}, asst("a", "why", [("c", "read", {"path": "/x"})]), {"role": "tool", "tool_call_id": "c", "name": "read", "content": "r"}], [{"type": "function", "function": {"name": "read"}}])
    body = tr.vllm_request(conv, 3, False)
    a = body["messages"][1]
    assert a["reasoning"] == "why" and "reasoning_content" not in a and "model" not in a
    assert a["tool_calls"][0]["function"]["arguments"] == '{"path": "/x"}'
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"} and body["add_generation_prompt"] is False
    assert "name" not in body["messages"][2]


# ---------------------------------------------------------------------------------------------------------------
# selection, split, end to end


def fake_rendered(n_tokens: int, assists: dict[int, tuple[int, int]]) -> tr.Rendered:
    return tr.Rendered("", list(range(n_tokens)), list(range(n_tokens)), {}, assists, {})


def test_select_windows_covers_without_heavy_overlap():
    # ten assistant turns of 200 tokens, 300 apart: windows (4096) would all overlap; picks must stay spread out
    assists = {i: (1000 + 300 * i, 1200 + 300 * i) for i in range(10)}
    rd = fake_rendered(5000, assists)
    picks = tr.select_windows(rd, list(assists))
    assert picks == [9]  # one window ending at the last turn already holds all of them
    # two clusters 6000 tokens apart need two windows
    two = {0: (500, 700), 1: (800, 1000), 2: (7000, 7200), 3: (7300, 7500)}
    assert tr.select_windows(fake_rendered(8000, two), list(two)) == [1, 3]
    # long distances: every turn gets its own window and nothing is lost
    far = {i: (5000 * i + 100, 5000 * i + 400) for i in range(5)}
    rd2 = fake_rendered(5000 * 5, far)
    assert tr.select_windows(rd2, list(far), max_tokens=30000) == [0, 1, 2, 3, 4]
    # turns ending beyond max_tokens are skipped
    assert tr.select_windows(rd2, list(far), max_tokens=12000) == [0, 1, 2]


def test_split_is_session_disjoint_stratified_and_deterministic():
    convs = []
    for i in range(40):
        convs += [conv_from([{"role": "user", "content": "u"}, asst("a")], session=f"pi-{i}"), conv_from([{"role": "user", "content": "u"}, asst("a")], session=f"pi-{i}")]
    for i in range(10):
        convs.append(conv_from([{"role": "user", "content": "u"}, asst("a", model="kimi")], session=f"pi-other-{i}"))
    for i in range(9):
        convs.append(conv_from([{"role": "user", "content": "u"}, asst("a")], session=f"zc-{i}", source="zeroclaw"))
    a, b = tr.split_sessions(convs), tr.split_sessions(convs)
    assert a == b
    assert set(a.values()) == {"train", "heldout"}
    by_src = {}
    for c in convs:
        by_src.setdefault(c["source"], set()).add(c["session"])
    for src, sids in by_src.items():
        held = sum(a[s] == "heldout" for s in sids)
        assert 0.1 <= held / len(sids) <= 0.25 and held >= 1, (src, held, len(sids))
    # segments of one session are never separated
    assert all(a[c["session"]] == a["pi-0"] for c in convs if c["session"] == "pi-0")
    assert tr.split_sessions(convs, seed=1) != a


def test_build_end_to_end(toy, tmp_path):
    r, tok = toy
    convs = []
    for i in range(12):
        m = [{"role": "system", "content": "SYS"}, {"role": "user", "content": f"task {i}"}]
        for k in range(6):
            m.append(asst(f"step {k} " + "w" * 150, "r" * 100, [(f"c{k}", "read", {"path": f"/f{k}"})], model="qwen-trial117" if i % 2 else "kimi-for-coding"))
            m.append({"role": "tool", "tool_call_id": f"c{k}", "content": "o" * 120})
        m.append(asst("done " + "z" * 60, "fin"))
        convs.append(conv_from(m, [{"type": "function", "function": {"name": "read", "parameters": {}}}], session=f"s{i}", carried=0))
    convs.append(conv_from([{"role": "user", "content": "zc question"}, asst("zc answer")], session="zc-1", source="zeroclaw"))
    out = tmp_path / "out"
    stats = tr.build(convs, r, tok, out, n_pi_prompts=5)
    assert stats["session_overlap_train_heldout"] == []
    train, held = list(tr.read_jsonl(out / "seqs-train.jsonl")), list(tr.read_jsonl(out / "seqs-heldout.jsonl"))
    assert train and held
    assert {x["session"] for x in train}.isdisjoint({x["session"] for x in held})
    for row in train + held:
        n = len(row["tokens"])
        a, b = row["window"]
        assert n <= tr.MAX_TOKENS and b == n and 0 <= a < b and b - a <= tr.WINDOW
        assert all(0 <= s < e <= n for s, e in row["assist"]) and row["assist"][-1][1] == n
        assert dec(tok, row["tokens"][row["assist"][-1][0] : n]).endswith("<|im_end|>")
        assert len(row["assist_model"]) == len(row["assist"]) and row["gen_start"] == row["assist"][-1][0]
    prompts = list(tr.read_jsonl(out / "prompts-heldout.jsonl"))
    assert prompts
    hdr = tok.ids(tr.ASSISTANT_HEADER)
    for p in prompts:
        assert p["prompt_tokens"][-len(hdr) :] == hdr and len(p["prompt_tokens"]) <= tr.MAX_PROMPT
        assert p["source"] in ("pi", "zeroclaw") and "in_train" in p
    assert any(p["source"] == "zeroclaw" for p in prompts)
    assert os.stat(out / "seqs-train.jsonl").st_mode & 0o777 == 0o600
    assert stats["samples"]["train/pi"] + stats["samples"].get("heldout/pi", 0) == len([x for x in train + held if x["source"] == "pi"])


def test_images_in_context_exclude_the_sample(toy, tmp_path):
    r, tok = toy
    m = [{"role": "user", "content": "u"}, asst("a" * 80, "t" * 80, [("c", "read", {"path": "/i"})]), {"role": "tool", "tool_call_id": "c", "content": "(see attached image)", "has_dropped_image": True}, asst("b" * 80, "t" * 80)]
    conv = conv_from(m, session="img")
    assert not tr.contains_dropped_image(conv, 2) and tr.contains_dropped_image(conv, 4)


def test_empty_reasoning_turn_is_tokenized_as_prompt_plus_completion(toy):
    """With an empty think block the completion starts with "\n"; a tokenizer that merges "\n\n" would fuse it with the
    prompt's newline. The toy tokenizer never merges, so use a merging one for this check."""
    r, _ = toy
    tk = Tokenizer(models.WordLevel({"a": 0, "\n": 1, "\n\n": 2, "[UNK]": 3}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(pattern=Regex("\n+|[^\n]"), behavior="isolated")
    tk.add_special_tokens(SPECIALS + ["x", "u", "s", "e", "r", "i", "t", "n", "g", "d", "o", "w", "m", "c", "l", "p"])
    tok = tr.Tok.__new__(tr.Tok)
    tok.tok = tk
    conv = conv_from([{"role": "user", "content": "x"}, asst("a", None, None)])
    rd = tr.render_segment(conv, r, tok)
    smp = tr.make_sample(conv, rd, 1, tok)
    assert smp["boundary_merged"]
    n = tk.token_to_id
    # prompt ends with the single newline the model was given; the completion's first token is a single newline too
    assert smp["tokens"][smp["gen_start"] - 1] == n("\n") and smp["tokens"][smp["gen_start"]] == n("\n")
    assert smp["assist"][-1] == [smp["gen_start"], len(smp["tokens"])] and smp["tokens"][-1] == n("<|im_end|>")
    assert rd.ids[smp["gen_start"] - 1] == n("\n\n")  # one tokenization of the whole text would have fused them
