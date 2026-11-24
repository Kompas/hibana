"""capfmt round trip, crash/resume and checksums, and the capture loop on a fake engine (CPU, synthetic data)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capfmt  # noqa: E402
import mtp_capture  # noqa: E402

H, K = capfmt.HIDDEN, capfmt.TOPK


def synthetic(i: int, n: int) -> dict:
    """One sequence's rows, deterministic in i: random bf16 bits (NaN patterns included), ids, logits."""
    rng = np.random.default_rng(1000 + i)
    h = rng.integers(0, 2**16, size=(n, H), dtype=np.uint16)
    return {"id": f"seq-{i:03d}", "source": ["pi", "claude", "codex"][i % 3], "context_rows": min(1024, n // 2),
            "tokens": rng.integers(0, 248320, size=n, dtype=np.int32), "positions": np.arange(5000 + i, 5000 + i + n, dtype=np.int32),
            "h": h, "topk_ids": rng.integers(0, 248320, size=(n, K), dtype=np.int32),
            "topk_logits": np.sort(rng.standard_normal((n, K)).astype(np.float32) * 8, axis=1)[:, ::-1].copy(),
            "assist": rng.integers(0, 2, size=n, dtype=np.uint8), "extra": {"i": i}}


def add(writer: capfmt.ShardWriter, s: dict) -> None:
    writer.add(s["id"], s["source"], s["context_rows"], s["tokens"], s["positions"], s["h"], s["topk_ids"], s["topk_logits"],
               s["assist"], extra=s["extra"])


def same(read: dict, s: dict) -> bool:
    return (read["id"] == s["id"] and read["source"] == s["source"] and read["context_rows"] == s["context_rows"]
            and read["extra"] == s["extra"]
            and all(np.array_equal(read[f], s[f]) and read[f].dtype == s[f].dtype for f in capfmt.FIELDS))


LENGTHS = [37, 300, 4096, 1, 2048, 777, 64, 1500, 3000, 129]


def test_round_trip_bit_identical(tmp_path):
    seqs = [synthetic(i, n) for i, n in enumerate(LENGTHS)]
    w = capfmt.ShardWriter(tmp_path / "train", shard_bytes=60 << 20, log=None)
    for s in seqs:
        add(w, s)
    w.close()
    shards = capfmt.list_shards(tmp_path / "train")
    full = 0
    for p in shards:  # the trainer reads one context_rows per shard: the minimum of its windows', the cap when all are full
        sh = capfmt.Shard(p)
        per = sh.meta["context_rows_per_seq"]
        assert sh.meta["context_rows"] == min(per) and (all(c == 1024 for c in per) or all(c < 1024 for c in per))
        full += sh.meta["context_rows"] == 1024
    assert full >= 1
    assert len(shards) >= 3  # several shards: no sequence is split, each closes once past 60 MiB
    back = list(capfmt.iter_sequences(tmp_path / "train"))
    assert sorted(b["id"] for b in back) == sorted(s["id"] for s in seqs)  # each once (full and short windows shard apart)
    by = {b["id"]: b for b in back}
    assert all(same(by[s["id"]], s) for s in seqs)
    v = capfmt.verify(tmp_path / "train")
    assert v["ok"], v
    assert v["rows"] == sum(LENGTHS) and v["sequences"] == len(seqs)
    assert v["bytes"] >= sum(LENGTHS) * capfmt.row_bytes()


def test_standard_safetensors_reader(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    s = synthetic(7, 513)
    w = capfmt.ShardWriter(tmp_path / "x", log=None)
    add(w, s)
    w.close()
    path = capfmt.list_shards(tmp_path / "x")[0]
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        h = f.get_tensor("h")
        assert h.dtype == torch.bfloat16 and tuple(h.shape) == (513, H)
        assert np.array_equal(h.view(torch.int16).numpy().view(np.uint16), s["h"])
        assert f.get_tensor("topk_logits").dtype == torch.float32
        assert f.get_tensor("seq_starts").tolist() == [0, 513]
        assert f.get_tensor("assist").dtype == torch.uint8
    assert meta["format"] == capfmt.FORMAT and json.loads(meta["seq_ids"]) == [s["id"]]
    assert json.loads(meta["context_rows"]) == s["context_rows"] and json.loads(meta["context_rows_per_seq"]) == [s["context_rows"]]
    # bf16 widening matches torch's
    x = torch.randn(1000) * 100
    bits = x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(capfmt.bf16_to_float32(bits), x.to(torch.bfloat16).float().numpy())
    assert np.array_equal(capfmt.float32_to_bf16(x.numpy()), bits)


def test_resume_after_crash(tmp_path, monkeypatch):
    d = tmp_path / "heldout"
    seqs = [synthetic(i, n) for i, n in enumerate(LENGTHS)]

    # run 1 dies inside the rename of its second shard: one finished shard, one fsynced .tmp, a buffer lost
    w = capfmt.ShardWriter(d, shard_bytes=90 << 20, log=None)
    real_replace, calls = os.replace, {"n": 0}

    def dying_replace(a, b):
        calls["n"] += 1
        if calls["n"] == 2:
            raise SystemExit("simulated crash")
        return real_replace(a, b)
    monkeypatch.setattr(capfmt.os, "replace", dying_replace)
    with pytest.raises(SystemExit):
        for s in seqs:
            add(w, s)
        w.flush()
    monkeypatch.setattr(capfmt.os, "replace", real_replace)
    assert len(capfmt.list_shards(d)) == 1 and list(d.glob("*.tmp"))
    first = set(capfmt.Shard(capfmt.list_shards(d)[0]).seq_ids)

    # run 2 dies between a shard's rename and its checksum line, and leaves a torn line
    w = capfmt.ShardWriter(d, shard_bytes=90 << 20, log=None)
    assert not list(d.glob("*.tmp")) and w.finished == first
    real_sum = capfmt._append_sum

    def dying_sum(*a):
        raise SystemExit("simulated crash")
    monkeypatch.setattr(capfmt, "_append_sum", dying_sum)
    with pytest.raises(SystemExit):
        for s in seqs:
            if s["id"] not in w.finished:
                add(w, s)
        w.flush()
    monkeypatch.setattr(capfmt, "_append_sum", real_sum)
    with open(d / capfmt.SUMS, "a") as f:
        f.write("deadbeef  shard-0")  # torn write
    assert len(capfmt.list_shards(d)) == 2 and not capfmt.verify(d)["ok"]  # shard 2 has no checksum yet

    # run 3 reconciles the checksum, skips both shards' sequences and finishes the rest
    w = capfmt.ShardWriter(d, shard_bytes=90 << 20, log=None)
    assert capfmt.verify(d)["ok"]
    todo = [s for s in seqs if s["id"] not in w.finished]
    assert 0 < len(todo) < len(seqs)
    for s in todo:
        add(w, s)
    w.close()
    v = capfmt.verify(d)
    assert v["ok"], v
    back = {b["id"]: b for b in capfmt.iter_sequences(d)}
    assert sorted(back) == sorted(s["id"] for s in seqs)  # every sequence exactly once
    assert all(same(back[s["id"]], s) for s in seqs)
    with pytest.raises(ValueError):
        add(w, seqs[0])  # already captured


def test_checksum_detects_corruption(tmp_path):
    w = capfmt.ShardWriter(tmp_path / "s", log=None)
    add(w, synthetic(1, 200))
    w.close()
    path = capfmt.list_shards(tmp_path / "s")[0]
    data = bytearray(path.read_bytes())
    data[-100] ^= 1
    path.write_bytes(bytes(data))
    v = capfmt.verify(tmp_path / "s")
    assert not v["ok"] and any("checksum mismatch" in p for p in v["problems"])


def test_add_validates(tmp_path):
    w = capfmt.ShardWriter(tmp_path / "v", log=None)
    s = synthetic(2, 10)
    with pytest.raises(ValueError):
        w.add("a", "x", 0, s["tokens"], s["positions"], s["h"][:, :100], s["topk_ids"], s["topk_logits"], s["assist"])
    with pytest.raises(ValueError):
        w.add("a", "x", 0, s["tokens"], s["positions"], s["h"].astype(np.int16), s["topk_ids"], s["topk_logits"], s["assist"])
    with pytest.raises(ValueError):
        w.add("a", "x", 10, s["tokens"], s["positions"], s["h"], s["topk_ids"], s["topk_logits"], s["assist"])


# ---------------------------------------------------------------- inputs and the capture loop

def test_assist_mask_and_spans():
    # tokens 0..9, assistant tokens 4..6 and 9: row i is 1 when token i + 1 is assistant
    m = mtp_capture.assist_mask([[4, 7], [9, 10]], 2, 10, 10)
    assert m.tolist() == [0, 1, 1, 1, 0, 0, 1, 0]  # rows 2..9 -> tokens 3..10 (10 is past the end)
    opener, closer = [1, 2, 3], 9
    ids = [5, 1, 2, 3, 7, 7, 9, 5, 1, 2, 3, 8]
    assert mtp_capture.assistant_spans(ids, opener, closer) == [[4, 7], [11, 12]]
    assert mtp_capture.context_rows_for(4096, 1024) == 1024 and mtp_capture.context_rows_for(1000, 1024) == 500


def fake_seqs(n: int) -> list[dict]:
    rng = np.random.default_rng(5)
    out = []
    for i in range(n):
        total = int(rng.integers(50, 400))
        a = int(rng.integers(0, total - 10))
        out.append({"id": f"s{i}", "session": f"sess{i // 3}", "source": ["pi", "claude"][i % 2], "model": "qwen-trial117",
                    "tokens": rng.integers(0, 1000, size=total).tolist(), "assist": [[a + 2, total]], "window": [a, total]})
    return out


def fake_engine(seq):
    """Deterministic rows from the sequence's tokens (stands in for EngineCapture)."""
    a, b = seq["window"]
    t = np.asarray(seq["tokens"][a:b], dtype=np.int64)
    h = ((t[:, None] * 31 + np.arange(H)[None, :]) % 65536).astype(np.uint16)
    ids = ((t[:, None] + np.arange(K)[None, :]) % 1000).astype(np.int32)
    return {"h": h, "topk_ids": ids, "topk_logits": -np.arange(K, dtype=np.float32)[None, :].repeat(b - a, 0),
            "timing": {"cached": 0, "context_tokens": a, "context_s": 0.001, "window_s": 0.001, "seconds": 0.002}}


def test_run_capture_caps_and_resume(tmp_path):
    seqs = fake_seqs(12)
    d = tmp_path / "train"
    rb = capfmt.row_bytes()
    cap = sum(s["window"][1] - s["window"][0] for s in seqs[:5]) * rb + (64 << 10)  # + shard headers; the 6th needs > 200 KB
    w = capfmt.ShardWriter(d, shard_bytes=2 << 20, log=None)
    r = mtp_capture.run_capture(seqs, w, fake_engine, deadline=float("inf"), max_bytes=cap, context_rows=1024, logf=lambda m: None)
    assert r["reason"] == "max-bytes" and r["captured"] == 5 and capfmt.verify(d)["ok"]

    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3
    w = capfmt.ShardWriter(d, shard_bytes=2 << 20, log=None)
    r = mtp_capture.run_capture(seqs, w, fake_engine, deadline=float("inf"), max_bytes=float("inf"), context_rows=1024,
                                stop=stop, progress=d / "progress.jsonl", logf=lambda m: None)
    assert r["reason"] == "signal" and r["captured"] == 3 and r["skipped_finished"] == 5

    w = capfmt.ShardWriter(d, shard_bytes=2 << 20, log=None)
    r = mtp_capture.run_capture(seqs, w, fake_engine, deadline=0.0, max_bytes=float("inf"), context_rows=1024, logf=lambda m: None)
    assert r["reason"] == "max-minutes" and r["captured"] == 0

    w = capfmt.ShardWriter(d, shard_bytes=2 << 20, log=None)
    r = mtp_capture.run_capture(seqs, w, fake_engine, deadline=float("inf"), max_bytes=float("inf"), context_rows=1024, logf=lambda m: None)
    assert r["reason"] == "complete" and r["captured"] == 4 and r["skipped_finished"] == 8
    assert r["top1_agree_assist_mean"] is not None and 0 <= r["top1_agree_assist_mean"] <= 1
    assert capfmt.verify(d)["ok"]
    back = {b["id"]: b for b in capfmt.iter_sequences(d)}
    assert sorted(back) == sorted(s["id"] for s in seqs)
    for s in seqs:
        a, b = s["window"]
        got, want = back[s["id"]], fake_engine(s)
        assert got["tokens"].tolist() == s["tokens"][a:b]
        assert got["positions"].tolist() == list(range(a, b))
        assert np.array_equal(got["h"], want["h"]) and np.array_equal(got["topk_ids"], want["topk_ids"])
        assert got["assist"].tolist() == mtp_capture.assist_mask(s["assist"], a, b, len(s["tokens"])).tolist()
        assert got["context_rows"] == min(1024, (b - a) // 2)
        assert got["extra"]["session"] == s["session"] and got["extra"]["model"] == s["model"]
    assert len((d / "progress.jsonl").read_text().splitlines()) == 3


def test_prepare_validates_and_interleaves(tmp_path):
    seqs = fake_seqs(6)
    seqs.append({"id": "bad", "source": "pi", "tokens": [1, 2, 3], "window": [2, 9]})
    seqs.append(dict(seqs[0]))  # duplicate id
    f = tmp_path / "seqs.jsonl"
    f.write_text("".join(json.dumps(s) + "\n" for s in seqs))
    ns = type("A", (), {"general": None, "seqs": str(f), "max_window": 4096, "max_tokens": 32768, "order": "interleave", "limit": 0})
    good, counts = mtp_capture.prepare(ns)
    assert counts["sequences"] == 6 and sum(counts["rejected"].values()) == 2
    assert [s["source"] for s in good] == ["pi", "claude"] * 3
    assert counts["bytes"] == counts["window_rows"] * capfmt.row_bytes()
