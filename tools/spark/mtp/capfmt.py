#!/usr/bin/env python3
"""The MTP-head capture format: shards of the target's hidden stream and top-k next-token logits over token windows.

A capture set is a directory `<root>/<set>/` of shards `shard-%05d.safetensors` plus `SHA256SUMS` (sha256sum format,
one line per finished shard). Each shard is a standard safetensors file (readable with `safetensors.safe_open`) with,
for N rows over S sequences:

    tokens       int32    [N]          window tokens, the sequences concatenated
    positions    int32    [N]          absolute position of each token in its conversation
    h            bfloat16 [N, 10240]   the target's `forward()["norm"]` at row i (the hyper stream after token i)
    topk_ids     int32    [N, 8]       the target's top-8 next-token ids at row i (predicting token i + 1)
    topk_logits  float32  [N, 8]       their logits, descending
    assist       uint8    [N]          1 if token i + 1 is assistant-produced
    seq_starts   int64    [S + 1]      row offsets: sequence j is rows seq_starts[j] .. seq_starts[j + 1]

`__metadata__` (safetensors string values, one JSON value per key): `format` ("mtp-capture-v1"), `set`, `hidden`,
`topk`, `seq_ids` and `sources` (lists, one per sequence), `context_rows` (one int for the shard: every window's
first rows that many are context only, not training targets; the minimum of the shard's per-sequence values, so
it never hides a row a window may train on), `context_rows_per_seq` (the per-sequence values: min(cap, window / 2)),
and `extra` (per sequence: free-form numbers such as the capture's timing, the session and the model).

Writes are atomic (a shard is written to `.tmp`, fsynced and renamed) and resumable: a writer opened on an existing
set skips every seq_id a finished shard holds; sequences buffered when a run died are captured again. No numpy
bfloat16 exists, so the reader returns `h` as uint16 bit patterns (`bf16_to_float32` widens them); torch users can
open the shards with `safetensors.torch` directly.

    python capfmt.py verify DIR      # every shard against SHA256SUMS and its own structure
    python capfmt.py summary DIR     # shards, sequences, rows, bytes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

FORMAT = "mtp-capture-v1"
HIDDEN = 10240  # Flash-Next: hc (4) x hidden (2560)
TOPK = 8
SUMS = "SHA256SUMS"
SHARD_RE = re.compile(r"^shard-(\d{5})\.safetensors$")
# field: (safetensors dtype, numpy dtype of the stored bits, trailing shape)
FIELDS = {
    "tokens": ("I32", np.int32, ()),
    "positions": ("I32", np.int32, ()),
    "h": ("BF16", np.uint16, ("hidden",)),
    "topk_ids": ("I32", np.int32, ("topk",)),
    "topk_logits": ("F32", np.float32, ("topk",)),
    "assist": ("U8", np.uint8, ()),
}
JSON_KEYS = ("seq_ids", "sources", "context_rows", "context_rows_per_seq", "extra")
NP_OF = {"I32": np.int32, "I64": np.int64, "U8": np.uint8, "F32": np.float32, "BF16": np.uint16}


def row_bytes(hidden: int = HIDDEN, topk: int = TOPK) -> int:
    """Bytes one captured row takes on disk."""
    return 4 + 4 + 2 * hidden + 4 * topk + 4 * topk + 1


def bf16_to_float32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32, exactly."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns, round to nearest even (as torch's .to(torch.bfloat16)); for tests and tools."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    out = (rounded >> 16).astype(np.uint16)
    nan = np.isnan(x)
    out[nan] = 0x7FC0
    return out


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------- reading

class Shard:
    """One shard, memory-mapped: `arrays[name]` are numpy views (h as uint16 bf16 bits), `meta` the parsed metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        self.data_start = 8 + n
        raw = header.pop("__metadata__", {})
        self.raw_meta = raw
        self.meta = {k: (json.loads(v) if k in JSON_KEYS else v) for k, v in raw.items()}
        size = self.path.stat().st_size
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        self.arrays: dict[str, np.ndarray] = {}
        for name, info in header.items():
            b, e = info["data_offsets"]
            if not (0 <= b <= e and self.data_start + e <= size):
                raise ValueError(f"{self.path.name}: {name} lies outside the file")
            dt = NP_OF[info["dtype"]]
            self.arrays[name] = self._map[self.data_start + b:self.data_start + e].view(dt).reshape(info["shape"])
        self.header = header

    @property
    def seq_ids(self) -> list[str]:
        return list(self.meta.get("seq_ids", []))

    def __len__(self) -> int:
        return len(self.seq_ids)

    def context_rows_of(self, j: int) -> int:
        per = self.meta.get("context_rows_per_seq")
        return int(per[j]) if per is not None else int(self.meta.get("context_rows", 0))

    def sequence(self, j: int) -> dict:
        """Sequence j: its id, source, context_rows, extra and every field's rows (views)."""
        s = self.arrays["seq_starts"]
        lo, hi = int(s[j]), int(s[j + 1])
        out = {"id": self.meta["seq_ids"][j], "source": self.meta["sources"][j], "context_rows": self.context_rows_of(j),
               "extra": (self.meta.get("extra") or [None] * len(self))[j]}
        out.update({name: self.arrays[name][lo:hi] for name in FIELDS})
        return out

    def check(self) -> list[str]:
        """Structural problems (empty when the shard is well formed)."""
        problems = []
        if self.meta.get("format") != FORMAT:
            problems.append(f"format {self.meta.get('format')!r}")
        need = set(FIELDS) | {"seq_starts"}
        if set(self.arrays) != need:
            problems.append(f"tensors {sorted(self.arrays)}")
            return problems
        s = self.arrays["seq_starts"]
        n = int(self.arrays["tokens"].shape[0])
        hidden, topk = int(self.meta.get("hidden", HIDDEN)), int(self.meta.get("topk", TOPK))
        for name, (_, _, trailing) in FIELDS.items():
            want = (n,) + tuple(hidden if t == "hidden" else topk for t in trailing)
            if tuple(self.arrays[name].shape) != want:
                problems.append(f"{name} shape {self.arrays[name].shape} != {want}")
        k = len(self.seq_ids)
        if s.shape != (k + 1,) or (k and (s[0] != 0 or s[-1] != n or np.any(np.diff(s) <= 0))):
            problems.append("seq_starts inconsistent")
        for key in ("sources", "context_rows_per_seq"):
            if len(self.meta.get(key, [])) != k:
                problems.append(f"{key} length")
        per = self.meta.get("context_rows_per_seq") or [0]
        if not isinstance(self.meta.get("context_rows"), int) or self.meta["context_rows"] != min(per):
            problems.append("context_rows is not the minimum of context_rows_per_seq")
        if len(set(self.seq_ids)) != k:
            problems.append("duplicate seq_ids")
        return problems


def list_shards(set_dir: str | Path) -> list[Path]:
    d = Path(set_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if SHARD_RE.match(p.name))


def read_sums(set_dir: Path) -> dict[str, str]:
    """name -> sha256 from SHA256SUMS (malformed lines, e.g. a torn last write, are ignored)."""
    out = {}
    f = Path(set_dir) / SUMS
    if f.is_file():
        for line in f.read_text(errors="replace").splitlines():
            m = re.fullmatch(r"([0-9a-f]{64})  (shard-\d{5}\.safetensors)", line.strip())
            if m:
                out[m.group(2)] = m.group(1)
    return out


def _append_sum(set_dir: Path, name: str, digest: str) -> None:
    f = Path(set_dir) / SUMS
    prefix = ""
    if f.is_file() and f.stat().st_size:
        with open(f, "rb") as r:
            r.seek(-1, os.SEEK_END)
            if r.read(1) != b"\n":
                prefix = "\n"
    with open(f, "a") as w:
        w.write(f"{prefix}{digest}  {name}\n")
        w.flush()
        os.fsync(w.fileno())


def iter_sequences(set_dir: str | Path):
    """Every captured sequence of a set, shard by shard (dicts as `Shard.sequence`)."""
    for path in list_shards(set_dir):
        shard = Shard(path)
        for j in range(len(shard)):
            yield shard.sequence(j)


def verify(set_dir: str | Path) -> dict:
    """Check every shard against SHA256SUMS and its own structure, and that no seq_id repeats across shards."""
    d = Path(set_dir)
    sums = read_sums(d)
    problems, seen, rows, seqs = [], set(), 0, 0
    shards = list_shards(d)
    for path in shards:
        want = sums.get(path.name)
        if want is None:
            problems.append(f"{path.name}: no checksum")
        elif sha256_file(path) != want:
            problems.append(f"{path.name}: checksum mismatch")
        try:
            shard = Shard(path)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            problems.append(f"{path.name}: unreadable ({error})")
            continue
        problems += [f"{path.name}: {p}" for p in shard.check()]
        dup = seen & set(shard.seq_ids)
        if dup:
            problems.append(f"{path.name}: {len(dup)} seq_ids also in an earlier shard")
        seen |= set(shard.seq_ids)
        rows += int(shard.arrays["tokens"].shape[0]) if "tokens" in shard.arrays else 0
        seqs += len(shard)
    for name in sorted(set(sums) - {p.name for p in shards}):
        problems.append(f"{name}: checksum without a shard")
    return {"ok": not problems, "shards": len(shards), "sequences": seqs, "rows": rows,
            "bytes": sum(p.stat().st_size for p in shards), "problems": problems}


# ---------------------------------------------------------------- writing

def _header(entries: list[tuple[str, str, tuple, int]], metadata: dict[str, str]) -> bytes:
    """The safetensors header for tensors laid out back to back in `entries` order, padded to 8 bytes."""
    head, offset = {}, 0
    for name, dtype, shape, nbytes in entries:
        head[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    head["__metadata__"] = metadata
    raw = json.dumps(head, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return len(raw).to_bytes(8, "little") + raw


class ShardWriter:
    """Accumulates captured sequences and writes them as ~`shard_bytes` shards; resumable (see the module doc).

    Windows whose context_rows reach `context_cap` and shorter windows fill separate shards, so a shard of full
    windows carries exactly the cap as its one `context_rows`."""

    def __init__(self, set_dir: str | Path, set_name: str | None = None, *, shard_bytes: int = 1 << 30,
                 hidden: int = HIDDEN, topk: int = TOPK, context_cap: int = 1024, log=print):
        self.dir = Path(set_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.set_name = set_name or self.dir.name
        self.shard_bytes, self.hidden, self.topk, self.log = shard_bytes, hidden, topk, log
        for stale in self.dir.glob("*.tmp"):  # a shard a dead run did not finish
            stale.unlink()
        self.finished: set[str] = set()
        shards = list_shards(self.dir)
        sums = read_sums(self.dir)
        for path in shards:
            shard = Shard(path)  # a renamed shard is complete (written and fsynced before the rename)
            problems = shard.check()
            if problems:
                raise ValueError(f"{path}: {problems}; move it away to recapture its sequences")
            self.finished |= set(shard.seq_ids)
            if path.name not in sums:  # the run died between the rename and the checksum line
                _append_sum(self.dir, path.name, sha256_file(path))
        self.next_index = (int(SHARD_RE.match(shards[-1].name).group(1)) + 1) if shards else 0
        self.disk_bytes = sum(p.stat().st_size for p in shards)
        self.context_cap = context_cap
        self._bufs: dict[bool, list[dict]] = {}  # full window (context_rows == cap) -> buffered sequences
        self._pending: dict[bool, int] = {}
        self.shards_written = 0

    @property
    def pending_bytes(self) -> int:
        return sum(self._pending.values())

    @property
    def total_bytes(self) -> int:
        """On disk plus buffered."""
        return self.disk_bytes + self.pending_bytes

    def add(self, seq_id: str, source: str, context_rows: int, tokens, positions, h, topk_ids, topk_logits, assist,
            extra: dict | None = None) -> None:
        """One sequence's rows (numpy; h as uint16 bf16 bits). Flushes a shard once the buffer reaches shard_bytes."""
        if seq_id in self.finished or any(b["id"] == seq_id for buf in self._bufs.values() for b in buf):
            raise ValueError(f"sequence {seq_id} is already captured")
        arrays = {"tokens": tokens, "positions": positions, "h": h, "topk_ids": topk_ids, "topk_logits": topk_logits,
                  "assist": assist}
        n = int(np.asarray(tokens).shape[0])
        if n < 1:
            raise ValueError(f"sequence {seq_id} has no rows")
        seq = {"id": str(seq_id), "source": str(source), "context_rows": int(context_rows), "extra": extra or {}}
        for name, (_, dt, trailing) in FIELDS.items():
            a = np.ascontiguousarray(arrays[name])
            want = (n,) + tuple(self.hidden if t == "hidden" else self.topk for t in trailing)
            if a.dtype != dt or a.shape != want:
                raise ValueError(f"{seq_id}: {name} is {a.dtype}{a.shape}, expected {np.dtype(dt)}{want}")
            seq[name] = a
        if not 0 <= seq["context_rows"] < n:
            raise ValueError(f"{seq_id}: context_rows {context_rows} not in [0, {n})")
        key = seq["context_rows"] >= self.context_cap
        self._bufs.setdefault(key, []).append(seq)
        self._pending[key] = self._pending.get(key, 0) + n * row_bytes(self.hidden, self.topk)
        if self._pending[key] >= self.shard_bytes:
            self._write(key)

    def flush(self) -> list[Path]:
        """Write every buffer as a shard (the full windows' first)."""
        return [p for key in sorted(self._bufs, reverse=True) if (p := self._write(key)) is not None]

    def _write(self, key: bool) -> Path | None:
        """Write one buffer as the next shard (atomically) and record its checksum."""
        buf = self._bufs.get(key)
        if not buf:
            return None
        n = sum(int(s["tokens"].shape[0]) for s in buf)
        starts = np.zeros(len(buf) + 1, dtype=np.int64)
        starts[1:] = np.cumsum([s["tokens"].shape[0] for s in buf])
        entries = []
        for name, (dtype, dt, trailing) in FIELDS.items():
            shape = (n,) + tuple(self.hidden if t == "hidden" else self.topk for t in trailing)
            entries.append((name, dtype, shape, int(np.prod(shape)) * np.dtype(dt).itemsize))
        entries.append(("seq_starts", "I64", starts.shape, starts.nbytes))
        meta = {"format": FORMAT, "set": self.set_name, "hidden": str(self.hidden), "topk": str(self.topk),
                "seq_ids": json.dumps([s["id"] for s in buf]), "sources": json.dumps([s["source"] for s in buf]),
                "context_rows": json.dumps(min(s["context_rows"] for s in buf)),
                "context_rows_per_seq": json.dumps([s["context_rows"] for s in buf]), "extra": json.dumps([s["extra"] for s in buf])}
        name = f"shard-{self.next_index:05d}.safetensors"
        final, tmp = self.dir / name, self.dir / (name + ".tmp")
        digest = hashlib.sha256()
        with open(tmp, "wb") as f:
            def put(data) -> None:
                mv = memoryview(data).cast("B")
                digest.update(mv)
                f.write(mv)
            put(_header(entries, meta))
            for field in FIELDS:  # each field's rows, sequence by sequence: no concatenated copy
                for s in buf:
                    put(s[field])
            put(starts)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        _fsync_dir(self.dir)
        _append_sum(self.dir, name, digest.hexdigest())
        size = final.stat().st_size
        self.disk_bytes += size
        self.finished |= {s["id"] for s in buf}
        self.next_index += 1
        self.shards_written += 1
        self._bufs[key], self._pending[key] = [], 0
        if self.log:
            self.log(f"wrote {name}: {len(buf)} sequences, {n} rows, {size / 2**30:.2f} GiB")
        return final

    def close(self) -> None:
        self.flush()


# ---------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["verify", "summary"])
    ap.add_argument("set_dir", type=Path)
    args = ap.parse_args(argv)
    if args.command == "verify":
        out = verify(args.set_dir)
        print(json.dumps(out, indent=1))
        return 0 if out["ok"] else 1
    shards = list_shards(args.set_dir)
    seqs = rows = assist = 0
    sources: dict[str, int] = {}
    for p in shards:
        s = Shard(p)
        seqs += len(s)
        rows += int(s.arrays["tokens"].shape[0])
        assist += int(s.arrays["assist"].sum())
        for src in s.meta["sources"]:
            sources[src] = sources.get(src, 0) + 1
    print(json.dumps({"shards": len(shards), "sequences": seqs, "rows": rows, "assist_rows": assist, "sources": sources,
                      "bytes": sum(p.stat().st_size for p in shards)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
