#!/usr/bin/env python3
"""HF-cache snapshot tools for the MTP draft-head experiment (stdlib only, no torch, no safetensors package).

Hibana serves whatever files sit in `<cache>/hub/models--<org>--<name>/snapshots/<revision>/`, resolved through
`model.safetensors.index.json` (`weight_map`: tensor name -> shard file). These tools build such cache roots out of hard
links so a second snapshot costs only the bytes that differ:

  stock       edp snapshot + stock originals of the files the abliteration changed -> a stock-weights cache root
              (the originals are verified against the HF LFS sha256 oids; the unchanged files against HF as well)
  extract-mtp the `mtp.*` tensors of a root, byte-exact, into one safetensors file
  writeback   a new root: hard links of a source root, a new `model-mtp-agent.safetensors` holding given `mtp.*`
              tensors, and a rewritten (real file) index pointing every `mtp.*` name at it
  compare-mtp the `mtp.*` tensors of two roots, read through their own indexes, byte-compared
  diff-roots  per file: shared inode / same size / different (cheap, no hashing)

Every tool only creates files under its --out-root; source snapshots are never written (inode, size and mtime of every
source file are re-checked after a writeback). Hard links need one filesystem: source and out-root must share it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MTP_PREFIX = "mtp."
MTP_COUNT = 31
MTP_FILE = "model-mtp-agent.safetensors"
INDEX = "model.safetensors.index.json"
# Repo docs the abliteration publisher wrote or edited; they never reach the engine, so a difference from HF is only reported.
DOC_FILES = {"README.md", "LICENSE", "PROVENANCE.json", ".gitattributes"}
MARKER = ".snapshot_tools_root"  # written into roots we create; --force only replaces roots that carry it
CHUNK = 8 << 20
MAX_HEADER = 256 << 20

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


class ToolError(Exception):
    """A refused or failed operation; the message says why."""


# ---------------------------------------------------------------- safetensors (header only parsing)

class Safetensors:
    """One safetensors file's validated header. `tensors`: name -> {dtype, shape, start, end} (absolute offsets)."""

    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ToolError(f"{self.path}: shorter than 8 bytes")
            (n,) = struct.unpack("<Q", raw)
            if n > MAX_HEADER or 8 + n > size:
                raise ToolError(f"{self.path}: implausible header length {n}")
            try:
                header = json.loads(f.read(n))
            except ValueError as e:
                raise ToolError(f"{self.path}: header is not JSON ({e})") from None
        if not isinstance(header, dict):
            raise ToolError(f"{self.path}: header is not an object")
        self.metadata = header.pop("__metadata__", None)
        self.data_start = 8 + n
        self.size = size
        self.tensors: dict[str, dict] = {}
        spans = []
        for name, t in header.items():
            try:
                lo, hi = t["data_offsets"]
                dtype, shape = t["dtype"], [int(x) for x in t["shape"]]
            except (KeyError, TypeError, ValueError):
                raise ToolError(f"{self.path}: malformed entry for {name}") from None
            if not 0 <= lo <= hi or self.data_start + hi > size:
                raise ToolError(f"{self.path}: {name} offsets {lo}:{hi} outside the file")
            if dtype in DTYPE_BYTES:
                numel = 1
                for d in shape:
                    numel *= d
                if numel * DTYPE_BYTES[dtype] != hi - lo:
                    raise ToolError(f"{self.path}: {name} {dtype}{shape} needs {numel * DTYPE_BYTES[dtype]} bytes, "
                                    f"offsets span {hi - lo}")
            self.tensors[name] = {"dtype": dtype, "shape": shape, "start": self.data_start + lo, "end": self.data_start + hi}
            spans.append((lo, hi, name))
        spans.sort()
        for (_, hi, a), (lo, _, b) in zip(spans, spans[1:]):
            if lo < hi:
                raise ToolError(f"{self.path}: tensors {a} and {b} overlap")

    def chunks(self, name, chunk=CHUNK):
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(t["start"])
            left = t["end"] - t["start"]
            while left:
                block = f.read(min(chunk, left))
                if not block:
                    raise ToolError(f"{self.path}: truncated while reading {name}")
                left -= len(block)
                yield block

    def digest(self, name):
        h = hashlib.sha256()
        for block in self.chunks(name):
            h.update(block)
        return h.hexdigest()


def write_safetensors(out_path, entries):
    """Write `entries` = [(name, dtype, shape, source Safetensors)] byte-exact from their sources, in the given order."""
    header, pos = {}, 0
    for name, dtype, shape, src in entries:
        n = src.tensors[name]["end"] - src.tensors[name]["start"]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + n]}
        pos += n
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)  # data starts 8-byte aligned
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, _, _, src in entries:
            for block in src.chunks(name):
                out.write(block)
        out.flush()
        os.fsync(out.fileno())


# ---------------------------------------------------------------- cache roots

def model_dir(repo):
    return "models--" + repo.replace("/", "--")


def find_snapshot(root, repo=REPO, revision=REVISION):
    """`<root>[/hub]/models--*/snapshots/<revision>` that holds an index; a root may also be the snapshot itself."""
    root = Path(root)
    if (root / INDEX).is_file():
        return root
    for base in (root, root / "hub"):
        snap = base / model_dir(repo) / "snapshots" / revision
        if (snap / INDEX).is_file():
            return snap
    raise ToolError(f"no snapshot {repo}@{revision[:7]} with {INDEX} under {root} (or {root}/hub)")


def read_index(snap):
    index = json.loads((Path(snap) / INDEX).read_text())
    if not isinstance(index.get("weight_map"), dict):
        raise ToolError(f"{snap / INDEX}: no weight_map")
    return index


def mtp_names(index, prefix=MTP_PREFIX):
    return sorted(n for n in index["weight_map"] if n.startswith(prefix))


def mtp_layout(snap, prefix=MTP_PREFIX):
    """name -> (Safetensors, dtype, shape), the `mtp.*` tensors read through the snapshot's own index."""
    index = read_index(snap)
    files: dict[str, Safetensors] = {}
    out = {}
    for name in mtp_names(index, prefix):
        shard = index["weight_map"][name]
        if shard not in files:
            files[shard] = Safetensors(Path(snap) / shard)
        t = files[shard].tensors.get(name)
        if t is None:
            raise ToolError(f"{snap}: index maps {name} to {shard}, which does not hold it")
        out[name] = (files[shard], t["dtype"], t["shape"])
    return out


def snapshot_files(snap):
    """All files under a snapshot as relative POSIX paths (links followed or not: only the names matter)."""
    snap = Path(snap)
    return sorted(p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file() or p.is_symlink())


def fingerprint(path):
    st = os.stat(path)
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def link(src, dst):
    """Hard link of the file `src` resolves to (an HF snapshot entry may be a symlink into blobs/)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(os.path.realpath(src), dst)
    except OSError as e:
        raise ToolError(f"hard link {src} -> {dst} failed ({e.strerror}); source and out-root must share a filesystem") from None


def prepare_out_root(out_root, force):
    """A fresh out-root with its marker; an existing one is refused unless --force and made by these tools."""
    out_root = Path(out_root)
    if out_root.exists() or out_root.is_symlink():
        if not force:
            raise ToolError(f"{out_root} exists; refusing to overwrite (use --force to replace a root made by these tools)")
        if not (out_root / MARKER).is_file():
            raise ToolError(f"{out_root} exists and was not made by snapshot_tools (no {MARKER}); not replacing it")
        shutil.rmtree(out_root)  # only unlinks: the data stays with the other hard links
    out_root.mkdir(parents=True)
    (out_root / MARKER).write_text("created by snapshot_tools.py\n")
    return out_root


def new_snapshot_dir(out_root, repo, revision):
    base = Path(out_root) / "hub" / model_dir(repo)
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text(revision)
    snap = base / "snapshots" / revision
    snap.mkdir(parents=True)
    return snap


# ---------------------------------------------------------------- hashing against Hugging Face

def hf_tree(repo, revision):
    """{path: entry} of the repo tree at `revision` (entries carry lfs.oid = the file's sha256 for LFS files)."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
    entries = {}
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "snapshot-tools"})
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
        with urllib.request.urlopen(req, timeout=60) as r:
            for e in json.load(r):
                if e.get("type") == "file":
                    entries[e["path"]] = e
            nxt = [p.split(";")[0].strip(" <>") for p in (r.headers.get("Link") or "").split(",") if 'rel="next"' in p]
        url = nxt[0] if nxt else None
    return entries


def hash_file(path, algo):
    """sha256 of the file, or the git blob sha1 (`blob <size>\\0` + bytes) for files HF stores outside LFS."""
    size = os.path.getsize(path)
    h = hashlib.new(algo)
    if algo == "sha1":
        h.update(f"blob {size}\0".encode())
    with open(path, "rb", buffering=0) as f:
        while block := f.read(CHUNK):
            h.update(block)
        if hasattr(os, "posix_fadvise"):  # ~90 GB of hashing must not push anything else out of the page cache
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def verify_against_hf(files, tree, jobs, log):
    """`files`: {relative path: absolute path}. Returns (checked, problems); LFS files by sha256, others by git sha1."""
    problems, todo = [], []
    for rel, path in sorted(files.items()):
        e = tree.get(rel)
        if e is None:
            problems.append(f"{rel}: not on HF at this revision")
        elif os.path.getsize(path) != e["size"]:
            problems.append(f"{rel}: size {os.path.getsize(path)} != HF {e['size']}")
        else:
            lfs = e.get("lfs")
            todo.append((rel, path, "sha256" if lfs else "sha1", lfs["oid"] if lfs else e["oid"]))
    done = 0
    with ThreadPoolExecutor(jobs) as pool:
        futures = {pool.submit(hash_file, p, a): (rel, a, oid) for rel, p, a, oid in todo}
        for fut in as_completed(futures):
            rel, algo, oid = futures[fut]
            got = fut.result()
            done += 1
            if got != oid:
                problems.append(f"{rel}: {algo} {got} != HF {oid}")
            if done % 25 == 0:
                log(f"  hashed {done}/{len(todo)}")
    return len(todo), problems


# ---------------------------------------------------------------- commands

def cmd_stock(a):
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    edp = Path(a.edp)
    originals = Path(a.originals)
    revision = a.revision or edp.name
    snap_files = {p: edp / p for p in snapshot_files(edp)}
    changed = sorted(p.name for p in originals.glob("*.safetensors"))
    if not changed:
        raise ToolError(f"no *.safetensors originals in {originals}")
    foreign = [n for n in changed if n not in snap_files]
    if foreign:
        raise ToolError(f"originals not in the edp snapshot: {foreign}")
    if a.expect_changed and len(changed) != a.expect_changed:
        raise ToolError(f"expected {a.expect_changed} originals, found {len(changed)}")
    out_root = Path(a.out_root)
    if (out_root.exists() or out_root.is_symlink()) and not a.force:
        raise ToolError(f"{out_root} exists; refusing to overwrite (use --force)")  # fail before hashing 90 GB
    t0 = time.time()
    if a.verify != "none":
        tree = hf_tree(a.repo, revision) if not a.oids_json else {e["path"]: e for e in json.load(open(a.oids_json)) if e.get("type") == "file"}
        log(f"HF tree: {len(tree)} files")
        if a.verify == "full":
            checked, problems = verify_against_hf({n: originals / n for n in changed}, tree, a.jobs, log)
            report = f"originals: {checked} files verified against HF sha256 oids, {len(problems)} problems"
            log(report)
            rest = {p: v for p, v in snap_files.items() if p not in set(changed)}
            c2, p2 = verify_against_hf(rest, tree, a.jobs, log)
            log(f"edp unchanged files: {c2} verified (sha256 for LFS, git sha1 otherwise), {len(p2)} problems")
            problems += p2
            docs = [p for p in problems if p.split(":")[0] in DOC_FILES]
            problems = [p for p in problems if p not in docs]
            for p in docs:
                log(f"  note (docs file, not served): {p}")
        else:  # sizes: cheap check, names + sizes of every file against HF
            problems = []
            for n in changed:
                e = tree.get(n)
                if e is None or e["size"] != os.path.getsize(originals / n):
                    problems.append(f"{n}: size mismatch with HF")
            for rel, path in snap_files.items():
                if rel not in changed and (rel not in tree or tree[rel]["size"] != os.path.getsize(path)):
                    problems.append(f"{rel}: missing on HF or size mismatch")
        log(f"verification wall time {time.time() - t0:.1f} s")
        if problems:
            raise ToolError("verification failed:\n  " + "\n  ".join(problems))
    prepare_out_root(out_root, a.force)
    snap = new_snapshot_dir(out_root, a.repo, revision)
    for rel in snap_files:
        link(originals / rel if rel in changed else snap_files[rel], snap / rel)
    print(f"stock root: {out_root}  ({len(snap_files)} files, {len(changed)} from originals)")
    print(f"snapshot: {snap}")


def cmd_extract(a):
    snap = find_snapshot(a.src_root, a.repo, a.revision)
    layout = mtp_layout(snap, a.prefix)
    if a.expect_count and len(layout) != a.expect_count:
        raise ToolError(f"{snap}: {len(layout)} tensors start with {a.prefix!r}, expected {a.expect_count}")
    out = Path(a.out)
    if out.exists() and not a.force:
        raise ToolError(f"{out} exists; refusing to overwrite (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    write_safetensors(tmp, [(n, dt, sh, f) for n, (f, dt, sh) in sorted(layout.items())])
    tmp.replace(out)
    print(f"extracted {len(layout)} tensors, {out.stat().st_size} bytes -> {out}")


def validate_mtp_file(path, layout, expect_count):
    """Exactly the source's mtp.* names, with the source's dtypes and shapes."""
    f = Safetensors(path)
    want, got = set(layout), set(f.tensors)
    if expect_count and len(want) != expect_count:
        raise ToolError(f"source has {len(want)} mtp tensors, expected {expect_count}")
    if want != got:
        raise ToolError(f"{path}: tensor names differ from the source: missing {sorted(want - got)}, extra {sorted(got - want)}")
    bad = [f"{n}: {f.tensors[n]['dtype']}{f.tensors[n]['shape']} != source {dt}{sh}"
           for n, (_, dt, sh) in sorted(layout.items()) if (f.tensors[n]["dtype"], f.tensors[n]["shape"]) != (dt, sh)]
    if bad:
        raise ToolError(f"{path}: dtype/shape mismatch:\n  " + "\n  ".join(bad))
    return f


def cmd_writeback(a):
    src = find_snapshot(a.src_root, a.repo, a.revision)
    revision = src.name
    index = read_index(src)
    layout = mtp_layout(src, a.prefix)
    new = validate_mtp_file(a.mtp_tensors, layout, a.expect_count)
    rels = snapshot_files(src)
    before = {r: fingerprint(src / r) for r in rels}
    out_root = prepare_out_root(a.out_root, a.force)
    out_snap = new_snapshot_dir(out_root, a.repo, revision)
    for rel in rels:
        if rel not in (INDEX, MTP_FILE):  # the index is rewritten, never linked; the mtp file is replaced
            link(src / rel, out_snap / rel)
    shutil.copyfile(a.mtp_tensors, out_snap / MTP_FILE)
    with open(out_snap / MTP_FILE, "rb") as fh:
        os.fsync(fh.fileno())
    if (out_snap / MTP_FILE).stat().st_size != new.size:
        raise ToolError("copy of the mtp tensors has the wrong size")
    wm = dict(index["weight_map"])
    for name in layout:
        wm[name] = MTP_FILE
    index = {**index, "weight_map": wm}
    with open(out_snap / INDEX, "w") as fh:  # a fresh inode: the source index shares its inode with other roots
        json.dump(index, fh)
        fh.flush()
        os.fsync(fh.fileno())
    # the source must be untouched, the new root must read back byte-equal through its own index
    after = {r: fingerprint(src / r) for r in rels}
    if before != after:
        raise ToolError(f"source snapshot changed: {[r for r in rels if before[r] != after[r]]}")
    back = mtp_layout(out_snap, a.prefix)
    wrong = sorted(n for n, (f, dt, sh) in back.items() if f.digest(n) != new.digest(n))
    if set(back) != set(layout) or wrong:
        raise ToolError(f"read-back through the new index differs from {a.mtp_tensors}: {wrong or 'names'}")
    shared = sum(1 for r in rels if r not in (INDEX, MTP_FILE) and os.path.samefile(src / r, out_snap / r))
    print(f"writeback root: {out_root}\nsnapshot: {out_snap}\n{len(layout)} mtp tensors -> {MTP_FILE} "
          f"({new.size} bytes); {shared} files share inodes with the source; source untouched ({len(rels)} files checked)")


def compare_roots(a_snap, b_snap, prefix, expect=0):
    """Problems between two snapshots' mtp.* tensors (each read through its own index); empty when byte-equal."""
    la, lb = mtp_layout(a_snap, prefix), mtp_layout(b_snap, prefix)
    bad = []
    if set(la) != set(lb):
        bad.append(f"names differ: only in a {sorted(set(la) - set(lb))}, only in b {sorted(set(lb) - set(la))}")
    if expect and len(la) != expect:
        bad.append(f"{len(la)} tensors, expected {expect}")
    for n in sorted(set(la) & set(lb)):
        (fa, da, sa), (fb, db, sb) = la[n], lb[n]
        if (da, sa) != (db, sb) or fa.digest(n) != fb.digest(n):
            bad.append(n)
    return bad


def cmd_compare(a):
    sa, sb = find_snapshot(a.a_root, a.repo, a.revision), find_snapshot(a.b_root, a.repo, a.revision)
    bad = compare_roots(sa, sb, a.prefix, expect=a.expect_count)
    n = len(mtp_layout(sa, a.prefix))
    if bad:
        raise ToolError(f"{len(bad)} of {n} mtp tensors differ: {bad}")
    print(f"{n} mtp tensors byte-equal between {sa} and {sb}")


def cmd_diff(a):
    sa, sb = find_snapshot(a.a_root, a.repo, a.revision), find_snapshot(a.b_root, a.repo, a.revision)
    fa, fb = set(snapshot_files(sa)), set(snapshot_files(sb))
    kinds = {"same inode": [], "same size, other inode": [], "different size": [], "only in a": sorted(fa - fb), "only in b": sorted(fb - fa)}
    for r in sorted(fa & fb):
        x, y = os.stat(sa / r), os.stat(sb / r)
        kinds["same inode" if (x.st_dev, x.st_ino) == (y.st_dev, y.st_ino) else
              "same size, other inode" if x.st_size == y.st_size else "different size"].append(r)
    for k, v in kinds.items():
        print(f"{k}: {len(v)}" + ("  " + ", ".join(v) if v and k != "same inode" else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, roots=("src-root",)):
        s.add_argument("--repo", default=REPO)
        s.add_argument("--revision", default=REVISION)
        s.add_argument("--prefix", default=MTP_PREFIX)
        s.add_argument("--expect-count", type=int, default=MTP_COUNT, help="number of mtp tensors (0: do not check)")
        for r in roots:
            s.add_argument("--" + r, required=True, help="HF cache root (<root>/hub/models--*/snapshots/<rev>) or a snapshot dir")

    s = sub.add_parser("stock", help="stock-weights cache root from the edp snapshot + the stock originals")
    s.add_argument("--edp", required=True, help="the edp snapshot directory")
    s.add_argument("--originals", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--repo", default=REPO)
    s.add_argument("--revision", default=None, help="default: the edp directory's name")
    s.add_argument("--verify", choices=("full", "sizes", "none"), default="full",
                   help="full: sha256 (LFS) / git sha1 of every file against HF; sizes: names and sizes only")
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--oids-json", help="a saved HF tree listing instead of the API")
    s.add_argument("--expect-changed", type=int, default=24)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_stock)

    s = sub.add_parser("extract-mtp", help="the mtp.* tensors of a root into one safetensors file")
    common(s)
    s.add_argument("--out", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("writeback", help="new root with the given mtp.* tensors in model-mtp-agent.safetensors")
    common(s)
    s.add_argument("--mtp-tensors", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_writeback)

    s = sub.add_parser("compare-mtp", help="byte-compare the mtp.* tensors of two roots")
    common(s, roots=("a-root", "b-root"))
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("diff-roots", help="per file: shared inode / same size / different")
    common(s, roots=("a-root", "b-root"))
    s.set_defaults(fn=cmd_diff)

    a = p.parse_args(argv)
    try:
        a.fn(a)
    except ToolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
