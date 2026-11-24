"""Tests for snapshot_tools.py on tiny synthetic safetensors snapshots (no torch, no network)."""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snapshot_tools as st  # noqa: E402

REV = "a" * 40
REPO = "Org/Model"
DTYPES = {"BF16": 2, "F32": 4, "U8": 1}


def write_st(path, tensors):
    """tensors: {name: (dtype, shape, bytes)}; a valid safetensors file."""
    header, pos, body = {}, 0, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + len(data)]}
        pos += len(data)
        body += data
    raw = json.dumps(header).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + body)


def tensor(dtype, shape, seed):
    n = DTYPES[dtype]
    for d in shape:
        n *= d
    return dtype, shape, bytes((seed * 31 + i) % 251 for i in range(n))


MTP = {f"mtp.t{i}": tensor("BF16" if i % 2 else "F32", [2, i + 1], 10 + i) for i in range(4)}
MTP_N = len(MTP)


def make_snapshot(root, variant=0):
    """<root>/hub/models--Org--Model/snapshots/<REV>: shard-a (trunk), shard-b (trunk + mtp), config, index."""
    snap = Path(root) / "hub" / st.model_dir(REPO) / "snapshots" / REV
    snap.mkdir(parents=True)
    write_st(snap / "shard-a.safetensors", {"w.a": tensor("BF16", [3, 3], 1 + variant), "w.b": tensor("U8", [5], 2)})
    write_st(snap / "shard-b.safetensors", {"w.c": tensor("F32", [2, 2], 3 + variant), **MTP})
    write_st(snap / "shard-c.safetensors", {"w.d": tensor("U8", [7], 4)})
    (snap / "config.json").write_text('{"a": 1}')
    wm = {"w.a": "shard-a.safetensors", "w.b": "shard-a.safetensors", "w.c": "shard-b.safetensors",
          "w.d": "shard-c.safetensors", **{n: "shard-b.safetensors" for n in MTP}}
    (snap / st.INDEX).write_text(json.dumps({"metadata": {"total_size": 123}, "weight_map": wm}))
    return snap


def run(*argv):
    return st.main([*map(str, argv)])


def same_inode(a, b):
    return os.path.samefile(a, b)


def snap_of(root):
    return Path(root) / "hub" / st.model_dir(REPO) / "snapshots" / REV


def tree_json(path, snap_hf):
    """What the HF tree API would return for the 'HF' snapshot `snap_hf` (LFS safetensors, plain config/index)."""
    entries = []
    for p in sorted(snap_hf.iterdir()):
        data = p.read_bytes()
        e = {"type": "file", "path": p.name, "size": len(data)}
        if p.suffix == ".safetensors":
            e["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
            e["oid"] = "0" * 40
        else:
            e["oid"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        entries.append(e)
    Path(path).write_text(json.dumps(entries))


@pytest.fixture
def edp_world(tmp_path):
    """An 'HF' snapshot (the stock originals), an edp snapshot that differs in shard-a, and an originals dir."""
    hf = make_snapshot(tmp_path / "hf")
    edp_root = tmp_path / "edp"
    edp = make_snapshot(edp_root, variant=1)  # differs in shard-a and shard-b (w.a / w.c); mtp bytes equal
    # edp differs from HF only in shard-a for this test: restore shard-b from HF content
    (edp / "shard-b.safetensors").unlink()
    os.link(hf / "shard-b.safetensors", edp / "shard-b.safetensors")
    originals = tmp_path / "originals"
    originals.mkdir()
    os.link(hf / "shard-a.safetensors", originals / "shard-a.safetensors")
    tree = tmp_path / "tree.json"
    tree_json(tree, hf)
    return {"tmp": tmp_path, "hf": hf, "edp": edp, "edp_root": edp_root, "originals": originals, "tree": tree}


def stock(w, out, *extra):
    return run("stock", "--edp", w["edp"], "--originals", w["originals"], "--out-root", out, "--repo", REPO,
               "--oids-json", w["tree"], "--expect-changed", 1, "--jobs", 2, *extra)


# ---------------------------------------------------------------- stock

def test_stock_links_changed_from_originals_and_rest_from_edp(edp_world):
    w, out = edp_world, edp_world["tmp"] / "stock"
    assert stock(w, out) == 0
    snap = snap_of(out)
    assert (out / "hub" / st.model_dir(REPO) / "refs" / "main").read_text() == REV
    assert same_inode(snap / "shard-a.safetensors", w["originals"] / "shard-a.safetensors")
    assert not same_inode(snap / "shard-a.safetensors", w["edp"] / "shard-a.safetensors")
    for name in ("shard-b.safetensors", "shard-c.safetensors", "config.json", st.INDEX):
        assert same_inode(snap / name, w["edp"] / name), name
    assert sorted(p.name for p in snap.iterdir()) == sorted(p.name for p in w["edp"].iterdir())


def test_stock_rejects_corrupt_original_before_building(edp_world, capsys):
    w, out = edp_world, edp_world["tmp"] / "stock"
    bad = w["tmp"] / "bad"
    bad.mkdir()
    data = bytearray((w["originals"] / "shard-a.safetensors").read_bytes())
    data[-1] ^= 1
    (bad / "shard-a.safetensors").write_bytes(bytes(data))
    w = {**w, "originals": bad}
    assert stock(w, out) == 1
    assert "sha256" in capsys.readouterr().err
    assert not out.exists()


def test_stock_rejects_changed_unchanged_file(edp_world, capsys):
    w, out = edp_world, edp_world["tmp"] / "stock"
    p = w["edp"] / "shard-c.safetensors"
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.unlink()  # the file is hard-linked with the 'HF' copy: never write through it
    p.write_bytes(bytes(data))
    assert stock(w, out) == 1
    assert "shard-c.safetensors" in capsys.readouterr().err and not out.exists()


def test_stock_docs_files_may_differ_from_hf(edp_world, capsys):
    w, out = edp_world, edp_world["tmp"] / "stock"
    (w["edp"] / "README.md").write_text("edp only")
    assert stock(w, out) == 0
    assert "README.md" in capsys.readouterr().err and (snap_of(out) / "README.md").exists()


def test_stock_refuses_existing_out_root(edp_world):
    w, out = edp_world, edp_world["tmp"] / "stock"
    out.mkdir()
    assert stock(w, out) == 1
    assert not (out / "hub").exists()


def test_stock_force_only_replaces_our_roots(edp_world):
    w, out = edp_world, edp_world["tmp"] / "stock"
    assert stock(w, out) == 0
    assert stock(w, out) == 1  # exists
    assert stock(w, out, "--force") == 0
    foreign = w["tmp"] / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x")
    assert stock(w, foreign, "--force") == 1
    assert (foreign / "keep.txt").exists()


def test_stock_originals_must_belong_to_edp(edp_world):
    w = edp_world
    (w["originals"] / "extra.safetensors").write_bytes(b"x")
    assert stock(w, w["tmp"] / "stock") == 1


# ---------------------------------------------------------------- extract / writeback

def extract(root, out, *extra):
    return run("extract-mtp", "--src-root", root, "--repo", REPO, "--revision", REV, "--expect-count", MTP_N,
               "--out", out, *extra)


def writeback(root, tensors, out, *extra):
    return run("writeback", "--src-root", root, "--repo", REPO, "--revision", REV, "--expect-count", MTP_N,
               "--mtp-tensors", tensors, "--out-root", out, *extra)


def test_extract_is_byte_exact(tmp_path):
    root = tmp_path / "r"
    make_snapshot(root)
    out = tmp_path / "mtp.safetensors"
    assert extract(root, out) == 0
    f = st.Safetensors(out)
    assert sorted(f.tensors) == sorted(MTP)
    for name, (dtype, shape, data) in MTP.items():
        assert (f.tensors[name]["dtype"], f.tensors[name]["shape"]) == (dtype, shape)
        assert b"".join(f.chunks(name)) == data
    assert extract(root, out) == 1  # exists
    assert extract(root, out, "--force") == 0


def test_extract_wrong_count(tmp_path):
    make_snapshot(tmp_path / "r")
    assert run("extract-mtp", "--src-root", tmp_path / "r", "--repo", REPO, "--revision", REV, "--expect-count", 31,
               "--out", tmp_path / "o.safetensors") == 1


def fingerprints(snap):
    return {p.name: (os.stat(p).st_ino, os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in snap.iterdir()}


def test_identity_roundtrip(tmp_path):
    root = tmp_path / "r"
    src = make_snapshot(root)
    mtp = tmp_path / "mtp.safetensors"
    assert extract(root, mtp) == 0
    before = fingerprints(src)
    index_before = (src / st.INDEX).read_bytes()
    out = tmp_path / "out"
    assert writeback(root, mtp, out) == 0
    new = snap_of(out)
    # source untouched
    assert fingerprints(src) == before and (src / st.INDEX).read_bytes() == index_before
    # non-mtp files are links, the index is a new real file, the mtp file is new
    for name in ("shard-a.safetensors", "shard-b.safetensors", "shard-c.safetensors", "config.json"):
        assert same_inode(new / name, src / name)
    assert not same_inode(new / st.INDEX, src / st.INDEX)
    assert not (new / st.INDEX).is_symlink() and (new / st.INDEX).stat().st_nlink == 1
    assert (new / st.MTP_FILE).stat().st_nlink == 1
    # index: mtp names -> new file, the rest unchanged, metadata kept
    old, idx = json.loads(index_before), json.loads((new / st.INDEX).read_text())
    assert idx["metadata"] == old["metadata"]
    for name, shard in idx["weight_map"].items():
        assert shard == (st.MTP_FILE if name in MTP else old["weight_map"][name])
    # all tensors read back through the new index are byte-equal to the source's
    layout = st.mtp_layout(new)
    for name, (dtype, shape, data) in MTP.items():
        f, dt, sh = layout[name]
        assert f.path.name == st.MTP_FILE and (dt, sh) == (dtype, shape) and b"".join(f.chunks(name)) == data
    assert run("compare-mtp", "--a-root", root, "--b-root", out, "--repo", REPO, "--revision", REV,
               "--expect-count", MTP_N) == 0


def test_writeback_refuses_existing_out_root_and_force_replaces(tmp_path):
    root = tmp_path / "r"
    make_snapshot(root)
    mtp = tmp_path / "mtp.safetensors"
    extract(root, mtp)
    out = tmp_path / "out"
    assert writeback(root, mtp, out) == 0
    assert writeback(root, mtp, out) == 1
    assert writeback(root, mtp, out, "--force") == 0
    # the source index is still intact after the force replacement
    assert json.loads((snap_of(root) / st.INDEX).read_text())["weight_map"]["mtp.t0"] == "shard-b.safetensors"


def modified_mtp(tmp_path, root, edit):
    """An mtp file derived from the source's, with `edit(tensors)` applied to a {name: (dtype, shape, bytes)} dict."""
    layout = st.mtp_layout(snap_of(root))
    tensors = {n: (dt, sh, b"".join(f.chunks(n))) for n, (f, dt, sh) in layout.items()}
    edit(tensors)
    path = tmp_path / "mod.safetensors"
    write_st(path, tensors)
    return path


def test_writeback_accepts_changed_values(tmp_path):
    root = tmp_path / "r"
    make_snapshot(root)
    mtp = modified_mtp(tmp_path, root, lambda t: t.__setitem__("mtp.t0", ("F32", [2, 1], b"\x01" * 8)))
    out = tmp_path / "out"
    assert writeback(root, mtp, out) == 0
    f, _, _ = st.mtp_layout(snap_of(out))["mtp.t0"]
    assert b"".join(f.chunks("mtp.t0")) == b"\x01" * 8
    assert run("compare-mtp", "--a-root", root, "--b-root", out, "--repo", REPO, "--revision", REV,
               "--expect-count", MTP_N) == 1  # differs, as it should


@pytest.mark.parametrize("edit", [
    lambda t: t.pop("mtp.t3"),                                                   # missing name
    lambda t: t.__setitem__("mtp.extra", ("U8", [1], b"x")),                     # extra name
    lambda t: t.__setitem__("mtp.t0", ("F32", [1, 2], b"\0" * 8)),               # wrong shape
    lambda t: t.__setitem__("mtp.t0", ("U8", [2, 1], b"\0" * 2)),                # wrong dtype
    lambda t: (t.pop("mtp.t3"), t.__setitem__("w.other", ("U8", [1], b"x"))),    # renamed
])
def test_writeback_refuses_bad_tensor_sets(tmp_path, edit):
    root = tmp_path / "r"
    src = make_snapshot(root)
    before = fingerprints(src)
    mtp = modified_mtp(tmp_path, root, edit)
    out = tmp_path / "out"
    assert writeback(root, mtp, out) == 1
    assert not out.exists() and fingerprints(src) == before


def test_writeback_refuses_corrupt_file(tmp_path):
    root = tmp_path / "r"
    make_snapshot(root)
    mtp = tmp_path / "mtp.safetensors"
    extract(root, mtp)
    mtp.write_bytes(mtp.read_bytes()[:-3])  # truncated: offsets run past the end
    assert writeback(root, mtp, tmp_path / "out") == 1


def test_writeback_over_a_writeback_replaces_the_agent_file(tmp_path):
    root = tmp_path / "r"
    make_snapshot(root)
    mtp = tmp_path / "mtp.safetensors"
    extract(root, mtp)
    first = tmp_path / "first"
    assert writeback(root, mtp, first) == 0
    mtp2 = modified_mtp(tmp_path, first, lambda t: t.__setitem__("mtp.t0", ("F32", [2, 1], b"\x02" * 8)))
    second = tmp_path / "second"
    assert writeback(first, mtp2, second) == 0
    assert not same_inode(snap_of(second) / st.MTP_FILE, snap_of(first) / st.MTP_FILE)
    f, _, _ = st.mtp_layout(snap_of(second))["mtp.t0"]
    assert b"".join(f.chunks("mtp.t0")) == b"\x02" * 8
    f1, _, _ = st.mtp_layout(snap_of(first))["mtp.t0"]
    assert b"".join(f1.chunks("mtp.t0")) == MTP["mtp.t0"][2]  # the first root is unchanged


def test_find_snapshot_accepts_snapshot_dir_and_hub_less_root(tmp_path):
    root = tmp_path / "r"
    snap = make_snapshot(root)
    assert st.find_snapshot(snap) == snap
    flat = tmp_path / "flat" / st.model_dir(REPO) / "snapshots"
    flat.mkdir(parents=True)
    os.symlink(snap, flat / REV)
    assert st.find_snapshot(tmp_path / "flat", REPO, REV) == flat / REV
    with pytest.raises(st.ToolError):
        st.find_snapshot(tmp_path / "nothing", REPO, REV)


def test_diff_roots_reports_shared_inodes(tmp_path, capsys):
    root = tmp_path / "r"
    make_snapshot(root)
    mtp = tmp_path / "mtp.safetensors"
    extract(root, mtp)
    out = tmp_path / "out"
    writeback(root, mtp, out)
    capsys.readouterr()
    assert run("diff-roots", "--a-root", root, "--b-root", out, "--repo", REPO, "--revision", REV) == 0
    text = capsys.readouterr().out
    assert "same inode: 4" in text and "only in b: 1  " + st.MTP_FILE in text
