# SPDX-License-Identifier: Apache-2.0
"""CPU judges for mtp_train.py on synthetic capture shards and the real checkpoint (experts and vocabulary sliced):
one batch overfits, frozen tensors stay bit-identical (in the module and in the exported file), a resumed run
reproduces the uninterrupted run's losses. Runs under pytest or as a script."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402

import mtp_torch as mt  # noqa: E402
import mtp_train  # noqa: E402

SNAP = os.environ.get("MTP_SNAPSHOT", "/hf")
ER = os.environ.get("MTP_EXPERTS", "0:16")
VOCAB = int(os.environ.get("MTP_VOCAB", "4096"))
THREADS = os.environ.get("THREADS", "16")
RESULTS: dict = {}


def record(name, **kv):
    RESULTS[name] = kv
    print(f"[{name}] " + json.dumps(kv, default=str), flush=True)


def have_snapshot():
    try:
        mt.resolve_snapshot(SNAP)
        return True
    except FileNotFoundError:
        return False


def skip_without_snapshot():
    if not have_snapshot():
        import pytest
        pytest.skip("no snapshot")


def capture_root(tmp: Path, rows=48) -> Path:
    root = tmp / "capture"
    for i, (name, wins) in enumerate({
        "train": [{"rows": rows, "source": "zeroclaw", "seq_id": "t0"}, {"rows": rows, "source": "hermes", "seq_id": "t1"},
                  {"rows": rows - 8, "source": "codex", "seq_id": "t2", "start": 300}],
        "general": [{"rows": rows, "source": "general", "seq_id": "g0", "start": 50}],
        "heldout": [{"rows": rows, "source": "zeroclaw", "seq_id": "h0"}, {"rows": rows, "source": "hermes", "seq_id": "h1"}],
    }.items()):
        (root / name).mkdir(parents=True)
        mt.write_synthetic_shard(root / name / "shard-00000.safetensors", wins, context_rows=4, seed=10 + i, vocab=VOCAB,
                                 meta_format="blob" if name == "general" else "keys")
    return root


def base_args(root, out, *extra):
    return ["--capture-root", str(root), "--snapshot", SNAP, "--out-dir", str(out), "--device", "cpu", "--compute-dtype", "fp32",
            "--expert-range", ER, "--vocab-rows", str(VOCAB), "--window-rows", "64", "--threads", THREADS, "--log-every", "1",
            "--eval-windows", "2", "--reserve-minutes", "0", *extra]


def history(out):
    return [json.loads(x) for x in (Path(out) / "history.jsonl").read_text().splitlines()]


def sha_region(path, start, n):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(start)
        left = n
        while left:
            b = fh.read(min(left, 64 << 20))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def test_overfit_and_export():
    skip_without_snapshot()
    tmp = Path(tempfile.mkdtemp())
    try:
        root = capture_root(tmp)
        out = tmp / "run"
        rc = mtp_train.main(base_args(root, out, "--overfit-one-batch", "--tokens-per-step", "1", "--max-steps", "30", "--lr", "3e-4",
                                      "--warmup", "1", "--min-lr-frac", "1.0", "--eval-every", "0", "--eval-windows", "0"))
        assert rc == 0
        hs = history(out)
        first = hs[0]["loss"]
        rep = json.loads((out / "report.json").read_text())
        # the exported file: the 31 names, frozen tensors byte-identical to the snapshot, trainable ones = the best state
        snap = mt.Snapshot(SNAP)
        header, base = mt.read_header(out / "mtp_trained.safetensors")
        names = sorted(k for k in header if k != "__metadata__")
        frozen_identical = {}
        for n in mt.FROZEN:
            a, b = header[n]["data_offsets"]
            src, s0, nb, _, _ = mt.raw_entry(snap, n)
            frozen_identical[n] = (b - a == nb) and sha_region(out / "mtp_trained.safetensors", base + a, b - a) == sha_region(src, s0, nb)
        from safetensors import safe_open
        last = torch.load(out / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)["params"]  # no held-out: the last state
        with safe_open(str(out / "mtp_trained.safetensors"), "pt") as fh:
            trained_match = all(torch.equal(fh.get_tensor(n), last[n].to(torch.bfloat16)) for n in mt.TRAINABLE)
            changed = sum(not torch.equal(fh.get_tensor(n), snap.get(n)) for n in mt.TRAINABLE)
        record("overfit_and_export", steps=len(hs), loss_first=first, loss_last=hs[-1]["loss"], drop=round(1 - hs[-1]["loss"] / first, 4),
               exported_tensors=len(names), frozen_byte_identical=frozen_identical, trainable_equal_last_state=trained_match,
               trainable_changed=changed, losses=[h["loss"] for h in hs])
        assert hs[-1]["loss"] <= 0.5 * first, (first, hs[-1]["loss"])
        assert names == sorted(mt.EXPECTED) and all(frozen_identical.values()) and trained_match and changed == len(mt.TRAINABLE)
    finally:
        shutil.rmtree(tmp)


def test_frozen_unchanged_in_module():
    skip_without_snapshot()
    from mtp_train import window_loss
    er = tuple(int(x) for x in ER.split(":"))
    h = mt.MtpHeadTorch.from_snapshot(SNAP, mode="float", compute_dtype=torch.float32, expert_range=er, vocab_rows=VOCAB, grad_ckpt=True)
    before = {b: getattr(h, b).clone() for b in ("router", "experts_gu", "experts_down", "embed", "lm_head")}
    before.update({n: getattr(h, mt.key(n)).clone() for n in mt.INDEXER})
    snap = mt.Snapshot(SNAP)
    tmp = Path(tempfile.mkdtemp())
    try:
        mt.write_synthetic_shard(tmp / "s.safetensors", [{"rows": 32, "source": "x"}], vocab=VOCAB)
        win = mt.CaptureShard(tmp / "s.safetensors").window(0)
    finally:
        shutil.rmtree(tmp)
    opt = torch.optim.AdamW(h.trainable_parameters(), lr=1e-3)
    v = win.targets(4)[2]
    denom = [int(v[:, j].sum()) for j in range(4)]
    p0 = h.param("mtp.fc_hidden.weight").detach().clone()
    for _ in range(3):
        opt.zero_grad()
        loss, _ = window_loss(h, win, 4, [1, .8, .6, .5], 0.3, denom, torch.device("cpu"))
        loss.backward()
        opt.step()
    same = {k: bool(torch.equal(t, getattr(h, k) if hasattr(h, k) else getattr(h, mt.key(k)))) for k, t in before.items()}
    vs_snapshot = {
        "router": torch.equal(h.router, snap.get(mt.ROUTER, er)),
        "experts_gu": torch.equal(h.experts_gu, snap.get(mt.EXPERTS_GU, er)),
        "experts_down": torch.equal(h.experts_down, snap.get(mt.EXPERTS_DOWN, er)),
        "lm_head": torch.equal(h.lm_head, snap.get(mt.LM_HEAD_NAME, (0, VOCAB))),
        "embed": torch.equal(h.embed, snap.get(mt.EMBED_NAME, (0, VOCAB))),
    }
    moved = float((h.param("mtp.fc_hidden.weight") - p0).abs().max())
    record("frozen_in_module", unchanged_after_steps=same, equal_to_snapshot=vs_snapshot, trainable_moved=moved)
    assert all(same.values()) and all(vs_snapshot.values()) and moved > 0


def test_resume_reproduces_loss():
    skip_without_snapshot()
    tmp = Path(tempfile.mkdtemp())
    try:
        root = capture_root(tmp)
        common = ["--tokens-per-step", "60", "--lr", "1e-4", "--warmup", "100", "--eval-every", "0", "--eval-mode", "float",
                  "--ckpt-every", "5", "--seed", "3"]
        # one thread: multithreaded CPU reductions differ run to run by ~1e-5 on these losses, resumed or not
        common += ["--threads", os.environ.get("RESUME_THREADS", "1")]
        a, b = tmp / "a", tmp / "b"
        assert mtp_train.main(base_args(root, a, *common, "--max-steps", "5")) == 0
        assert mtp_train.main(base_args(root, a, *common, "--max-steps", "10", "--resume")) == 0
        assert mtp_train.main(base_args(root, b, *common, "--max-steps", "10")) == 0
        torch.set_num_threads(int(THREADS))
        ha, hb = history(a), history(b)
        la, lb = [h["loss"] for h in ha], [h["loss"] for h in hb]
        diff = max(abs(x - y) for x, y in zip(la[5:], lb[5:]))
        pre = max(abs(x - y) for x, y in zip(la[:5], lb[:5]))
        record("resume", steps_a=len(la), steps_b=len(lb), resumed_losses=la[5:], uninterrupted_losses=lb[5:], max_abs_diff=diff,
               max_rel_diff=max(abs(x - y) / abs(y) for x, y in zip(la[5:], lb[5:])), steps_1_5_diff_between_runs=pre)
        assert len(la) == len(lb) == 10 and diff <= 1e-6, (la, lb)
    finally:
        shutil.rmtree(tmp)


def test_tools_end_to_end_cpu():
    """mtp_equiv (an fp8 MtpHeadTorch standing in for Hibana's CUDA head) and mtp_tf_accept --impl torch on synthetic
    shards: every line of their plumbing except Hibana's head itself."""
    skip_without_snapshot()
    import mtp_equiv
    import mtp_tf_accept
    tmp = Path(tempfile.mkdtemp())
    try:
        root = capture_root(tmp, rows=40)
        knobs = ["--expert-range", ER, "--vocab-rows", str(VOCAB), "--device", "cpu"]
        rc = mtp_equiv.main(["--capture-root", str(root), "--set", "heldout", "--snapshot", SNAP, "--standin", "--max-windows", "2",
                             "--chains-per-window", "12", "--batch", "5", "--out", str(tmp / "v.json"), "--dump-dir", str(tmp / "dump"),
                             "--bootstrap", "20", *knobs])
        v = json.loads((tmp / "v.json").read_text())
        dumps = sorted(p.name for p in (tmp / "dump").iterdir())
        rc2 = mtp_tf_accept.main(["--capture-root", str(root), "--set", "heldout", "--set", "general", "--impl", "torch", "--mode", "fp8",
                                  "--snapshot", SNAP, "--bootstrap", "50", "--out", str(tmp / "tf.json"), "--compute-dtype", "fp32", *knobs])
        tf = json.loads((tmp / "tf.json").read_text())
        record("tools_e2e", equiv_rc=rc, equiv_pass=v["pass"], equiv_chains=v["chains"], top1=v["top1_agreement"],
               cos_fp8=v["cos_d"]["fp8"]["mean"], taps_fp8={k: x.get("mean") for k, x in v["taps"]["fp8"].items()},
               checks_failed=[k for k, c in v["checks"].items() if not c["pass"]], dumps=dumps, tf_rc=rc2,
               tf_sets=list(tf["sets"]), tf_heldout_all=tf["sets"]["heldout"]["all"])
        # the stand-in compares the fp8 emulation with itself: identical fp8 drafts, d and taps, every fp8 check passes;
        # float mode on random rows (flat logits) is not expected to reach its line, so the exit code is 2
        failed = [k for k, c in v["checks"].items() if not c["pass"]]
        assert rc in (0, 2) and set(failed) <= {"top1_float"} and v["top1_agreement"]["fp8"]["pooled"] == 1.0 and rc2 == 0, failed
        assert len(dumps) == 2
        assert "exact" in tf["sets"]["heldout"]["all"] and "fast" in tf["sets"]["heldout"]["all"]
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    torch.set_num_threads(int(THREADS))
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}"[:2000], flush=True)
    print(json.dumps(RESULTS, indent=1, default=str))
    sys.exit(1 if fails else 0)
