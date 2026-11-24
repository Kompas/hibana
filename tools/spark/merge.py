"""Published W3 checkpoint + recalibrated trunk matrices + refitted expert layers -> one serving checkpoint."""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
pub, trunk, expdir, out, changed_path, layers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

pm = json.loads((pub / "manifest.json").read_text())
changed = set(json.loads(changed_path.read_text()))
tm = json.loads((trunk / "manifest.json").read_text())
ep = json.loads((expdir / "progress.json").read_text())["weights"]
exp_names = {f"layers.{i}.experts.{k}" for i in map(int, layers.split(",")) for k in ("w13", "w2")}
if set(tm["weights"]) != changed:
    sys.exit(f"trunk calibration covers {len(tm['weights'])} matrices, expected the {len(changed)} changed ones")
if not exp_names <= set(ep):
    sys.exit(f"expert fit lacks {sorted(exp_names - set(ep))}")
tmp = out.with_name(out.name + ".tmp")
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)
weights, reports = {}, dict(pm.get("reports", {}))
for name, entry in pm["weights"].items():
    if name in changed:
        src, e = trunk, tm["weights"][name]
        reports[name] = tm["reports"].get(name)
    elif name in exp_names:
        src, e = expdir, ep[name]
    else:
        src, e = pub, entry
    for k in ("out_features", "in_features"):
        if e.get(k) != entry.get(k):
            sys.exit(f"{name}: {k} {e.get(k)} != published {entry.get(k)}")
    if e.get("codebook") != entry.get("codebook"):
        print(f"note: {name} codebook differs from the published entry", flush=True)
    os.link(src / e["file"], tmp / e["file"])
    weights[name] = e
for name in sorted(changed | exp_names):
    if sha256(tmp / weights[name]["file"]) != weights[name]["sha256"]:
        sys.exit(f"checksum mismatch {name}")
m = dict(pm)
m["weights"], m["reports"] = weights, reports
m["edp_recal"] = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": str(pub.resolve()),
                  "base_manifest_sha256": sha256(pub / "manifest.json"),
                  "trunk_gptq": sorted(changed), "trunk_manifest_sha256": sha256(trunk / "manifest.json"),
                  "experts_refit_layers": layers, "note": "edp1096 abliterated weights: changed matrices refitted, rest published (QAD trunk, GPTQ experts)"}
(tmp / "manifest.json").write_text(json.dumps(m, indent=1))
if out.exists():
    shutil.rmtree(out)
tmp.rename(out)
print(f"merged: {len(weights)} entries ({len(changed)} trunk + {len(exp_names)} expert replaced) -> {out} manifest {sha256(out / 'manifest.json')}")
