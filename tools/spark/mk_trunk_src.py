"""Trunk-only view of a published W3 checkpoint (hard links), as flashnext_w3_experts fit --source needs."""
import json, os, sys, shutil
from pathlib import Path
pub, out = Path(sys.argv[1]), Path(sys.argv[2])
pm = json.loads((pub / "manifest.json").read_text())
tw = {n: e for n, e in pm["weights"].items() if ".experts." not in n}
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
for e in tw.values():
    os.link(pub / e["file"], out / e["file"])
m = {k: v for k, v in pm.items() if k != "experts"}
m["weights"] = tw
m["reports"] = {n: r for n, r in pm.get("reports", {}).items() if n in tw}
(out / "manifest.json").write_text(json.dumps(m, indent=1))
print(f"trunk-only source: {len(tw)} entries -> {out}")
