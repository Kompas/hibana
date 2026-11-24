"""Fallback when generate hit its time limit: the answers it saved so far as an answers file."""
import json, sys
from pathlib import Path
partial, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in partial.read_text().split("\n") if l.strip()]
out.write_text(json.dumps({"version": "calibration-answers-v2", "note": "from partial (time limit)", "items": rows}))
print(f"{len(rows)} answers from {partial} -> {out}")
