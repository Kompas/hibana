# tools/spark — our one-Spark setup on top of Hibana

This is the `spark` branch of the Kompas fork: upstream `main` plus the fixes and tools we run on our DGX Spark
(GB10, 128 GB). Branches: `main` mirrors upstream; `fix/*` are small upstream-bound fixes (each also a PR to
sharar-project/hibana); `exp/*` are speed experiments, merged here only after the lossy gate and tool-eval-bench.

## Image

```bash
docker build -f tools/spark/docker/Dockerfile --build-arg HIBANA_COMMIT=$(git rev-parse HEAD) \
  -t hibana-spark:$(git rev-parse --short HEAD) tools/spark/docker
```

The compose file is the dedene recipe's (`~/hibana-step2/recipe` on the Spark); set `HIBANA_IMAGE=hibana-spark:<sha>`
and add `-f tools/spark/override.yaml` (W3 checkpoint by `W3_CKPT`, the config file with `/data` and `/hf` paths).

## Tools

| File | What it does |
|---|---|
| `recal-night.sh` | Unattended recalibration of the 3-bit copies for changed weights (e.g. an abliterated model): exact serve → `generate` → `build` → trunk GPTQ for the changed matrices only → expert capture → fit of the changed layers → `merge.py` → fast smoke test → lossy gate. Per-phase time limits, vLLM restored on exit, Telegram summary. `END_AT=HH:MM` sets the deadline. |
| `merge.py` | Published W3 checkpoint + refitted trunk matrices + refitted expert layers → one serving checkpoint (hard links, hashes checked). |
| `mk_trunk_src.py` | Trunk-only view of a checkpoint (hard links), which `flashnext_w3_experts fit --source` requires. |
| `partial_to_answers.py` | Turns `generate`'s partial answers into an answers file when it hit its time limit. |
| `smoke_test.py` | Speed (prose/code, median of 3), one tool call, 8 borderline prompts; same script for vLLM and Hibana. |
| `teb.sh` | tool-eval-bench 3.1.0 hardmode on vLLM, Hibana exact and Hibana fast, vLLM restored afterwards. |
| `guard.sh` | Backstop timer: stops a running night job and starts vLLM. |
| `tests/test_tool_args_anyof.py` | The tool-argument patch's test (from dedene's recipe). |

Paths in the scripts are our Spark's (`/home/tim/hibana-step2`, `/data` = `~/.cache/hibana-flashnext`). `tg.env`
(Telegram token) stays local and is not tracked.
