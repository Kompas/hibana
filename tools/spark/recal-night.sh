#!/bin/bash
# Recalibrate Hibana's 3-bit copies for the edp1096 abliterated weights, overnight.
# vLLM is down for the duration; restore() always brings it back (EXIT trap; guard.sh is the 07:00 backstop).
set -uo pipefail
H=/home/tim/hibana-step2; S=$H/step3; R=$H/recipe; LOG=$S/logs; OUT=$S/out
D=/data/step3; PUB=/data/w3-checkpoints/flashnext-5c4c62a62853; RECAL=/data/w3-checkpoints/flashnext-edp-recal
LAYERS=2,4,30,46,47
END_BY=$(date -d "${END_AT:-06:40}" +%s)
RESERVE=1800   # restore + slack
mkdir -p "$LOG" "$OUT"; source "$S/tg.env"
cd "$R"; CF="-f compose.yaml -f $S/override.yaml"
exec >> "$LOG/step3.log" 2>&1
log() { echo "[$(date +%T)] $*"; }
tg() { curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=Spark step3: $*" >/dev/null || true; log "TG: $*"; }
left() { echo $(( END_BY - $(date +%s) - RESERVE )); }
tmo() { local want=$1 l; l=$(left); [ "$l" -lt "$want" ] && want=$l; echo "$want"; }
crun() {  # crun NAME MAXSEC CMD...: one tool in a throwaway engine container, killed at its limit
  local name=$1 max=$2 t rc; shift 2; t=$(tmo "$max")
  [ "$t" -lt 120 ] && { log "$name: no time left"; return 99; }
  log "$name: start (limit ${t}s)"
  timeout --kill-after=30 "$t" docker compose $CF run --rm --no-deps -T --name "s3-$name" -e HIBANA_TARGET=flashnext \
    --entrypoint bash hibana -c "cd /opt/hibana && $*" >> "$LOG/$name.log" 2>&1; rc=$?
  docker rm -f "s3-$name" >/dev/null 2>&1; log "$name: rc=$rc"; return $rc
}
serve() {  # serve PROFILE CKPT: Hibana on :8888, wait for health
  log "serve $1 ($2)"; PROFILE=$1 W3_CKPT=$2 docker compose $CF up -d --no-deps --force-recreate hibana >> "$LOG/serve.log" 2>&1
  for i in $(seq 1 90); do
    curl -sf -m 3 http://127.0.0.1:8888/health >/dev/null && { log "serve $1 healthy after $((i*10))s"; return 0; }
    [ "$(docker inspect -f '{{.State.Status}}' hibana-flashnext 2>/dev/null)" = running ] || break; sleep 10
  done
  docker logs --tail 40 hibana-flashnext >> "$LOG/serve-$1-fail.log" 2>&1; log "serve $1 FAILED"; return 1
}
unserve() { docker compose $CF stop hibana >> "$LOG/serve.log" 2>&1; docker logs hibana-flashnext > "$LOG/serve-last.log" 2>&1; docker rm -f hibana-flashnext >/dev/null 2>&1; }
RESTORED=0; SUMMARY=""
restore() {
  [ $RESTORED = 1 ] && return; RESTORED=1
  log "restore: stopping Hibana, starting vLLM"; unserve
  docker ps -aq --filter name=s3- | xargs -r docker rm -f >/dev/null 2>&1
  docker start vllm_node >/dev/null
  for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && break; sleep 10; done
  if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null; then tg "done, vLLM back up $(date +%H:%M). ${SUMMARY:-no results}"; else tg "vLLM NOT healthy after 25 min, check: docker logs vllm_node"; fi
}
fail() { SUMMARY="STOPPED at $1 (see ~/hibana-step2/step3/logs/$1.log). $SUMMARY"; log "FAIL $1"; exit 1; }
trap restore EXIT
trap 'log "signal"; exit 1' TERM INT

tg "start $(date +%H:%M), continuation, vLLM off until ${END_AT:-06:40} at the latest"
docker stop vllm_node >/dev/null && log "vLLM stopped"

# 1. calibration answers from the abliterated model itself (exact, thinking on)
if [ ! -e "$OUT/answers.ok" ]; then
  serve exact auto || fail serve-exact
  crun generate 4800 "python -m hibana.weights.calibration_inputs generate --prompts $D/prompts-160.json --url http://127.0.0.1:8888 --output $D/answers.json --concurrency 4"
  rc=$?
  if [ $rc -ne 0 ]; then crun partial 300 "python /opt/hibana/tools/spark/partial_to_answers.py $D/answers.json.partial.jsonl $D/answers.json" || fail generate; fi
  unserve; touch "$OUT/answers.ok"
fi
crun build 900 "python -m hibana.weights.calibration_inputs build --target flashnext --answers $D/answers.json --output $D/inputs.json" || fail build
SUMMARY="samples: $(grep -o '[0-9]* answered turns -> [0-9]* samples, [0-9]* tokens' "$LOG/build.log" | tail -1)."

# 2. GPTQ for the 96 changed trunk matrices only
crun trunk 6000 "[ -f $D/trunk96/manifest.json ] || python -m hibana.weights.flashnext_w3_calibrate --inputs $D/inputs.json --output $D/trunk96 --exclude '^(?!layers\.\d+\.(mixer\.out|mixer\.o|mlp\.shared_down)\$)'" || fail trunk
grep -q "^96 projections to fit" "$LOG/trunk.log" || [ -f "$H/.trunk-skip-check" ] || fail trunk-count

# 3. routed experts of the 5 changed layers
crun capture 4500 "[ -f $D/act/capture.json ] || python -m hibana.weights.flashnext_w3_experts capture --inputs $D/inputs.json --activations $D/act" || fail capture
crun fit 4500 "python /opt/hibana/tools/spark/mk_trunk_src.py $PUB $D/src-trunk && python -m hibana.weights.flashnext_w3_experts fit --source $D/src-trunk --output $D/experts5 --activations $D/act --layers $LAYERS" || fail fit
crun merge 1200 "python /opt/hibana/tools/spark/merge.py $PUB $D/trunk96 $D/experts5.partial $RECAL $D/changed-names.json $LAYERS" || fail merge
SUMMARY="$SUMMARY merged checkpoint ok."

# 4. fast with the recalibrated copies: does it load, speed, tools, refusals (same test as step 2)
if [ "$(left)" -gt 1200 ] && serve fast "$RECAL"; then
  timeout 1500 python3 "$S/smoke_test.py" http://127.0.0.1:8888 "$OUT/fast-recal.json" > "$LOG/test-fast-recal.log" 2>&1
  SUMMARY="$SUMMARY fast-recal: $(head -1 "$LOG/test-fast-recal.log")."
  unserve
else SUMMARY="$SUMMARY fast-recal serve failed or no time."; fi

# 5. lossy gate layers 1-2 (in-process, fixed prompts): recalibrated vs exact, then the unchanged copies vs exact
for arm in recal:"$RECAL" stock:"$PUB"; do
  n=${arm%%:*}; c=${arm#*:}
  crun gate-$n 4800 "python tools/lossy_gate.py --target flashnext --name edp-$n --profile fast --env HIBANA_W3_CHECKPOINT=$c --through inprocess --hold-minutes 75" \
    && SUMMARY="$SUMMARY gate-$n: $(grep -iE 'verdict|KL' "$LOG/gate-$n.log" | tail -2 | tr '\n' ' ')" || SUMMARY="$SUMMARY gate-$n: not finished."
done
log "all phases done"
