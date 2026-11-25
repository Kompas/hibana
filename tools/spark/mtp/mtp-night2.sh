#!/bin/bash
# Night 2: continue training the MTP head on qwen-trial117 agent windows only (captures from night 1), then evaluate.
# Phases (each time-boxed; later phases get what is left before END_AT):
#   0 agent eval, stock head on edp weights (also the image smoke test)  1 gap: stock head on stock weights
#   2 capture heldout + general  3 equivalence gate (torch head == Hibana head)  4 teacher-forced baseline
#   5 capture train  6 train (only if 3 passed)  7 write-back + Hibana-head check  8 agent eval tuned + compare
#   9 smoke serve of the tuned snapshot  10 lossy gate tuned (if time)
# vLLM is down for the duration; restore() always brings it back (EXIT trap; guard-mtp.sh is the backstop).
# Test run: DRY=1 END_AT=+20min bash mtp-night.sh  (fake phases, vLLM untouched, Telegram marked TEST)
set -uo pipefail
H=/home/tim/hibana-step2; M=$H/mtp; R=$H/recipe; S=$H/step3; LOG=$M/logs/night; CODE=/mtp/code
EDP=/home/tim/hibana-edp-hf; STOCK=/home/tim/hibana-stock-hf; TUNED=/home/tim/hibana-mtp-roots/agent2; TUNED1=/home/tim/hibana-mtp-roots/agent
RECAL=/data/w3-checkpoints/flashnext-edp-recal; PUB=/data/w3-checkpoints/flashnext-5c4c62a62853
NIGHT=${NIGHT:-$(date -d "+6 hours" +%Y%m%d)}
CAP=/data/mtp/capture; TR=/data/mtp/train-$NIGHT; EV=/data/mtp/eval-$NIGHT; HOSTDATA=/home/tim/.cache/hibana-flashnext
HEV=$HOSTDATA/mtp/eval-$NIGHT; HTR=$HOSTDATA/mtp/train-$NIGHT
DRY=${DRY:-0}
END_BY=$(date -d "${END_AT:-06:10}" +%s); [ "$END_BY" -le "$(date +%s)" ] && END_BY=$((END_BY + 86400))
RESERVE=${RESERVE:-1500}   # restore + slack
export HIBANA_IMAGE=${HIBANA_IMAGE:-hibana-spark:ce49b40}
mkdir -p "$LOG"; source "$S/tg.env"
cd "$R" || exit 1; CF="-f compose.yaml -f $M/code/override-mtp.yaml"
[ "$DRY" = 1 ] || exec >> "$LOG/night.log" 2>&1
log() { echo "[$(date +%T)] $*"; }
tg() { local p="Spark mtp:"; [ "$DRY" = 1 ] && p="Spark mtp TEST:"
  curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=$p $*" >/dev/null || true; log "TG: $*"; }
left() { echo $(( END_BY - $(date +%s) - RESERVE )); }
tmo() { local want=$1 l; l=$(left); [ "$l" -lt "$want" ] && want=$l; echo "$want"; }
mins() { echo $(( ($1 - 300) / 60 )); }   # tool --max-minutes: 5 min under the phase limit
crun() {  # [ROOT=dir] crun NAME MAXSEC CMD: one tool in a throwaway engine container, killed at its limit
  local name=$1 max=$2 t rc; shift 2; t=$(tmo "$max")
  [ "$t" -lt 240 ] && { log "$name: no time left"; return 99; }
  log "$name: start (limit ${t}s, root ${ROOT:-$EDP})"
  if [ "$DRY" = 1 ]; then sleep "${DRY_SLEEP:-5}"; rc=${DRY_RC[$name]:-0}; log "$name: rc=$rc (dry)"; return $rc; fi
  HF_CACHE=${ROOT:-$EDP} timeout --kill-after=60 "$t" docker compose $CF run --rm --no-deps -T --name "mtp-$name" \
    -e HIBANA_TARGET=flashnext -e HIBANA_SSD_DIR=/data/ssd-mtp-$name -e TL="$t" \
    --entrypoint bash hibana -c "cd /opt/hibana && $*" >> "$LOG/$name.log" 2>&1; rc=$?
  docker rm -f "mtp-$name" >/dev/null 2>&1; log "$name: rc=$rc"; return $rc
}
serve() {  # ROOT=dir serve PROFILE CKPT: Hibana on :8888, wait for health
  log "serve $1 ($2, root ${ROOT:-$EDP})"; [ "$DRY" = 1 ] && return 0
  HF_CACHE=${ROOT:-$EDP} PROFILE=$1 W3_CKPT=$2 docker compose $CF up -d --no-deps --force-recreate hibana >> "$LOG/serve.log" 2>&1
  for i in $(seq 1 90); do
    curl -sf -m 3 http://127.0.0.1:8888/health >/dev/null && { log "serve $1 healthy after $((i*10))s"; return 0; }
    [ "$(docker inspect -f '{{.State.Status}}' hibana-flashnext 2>/dev/null)" = running ] || break; sleep 10
  done
  docker logs --tail 60 hibana-flashnext >> "$LOG/serve-$1-fail.log" 2>&1; log "serve $1 FAILED"; return 1
}
unserve() { [ "$DRY" = 1 ] && return 0; docker compose $CF stop hibana >> "$LOG/serve.log" 2>&1; docker logs hibana-flashnext > "$LOG/serve-last.log" 2>&1; docker rm -f hibana-flashnext >/dev/null 2>&1; }
jget() { python3 -c "import json,sys
try:
  d=json.load(open(sys.argv[1]))
  for k in sys.argv[2].split('.'): d=d[int(k)] if isinstance(d,list) else d[k]
  print(round(d,3) if isinstance(d,float) else d)
except Exception: print('?')" "$@"; }
RESTORED=0; SUMMARY=""
add() { SUMMARY="$SUMMARY $*"; log "SUMMARY+ $*"; }
restore() {
  [ $RESTORED = 1 ] && return; RESTORED=1
  echo "$SUMMARY" > "$LOG/summary.txt"
  if [ "$DRY" = 1 ]; then tg "dry run done.$SUMMARY"; return; fi
  log "restore: stopping Hibana, starting vLLM"; unserve
  docker ps -aq --filter name=mtp- | xargs -r docker rm -f >/dev/null 2>&1
  docker start vllm_node >/dev/null
  for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && break; sleep 10; done
  if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null; then tg "done, vLLM back up $(date +%H:%M).$SUMMARY"; else tg "vLLM NOT healthy after 25 min, check: docker logs vllm_node.$SUMMARY"; fi
}
trap restore EXIT
trap 'log "signal"; exit 1' TERM INT
declare -A DRY_RC; for kv in ${DRY_FAIL:-}; do DRY_RC[${kv%%=*}]=${kv#*=}; done

free_gb=$(df -BG --output=avail "$HOSTDATA" | tail -1 | tr -dc 0-9)
[ "$free_gb" -lt 60 ] && { tg "not started: only ${free_gb} GB free"; RESTORED=1; exit 1; }
N1=${N1:-20261125}; EV1=/data/mtp/eval-$N1; HEV1=$HOSTDATA/mtp/eval-$N1
INIT=/data/mtp/train-$N1/mtp_trained.safetensors
[ "$DRY" = 1 ] || [ -f $HOSTDATA/mtp/train-$N1/mtp_trained.safetensors ] || { tg "not started: night-1 head missing"; RESTORED=1; exit 1; }
tg "start $(date +%H:%M), night 2 (qwen-only training), vLLM off until ${END_AT:-06:10} at the latest"
[ "$DRY" = 1 ] || { docker stop vllm_node >/dev/null && log "vLLM stopped"; }
AE="python $CODE/agent_eval.py --prompts /mtp/data/prompts-eval.jsonl --max-new 256"

# 1. train from night 1's head, qwen-trial117 windows only (+30% general); keep 75 min for write-back and evals
RES2=$RESERVE; RESERVE=$((RESERVE + 4500))
crun train 18000 "python $CODE/mtp_train.py --capture-root $CAP --snapshot /hf --out-dir $TR --init-weights $INIT --include-ids /mtp/data/train-ids-qwen.txt --lr 1.5e-5 --epochs 4 --max-minutes \$((TL/60-15)) --reserve-minutes 15 \$( [ -f $TR/ckpt/latest.pt ] && echo --resume )"
RESERVE=$RES2
RPT=$HTR/report.json
add "train2: stop $(jget $RPT stop_reason), steps $(jget $RPT steps), pool $(jget $RPT pool.agent_windows) agent windows, held-out fast tok/round@4 $(jget $RPT baseline.metric) (night-1 head) -> best $(jget $RPT best.metric) (step $(jget $RPT best.step))."
GAIN=$(python3 -c "import json
try:
  r=json.load(open('$RPT')); print(int(r['improved']))
except Exception: print(0)")
[ "$DRY" = 1 ] && GAIN=${DRY_GAIN:-1}
if [ "$GAIN" = 1 ] && { [ "$DRY" = 1 ] || [ -f $HTR/mtp_trained.safetensors ]; }; then
  log "writeback"
  if [ "$DRY" = 1 ] || timeout 900 python3 $M/code/snapshot_tools.py writeback --src-root $EDP --mtp-tensors $HTR/mtp_trained.safetensors --out-root $TUNED --force >> "$LOG/writeback.log" 2>&1; then
    ROOT=$TUNED crun tf-tuned2 1800 "python $CODE/mtp_tf_accept.py --capture-root $CAP --set heldout --set general --impl hibana --snapshot /hf --depth 4 --out $EV/tf-tuned2.json"
    add "tf heldout night-1 $(jget $HEV1/tf-tuned.json sets.heldout@served-w2.all.fast.tokens_per_round.3) -> night-2 $(jget $HEV/tf-tuned2.json sets.heldout@served-w2.all.fast.tokens_per_round.3) (stock $(jget $HEV1/tf-stock.json sets.heldout@served-w2.all.fast.tokens_per_round.3))."
    ROOT=$TUNED crun eval-tuned2 2400 "$AE --hf-root /hf --w3-checkpoint $RECAL --label edp-tunedhead2 --out $EV/tuned2.json --max-minutes \$((TL/60-5))" \
      && crun eval-cmp 300 "python $CODE/agent_eval.py --compare $EV1/stock.json $EV/tuned2.json --out $EV/compare-stock.json && python $CODE/agent_eval.py --compare $EV1/tuned.json $EV/tuned2.json --out $EV/compare-n1.json" \
      && add "AGENT night-2/stock tok/round: all $(jget $HEV/compare-stock.json groups.all.tok_per_round.ratio) $(jget $HEV/compare-stock.json groups.all.tok_per_round.ci95), qwen $(jget $HEV/compare-stock.json groups.pi/qwen-trial117.tok_per_round.ratio) $(jget $HEV/compare-stock.json groups.pi/qwen-trial117.tok_per_round.ci95), tok/s all $(jget $HEV/compare-stock.json groups.all.decode_tps.ratio); vs night-1 all $(jget $HEV/compare-n1.json groups.all.tok_per_round.ratio)."
    if [ "$(left)" -gt 1200 ] && ROOT=$TUNED serve fast "$RECAL"; then
      [ "$DRY" = 1 ] || timeout 1200 python3 "$S/step2_test.py" http://127.0.0.1:8888 "$LOG/smoke-tuned2.json" > "$LOG/smoke-tuned2.log" 2>&1
      add "smoke: $(head -1 "$LOG/smoke-tuned2.log" 2>/dev/null)."; unserve
    else add "smoke serve skipped/failed."; fi
    ROOT=$TUNED crun gate-tuned2 3600 "python tools/lossy_gate.py --target flashnext --name edp-tunedhead2 --profile fast --env HIBANA_W3_CHECKPOINT=$RECAL --through inprocess --hold-minutes 45" \
      && add "gate: $(grep -E '^VERDICT' "$LOG/gate-tuned2.log" | tail -1 | cut -c1-200)." || add "gate: not run/finished."
  else add "writeback FAILED."; fi
else add "night 2 did not improve on night 1's head: nothing written."; fi
log "all phases done"
