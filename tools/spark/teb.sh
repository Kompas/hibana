#!/bin/bash
# tool-eval-bench 3.1.0 hardmode: vLLM trial117 vs Hibana exact vs Hibana fast (edp-recal). vLLM back at the end, 10:40 at the latest.
set -uo pipefail
H=/home/tim/hibana-step2; S=$H/step3; R=$H/recipe; LOG=$S/logs; OUT=$S/out/teb; RECAL=/data/w3-checkpoints/flashnext-edp-recal
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$OUT"; source "$S/tg.env"; cd "$R"; CF="-f compose.yaml -f $S/override.yaml"
exec >> "$LOG/teb.log" 2>&1
END_BY=$(date -d "10:40" +%s); [ "$END_BY" -lt "$(date +%s)" ] && END_BY=$(date -d "tomorrow 10:40" +%s)
log() { echo "[$(date +%T)] $*"; }
tg() { curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=Spark tool-eval: $*" >/dev/null || true; log "TG: $*"; }
bench() {  # bench LABEL URL MODEL
  local t=$(( END_BY - $(date +%s) - 1500 )); [ $t -gt 7200 ] && t=7200; [ $t -lt 300 ] && { log "$1: no time"; return 99; }
  log "$1: start (limit ${t}s)"
  timeout "$t" tool-eval-bench --base-url "$2" --model "$3" --hardmode --seed 42 --temperature 0 --parallel 4 \
     --no-live --label "$1" --json-file "$OUT/$1.json" > "$LOG/teb-$1.log" 2>&1; log "$1: rc=$?"
  SUMMARY="$SUMMARY $1: $(python3 -c "import json;d=json.load(open('$OUT/$1.json'));print(d.get('score', d.get('summary',{}).get('score','?')))" 2>/dev/null || echo '?')."
}
serve() {
  log "serve $1"; PROFILE=$1 W3_CKPT=$2 docker compose $CF up -d --no-deps --force-recreate hibana >> "$LOG/serve.log" 2>&1
  for i in $(seq 1 90); do curl -sf -m 3 http://127.0.0.1:8888/health >/dev/null && { log "serve $1 healthy"; return 0; }; sleep 10; done; return 1
}
unserve() { docker compose $CF stop hibana >> "$LOG/serve.log" 2>&1; docker rm -f hibana-flashnext >/dev/null 2>&1; }
RESTORED=0; SUMMARY=""
restore() {
  [ $RESTORED = 1 ] && return; RESTORED=1; unserve; docker start vllm_node >/dev/null
  for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && break; sleep 10; done
  curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null && tg "done, vLLM back $(date +%H:%M).$SUMMARY" || tg "vLLM NOT healthy, check docker logs vllm_node"
}
trap restore EXIT; trap 'exit 1' TERM INT
tg "start $(date +%H:%M): vLLM arm first, then vLLM off for Hibana exact + fast"
bench vllm-trial117 http://127.0.0.1:8000 qwen-trial117
docker stop vllm_node >/dev/null; log "vLLM stopped"
serve exact auto && bench hibana-exact http://127.0.0.1:8888 RadixArk/Qwen3.8-Flash-Next-NVFP4; unserve
serve fast "$RECAL" && bench hibana-fast-recal http://127.0.0.1:8888 RadixArk/Qwen3.8-Flash-Next-NVFP4; unserve
log "all done"
