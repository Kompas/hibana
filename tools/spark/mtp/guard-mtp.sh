#!/bin/bash
# 07:00 backstop for mtp-night.sh: stop whatever it is doing and bring vLLM back.
S=/home/tim/hibana-step2/step3; source $S/tg.env
msg() { curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=Spark mtp guard: $*" >/dev/null; }
if systemctl is-active -q hibana-mtp.service; then
  systemctl kill -s TERM hibana-mtp.service; sleep 90; systemctl stop hibana-mtp.service
  docker ps -aq --filter name=mtp- | xargs -r docker rm -f; docker rm -f hibana-flashnext; docker start vllm_node
  msg "stopped the run at $(date +%H:%M), vLLM started"
elif ! docker ps --format '{{.Names}}' | grep -qx vllm_node; then
  docker ps -aq --filter name=mtp- | xargs -r docker rm -f; docker rm -f hibana-flashnext; docker start vllm_node
  msg "vLLM was not running at $(date +%H:%M), started it"
fi
