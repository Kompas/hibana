#!/bin/bash
# 07:00 backstop: whatever step3 is doing, stop it and bring vLLM back.
S=/home/tim/hibana-step2/step3; source $S/tg.env
if systemctl is-active -q hibana-step3.service || systemctl is-active -q hibana-step3-cont.service; then
  systemctl kill -s TERM hibana-step3.service hibana-step3-cont.service; sleep 60; systemctl stop hibana-step3.service hibana-step3-cont.service
  docker ps -aq --filter name=s3- | xargs -r docker rm -f; docker rm -f hibana-flashnext; docker start vllm_node
  curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=Spark step3: guard stopped the run at 11:00, vLLM started" >/dev/null
elif ! docker ps --format '{{.Names}}' | grep -qx vllm_node; then
  docker rm -f hibana-flashnext; docker start vllm_node
  curl -s -m 20 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=Spark step3 guard: vLLM was not running at 11:00, started it" >/dev/null
fi
