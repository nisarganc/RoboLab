#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
CFG_FILE="${CFG_FILE:-valpa-angledreach/droid-224px-8f-dual.yaml}"
TASK="${TASK:-AngledReachDrillTask}"
TASK_DIRS="${TASK_DIRS:-wm_tasks/angledreach}"
TARGET_STATUS="${TARGET_STATUS:-assets/wm_tasks/${TASK}/status.json}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8300}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
HEADLESS="${HEADLESS:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"

server_args=(
    --cfg-file "$CFG_FILE"
    --host 0.0.0.0
    --port "$REMOTE_PORT"
    --target-status "$TARGET_STATUS"
    --random-seed "$RANDOM_SEED"
)
if [[ -n "$OUTPUT_DIR" ]]; then
    server_args+=(--output-dir "$OUTPUT_DIR")
fi

mkdir -p output/world_model_patch_loss
server_log="output/world_model_patch_loss/server_${REMOTE_PORT}.log"
"$ISAAC_PYTHON" valpa/inference/serve_policy_quant.py "${server_args[@]}" \
    >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

port_open() {
    (echo >"/dev/tcp/$REMOTE_HOST/$REMOTE_PORT") >/dev/null 2>&1
}

deadline=$((SECONDS + SERVER_START_TIMEOUT))
until port_open; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Analysis server exited; see $server_log" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for analysis server; see $server_log" >&2
        exit 1
    fi
    sleep 5
done

runner_args=(
    --task "$TASK"
    --task-dirs "$TASK_DIRS"
    --remote-host "$REMOTE_HOST"
    --remote-port "$REMOTE_PORT"
)
if [[ "$HEADLESS" == "1" ]]; then
    runner_args+=(--headless)
fi
"$ISAAC_PYTHON" examples/policy/run_world_model_patch_loss.py "${runner_args[@]}"
