#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_CONFIG:?Set MODEL_CONFIG to a valpa-angledreach YAML filename}"
: "${REMOTE_PORT:?Set REMOTE_PORT to a unique policy-server port}"

ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
DEVICE="${DEVICE:-cuda:0}"
NUM_BACKGROUNDS="${NUM_BACKGROUNDS:-5}"
BACKGROUND_SEED="${BACKGROUND_SEED:-1}"
NUM_RUNS_PER_VARIANT="${NUM_RUNS_PER_VARIANT:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_MODE="${VIDEO_MODE:-sensor}"
HEADLESS="${HEADLESS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/output}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUTPUT_ROOT/bg_distractor_server_logs}"
VALPA_CONFIG_DIR="${VALPA_CONFIG_DIR:-$PWD/valpa/configs/inference/valpa-angledreach}"

SERVER_PID=""

port_open() {
    (echo >"/dev/tcp/$REMOTE_HOST/$REMOTE_PORT") >/dev/null 2>&1
}

cleanup_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))
    until port_open; do
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 1
        fi
        if (( SECONDS >= deadline )); then
            return 1
        fi
        sleep 5
    done
}

trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

cfg_path="$VALPA_CONFIG_DIR/$MODEL_CONFIG"
if [[ ! -f "$cfg_path" ]]; then
    echo "Model config not found: $cfg_path"
    exit 1
fi
if port_open; then
    echo "Port $REMOTE_HOST:$REMOTE_PORT is already occupied."
    exit 1
fi

mkdir -p "$SERVER_LOG_DIR"
cfg_name="${MODEL_CONFIG%.yaml}"
server_log="$SERVER_LOG_DIR/${cfg_name}.log"
eval_log="$SERVER_LOG_DIR/${cfg_name}_eval.log"

echo "=== Starting $MODEL_CONFIG on $REMOTE_HOST:$REMOTE_PORT ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" valpa/inference/serve_policy.py \
    --cfg-file "valpa-angledreach/$MODEL_CONFIG" \
    --host "$SERVER_HOST" \
    --port "$REMOTE_PORT" \
    >"$server_log" 2>&1 &
SERVER_PID="$!"

if ! wait_for_server; then
    echo "Policy server failed to start. Last log lines:"
    tail -n 80 "$server_log" || true
    exit 1
fi

EXTRA_ARGS=()
if [[ "$HEADLESS" == "1" ]]; then
    EXTRA_ARGS+=(--headless)
fi

echo "=== Evaluating 5 bg_distractor tasks x ${NUM_BACKGROUNDS} backgrounds x 5 object counts ==="
set +e
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" examples/policy/run_eval_background_variation.py \
    --angled-reach \
    --policy valpa \
    --task-dirs wm_tasks/bg_distractor \
    --num-backgrounds "$NUM_BACKGROUNDS" \
    --background-seed "$BACKGROUND_SEED" \
    --num-runs "$NUM_RUNS_PER_VARIANT" \
    --num-envs "$NUM_ENVS" \
    --video-mode "$VIDEO_MODE" \
    --device "$DEVICE" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$eval_log"
eval_status="${PIPESTATUS[0]}"
set -e

if (( eval_status != 0 )) || grep -Eq '^Traceback \(most recent call last\):|\[RoboLab\] Terminated with error:' "$eval_log"; then
    echo "Evaluation failed for $MODEL_CONFIG; see $eval_log"
    exit 1
fi

echo "=== Completed $MODEL_CONFIG ==="
