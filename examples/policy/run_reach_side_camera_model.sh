#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_CONFIG:?Set MODEL_CONFIG to a valpa-reach YAML filename}"
: "${REMOTE_PORT:?Set REMOTE_PORT to a unique policy-server port}"

ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
DEVICE="${DEVICE:-cuda:0}"
TASK="${TASK:-ReachBananaTask}"
NUM_RUNS="${NUM_RUNS:-30}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_MODE="${VIDEO_MODE:-sensor}"
HEADLESS="${HEADLESS:-1}"
SIDE_CAMERA_NAME="${SIDE_CAMERA_NAME:-over_shoulder_right_camera}"
POSITION_X="${POSITION_X:-0.2}"
POSITION_Y="${POSITION_Y:-0.2}"
POSITION_Z="${POSITION_Z:-0.1}"
ANGLE_ROLL="${ANGLE_ROLL:-0.2}"
ANGLE_PITCH="${ANGLE_PITCH:-0.2}"
ANGLE_YAW="${ANGLE_YAW:-0.2}"
RANDOMIZATION_SEED="${RANDOMIZATION_SEED:-20260821}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/output}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUTPUT_ROOT/side_camera_server_logs}"
VALPA_CONFIG_DIR="${VALPA_CONFIG_DIR:-$PWD/valpa/configs/inference/valpa-reach}"
ARCHIVE_AFTER_RUNS="${ARCHIVE_AFTER_RUNS:-1}"

TASKS=(
    ReachBananaTask
    # ReachCoffeeCanTask
    # ReachCoffeePotTask
    # ReachOrangeJuiceCartonTask
    # ReachPitcherTask
    # ReachSpoonBigTask
    # ReachYogurtCupTask
    # ReachAppleTask
    ReachBagelTask
    # ReachOrangeTask
    # ReachCeramicMugTask
)

task_is_valid=0
for available_task in "${TASKS[@]}"; do
    if [[ "$TASK" == "$available_task" ]]; then
        task_is_valid=1
        break
    fi
done
if (( task_is_valid == 0 )); then
    echo "Unknown TASK: $TASK"
    echo "Choose one of: ${TASKS[*]}"
    exit 2
fi
if (( NUM_RUNS < 1 || NUM_ENVS < 1 )); then
    echo "NUM_RUNS and NUM_ENVS must both be positive."
    exit 2
fi

cfg_path="$VALPA_CONFIG_DIR/$MODEL_CONFIG"
if [[ ! -f "$cfg_path" ]]; then
    echo "Model config not found: $cfg_path"
    exit 1
fi

cfg_name="${MODEL_CONFIG%.yaml}"
output_folder_name="${OUTPUT_FOLDER_NAME:-${cfg_name}_side_camera_${TASK}_seed${RANDOMIZATION_SEED}${OUTPUT_SUFFIX}}"
output_dir="$OUTPUT_ROOT/$output_folder_name"
archive_file="$OUTPUT_ROOT/${output_folder_name}.zip"
server_log="$SERVER_LOG_DIR/${cfg_name}${OUTPUT_SUFFIX}_serve_policy.log"
SERVER_PID=""

port_open() {
    (echo >"/dev/tcp/$REMOTE_HOST/$REMOTE_PORT") >/dev/null 2>&1
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

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

check_output_complete() {
    local episode_results_file="$output_dir/episode_results.jsonl"
    local expected_episodes=$((NUM_RUNS * NUM_ENVS))

    if [[ ! -f "$episode_results_file" ]]; then
        echo "Missing episode results file: $episode_results_file"
        return 1
    fi

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "$ISAAC_PYTHON" - "$episode_results_file" "$TASK" "$expected_episodes" <<'PY'
import json
import pathlib
import sys

results_path = pathlib.Path(sys.argv[1])
task = sys.argv[2]
expected = int(sys.argv[3])
episodes = set()

with results_path.open("r", encoding="utf-8") as stream:
    for line in stream:
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if result.get("env_name") == task and isinstance(result.get("episode"), int):
            episodes.add(result["episode"])

if len(episodes) < expected:
    raise SystemExit(f"Incomplete output for {task}: {len(episodes)}/{expected} episodes")
PY
}

trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

if [[ -f "$archive_file" ]]; then
    echo "Archive already exists; nothing to run: $archive_file"
    exit 0
fi
if port_open; then
    echo "Port $REMOTE_HOST:$REMOTE_PORT is already occupied; refusing to replace its server."
    exit 1
fi

mkdir -p "$SERVER_LOG_DIR"
echo "=== Starting $MODEL_CONFIG on $REMOTE_HOST:$REMOTE_PORT ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" valpa/inference/serve_policy.py \
    --cfg-file "valpa-reach/$MODEL_CONFIG" \
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

echo "=== Evaluating $TASK for $NUM_RUNS runs x $NUM_ENVS envs ==="
echo "=== Shared randomization schedule: base seed $RANDOMIZATION_SEED; run N uses seed RANDOMIZATION_SEED + N ==="
echo "=== Side camera: $SIDE_CAMERA_NAME; position ±($POSITION_X, $POSITION_Y, $POSITION_Z) m; angle ±($ANGLE_ROLL, $ANGLE_PITCH, $ANGLE_YAW) rad ==="
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" examples/policy/run_reach_eval.py \
    --policy valpa \
    --task "$TASK" \
    --num-runs "$NUM_RUNS" \
    --num-envs "$NUM_ENVS" \
    --device "$DEVICE" \
    --video-mode "$VIDEO_MODE" \
    --output-folder-name "$output_folder_name" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    --randomize-side-camera \
    --side-camera-name "$SIDE_CAMERA_NAME" \
    --side-camera-position-range "$POSITION_X" "$POSITION_Y" "$POSITION_Z" \
    --side-camera-angle-range "$ANGLE_ROLL" "$ANGLE_PITCH" "$ANGLE_YAW" \
    --run-randomization-seed "$RANDOMIZATION_SEED" \
    "${EXTRA_ARGS[@]}" \
    --kit_args "--reset-user --/rtx/post/aa/op=1 --/rtx/post/dlss/enabled=false --/rtx-transient/dlssg/enabled=0 --/rtx-transient/dldenoiser/enabled=0"

check_output_complete

if [[ "$ARCHIVE_AFTER_RUNS" == "1" ]]; then
    echo "=== Archiving $output_dir -> $archive_file ==="
    (
        cd "$OUTPUT_ROOT"
        zip -r -q "$(basename "$archive_file")" "$output_folder_name"
    )
fi

echo "=== Completed $MODEL_CONFIG on $TASK ==="
