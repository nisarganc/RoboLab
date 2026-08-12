set -euo pipefail

export ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-9000}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/anvme/workspace/v106be10-valpa-robolab/RoboLab/output}"
HEADLESS="${HEADLESS:-1}"
VIDEO_MODE="${VIDEO_MODE:-sensor}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-}"
DEVICE="${DEVICE:-cuda:0}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUTPUT_ROOT/${REMOTE_PORT}}"
ARCHIVE_AFTER_MODEL="${ARCHIVE_AFTER_MODEL:-1}"
DELETE_UNZIPPED_AFTER_ARCHIVE="${DELETE_UNZIPPED_AFTER_ARCHIVE:-1}"
NUM_RUNS_PER_TASK="${NUM_RUNS_PER_TASK:-10}"

MODEL_CONFIGS=(
    droid-224px-8f-right.yaml
)

TASKS=(
    ReachBananaTask
    ReachCoffeeCanTask
    ReachCoffeePotTask
    ReachOrangeJuiceCartonTask
    ReachPitcherTask
    ReachSpoonBigTask
    ReachYogurtCupTask
    ReachAppleTask
    ReachBagelTask
    ReachOrangeTask
    ReachCeramicMugTask
)

SERVER_PID=""
MODEL_NAMES=()

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
    
    # Wait for port to be released
    local max_wait=30
    local elapsed=0
    while port_open && (( elapsed < max_wait )); do
        sleep 1
        ((++elapsed))
    done
}

cfg_model_name() {
    local cfg_file="$1"

    awk '
        function trim(s) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
            return s
        }

        function strip_quotes(s) {
            first = substr(s, 1, 1)
            last = substr(s, length(s), 1)

            if ((first == "\"" && last == "\"") || (first == "'"'"'" && last == "'"'"'")) {
                return substr(s, 2, length(s) - 2)
            }

            return s
        }

        /^[[:space:]]*log[[:space:]]*:[[:space:]]*($|#)/ {
            in_log = 1
            next
        }

        in_log && /^[^[:space:]#][^:]*[[:space:]]*:/ {
            exit 1
        }

        in_log && /^[[:space:]]*modelname[[:space:]]*:/ {
            sub(/^[[:space:]]*modelname[[:space:]]*:[[:space:]]*/, "")
            sub(/[[:space:]]+#.*$/, "")

            value = strip_quotes(trim($0))

            if (value == "") {
                exit 1
            }

            print value
            found = 1
            exit
        }

        END {
            if (!found) {
                exit 1
            }
        }
    ' "$cfg_file"
}

hosted_model_name() {
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    REMOTE_HOST="$REMOTE_HOST" \
    REMOTE_PORT="$REMOTE_PORT" \
    "$ISAAC_PYTHON" -c '
import os
import pickle
import socket
import struct

header = struct.Struct("!Q")
host = os.environ["REMOTE_HOST"]
port = int(os.environ["REMOTE_PORT"])
payload = pickle.dumps({"method": "metadata"}, protocol=pickle.HIGHEST_PROTOCOL)

with socket.create_connection((host, port), timeout=30) as sock:
    sock.sendall(header.pack(len(payload)) + payload)
    size_data = b""
    while len(size_data) < header.size:
        chunk = sock.recv(header.size - len(size_data))
        if not chunk:
            raise ConnectionError("VALPA policy socket closed while reading metadata header")
        size_data += chunk
    size = header.unpack(size_data)[0]
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("VALPA policy socket closed while reading metadata payload")
        data += chunk

response = pickle.loads(data)
if error := response.get("error"):
    raise RuntimeError(error)
print(response["modelname"])
'
}

archive_model_output() {
    local model_name="$1"
    local model_dir="$OUTPUT_ROOT/$model_name"
    local archive_file="$OUTPUT_ROOT/${model_name}.zip"

    if [[ "$ARCHIVE_AFTER_MODEL" != "1" ]]; then
        return 0
    fi

    if [[ ! -d "$model_dir" ]]; then
        echo "Expected output folder not found, skipping archive: $model_dir"
        return 1
    fi


    echo "=== Zipping $model_dir -> $archive_file ==="
    rm -f "$archive_file"
    (
        cd "$OUTPUT_ROOT"
        zip -r -q "$(basename "$archive_file")" "$model_name"
    )

    if [[ "$DELETE_UNZIPPED_AFTER_ARCHIVE" == "1" ]]; then
        echo "=== Removing expanded output folder after successful zip: $model_dir ==="
        rm -rf "$model_dir"
    fi
}

check_model_output_complete() {
    local model_name="$1"
    local expected_runs="$2"
    local model_dir="$OUTPUT_ROOT/$model_name"
    local episode_results_file="$model_dir/episode_results.jsonl"

    if [[ ! -f "$episode_results_file" ]]; then
        echo "Missing episode results file: $episode_results_file"
        return 1
    fi

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "$ISAAC_PYTHON" - "$episode_results_file" "$expected_runs" "${TASKS[@]}" <<'PY'
import json
import pathlib
import sys

episode_results_file = pathlib.Path(sys.argv[1])
expected_runs = int(sys.argv[2])
tasks = sys.argv[3:]

episodes_per_task = {task: set() for task in tasks}

with episode_results_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue

        env_name = ep.get("env_name")
        episode = ep.get("episode")

        if env_name in episodes_per_task and isinstance(episode, int):
            episodes_per_task[env_name].add(episode)

incomplete = []
for task in tasks:
    count = len(episodes_per_task[task])
    if count < expected_runs:
        incomplete.append((task, count, expected_runs))

if incomplete:
    print("Model output is incomplete. Missing episodes:", file=sys.stderr)
    for task, count, expected in incomplete:
        print(f"  {task}: {count}/{expected}", file=sys.stderr)
    raise SystemExit(1)
PY
}

trap cleanup_server EXIT
trap 'cleanup_server; exit 130' INT
trap 'cleanup_server; exit 143' TERM

if port_open; then
    echo "Port $REMOTE_HOST:$REMOTE_PORT is already open."
    echo "Stop the existing VALPA server before running the model sweep, so each cfg is evaluated against the intended hosted model."
    pkill -f "valpa/inference/serve_policy_ablation.py.*--port $REMOTE_PORT" || true
fi

mkdir -p "$SERVER_LOG_DIR"
echo "=== GPU visibility: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, DEVICE=$DEVICE ==="

FILTERED_MODEL_CONFIGS=()

for cfg_file in "${MODEL_CONFIGS[@]}"; do
    if ! model_name="$(cfg_model_name "/workspace/robolab/valpa/configs/inference/valpa-reach/ablation/$cfg_file")"; then
        echo "Could not find modelname in config file: $cfg_file"
        exit 1
    fi

    if [[ -z "$model_name" ]]; then
        echo "Empty modelname in config file: $cfg_file"
        exit 1
    fi

    archive_file="$OUTPUT_ROOT/${model_name}.zip"

    if [[ -f "$archive_file" ]]; then
        echo "=== Removing $cfg_file from MODEL_CONFIGS because archive already exists: $archive_file ==="
        continue
    fi

    FILTERED_MODEL_CONFIGS+=("$cfg_file")
done

MODEL_CONFIGS=("${FILTERED_MODEL_CONFIGS[@]}")
unset FILTERED_MODEL_CONFIGS

if ((${#MODEL_CONFIGS[@]} == 0)); then
    echo "All model configs already have archives in $OUTPUT_ROOT. Nothing to run."
    exit 0
fi

for cfg_file in "${MODEL_CONFIGS[@]}"; do
    cfg_path="/workspace/robolab/valpa/configs/inference/valpa-reach/ablation/$cfg_file"

    if ! cfg_model_name_value="$(cfg_model_name "$cfg_path")"; then
        echo "Could not find modelname in config file: $cfg_file"
        exit 1
    fi

    archive_file="$OUTPUT_ROOT/${cfg_model_name_value}.zip"

    if [[ -f "$archive_file" ]]; then
        echo
        echo "=== Skipping $cfg_file because archive now exists: $archive_file ==="
        continue
    fi

    cfg_name="${cfg_file%.yaml}"
    server_log="$SERVER_LOG_DIR/${cfg_name}_serve_policy_ablation.log"

    echo
    echo "=== Starting VALPA server: $cfg_file ==="
    
    # Ensure any stray serve_policy processes are killed
    pkill -f "valpa/inference/serve_policy_ablation.py.*--port $REMOTE_PORT" || true
    sleep 2
    
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "$ISAAC_PYTHON" valpa/inference/serve_policy_ablation.py \
        --cfg-file "valpa-reach/ablation/$cfg_file" \
        --host "$SERVER_HOST" \
        --port "$REMOTE_PORT" \
        >"$server_log" 2>&1 &
    SERVER_PID="$!"

    if ! wait_for_server; then
        echo "VALPA server failed to become ready for $cfg_file."
        echo "Last server log lines from $server_log:"
        tail -n 80 "$server_log" || true
        exit 1
    fi

    model_name="$(hosted_model_name)"

    if [[ "$model_name" != "$cfg_model_name_value" ]]; then
        echo "Config/server model name mismatch for $cfg_file."
        echo "Config log.modelname: $cfg_model_name_value"
        echo "Hosted modelname: $model_name"
        cleanup_server
        exit 1
    fi

    MODEL_NAMES+=("$model_name")

    echo "=== Running eval for hosted model $model_name (${NUM_RUNS_PER_TASK} runs/task) ==="
    if ! REMOTE_HOST="$REMOTE_HOST" \
        REMOTE_PORT="$REMOTE_PORT" \
        HEADLESS="$HEADLESS" \
        VIDEO_MODE="$VIDEO_MODE" \
        OUTPUT_FOLDER_NAME="$OUTPUT_FOLDER_NAME" \
        DEVICE="$DEVICE" \
        NUM_RUNS_PER_TASK="$NUM_RUNS_PER_TASK" \
        bash examples/policy/run_reach_eval_10x.sh; then
        echo "=== Eval failed for $model_name. Skipping archive to preserve partial outputs. ==="
        cleanup_server
        exit 1
    fi

    if ! check_model_output_complete "$model_name" "$NUM_RUNS_PER_TASK"; then
        echo "=== Output for $model_name is incomplete. Skipping archive to preserve partial outputs. ==="
        cleanup_server
        exit 1
    fi

    archive_model_output "$model_name"
    cleanup_server
done
