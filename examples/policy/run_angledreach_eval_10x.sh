#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8002}"
HEADLESS="${HEADLESS:-1}"
VIDEO_MODE="${VIDEO_MODE:-sensor}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-}"
DEVICE="${DEVICE:-cuda:0}"
NUM_RUNS_PER_TASK="${NUM_RUNS_PER_TASK:-10}"

EXTRA_ARGS=()
if [[ "$HEADLESS" == "1" ]]; then
    EXTRA_ARGS+=(--headless)
fi
if [[ -n "$VIDEO_MODE" ]]; then
    EXTRA_ARGS+=(--video-mode "$VIDEO_MODE")
fi

TASKS=(
    AngledReachBananaTask
    AngledReachMarkerTask
    AngledReachDrillTask
    AngledReachCartoonTask
    AngledReachKetchupTask
    AngledReachCartoon2Task
)

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

MODEL_NAME="$(hosted_model_name)"
if [[ -z "$OUTPUT_FOLDER_NAME" ]]; then
    OUTPUT_FOLDER_NAME="${MODEL_NAME}_angledreach"
fi
EXTRA_ARGS+=(--output-folder-name "$OUTPUT_FOLDER_NAME")

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "=== Eval GPU visibility: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, DEVICE=$DEVICE ==="
else
    echo "=== Eval device: DEVICE=$DEVICE (CUDA_VISIBLE_DEVICES unset) ==="
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" examples/policy/run_angledreach_eval.py \
    --policy valpa \
    --num-runs $NUM_RUNS_PER_TASK \
    --num-envs 1 \
    --device "$DEVICE" \
    --task-dirs wm_tasks/angledreach \
    --task "${TASKS[@]}" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    "${EXTRA_ARGS[@]}"
