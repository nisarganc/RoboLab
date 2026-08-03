#!/usr/bin/env bash
set -euo pipefail

ISAAC_PYTHON="${ISAAC_PYTHON:-python-rtx-compat}"
DEVICE="${DEVICE:-cuda:0}"
NUM_BACKGROUNDS="${NUM_BACKGROUNDS:-5}"
BACKGROUND_SEED="${BACKGROUND_SEED:-1}"
HEADLESS="${HEADLESS:-1}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_ARGS=()
if [[ "$HEADLESS" == "1" ]]; then
    EXTRA_ARGS+=(--headless)
fi
if [[ "$OVERWRITE" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "$ISAAC_PYTHON" examples/demo/generate_bg_distractor_goal_images.py \
    --task-dirs wm_tasks/bg_distractor \
    --num-backgrounds "$NUM_BACKGROUNDS" \
    --background-seed "$BACKGROUND_SEED" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"
