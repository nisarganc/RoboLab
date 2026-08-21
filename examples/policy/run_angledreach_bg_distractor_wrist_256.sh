#!/usr/bin/env bash
set -euo pipefail

export MODEL_CONFIG="droid-256px-8f-wrist.yaml"
export REMOTE_PORT="${REMOTE_PORT:-6000}"
exec bash examples/policy/run_angledreach_bg_distractor_model.sh
