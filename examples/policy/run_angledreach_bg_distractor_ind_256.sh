#!/usr/bin/env bash
set -euo pipefail

export MODEL_CONFIG="droid-256px-8f-ind.yaml"
export REMOTE_PORT="${REMOTE_PORT:-8600}"
exec bash examples/policy/run_angledreach_bg_distractor_model.sh
