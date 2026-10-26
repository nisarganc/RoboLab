#!/usr/bin/env bash
set -euo pipefail

export MODEL_CONFIG="droid-224px-8f-dual.yaml"
export REMOTE_PORT="${REMOTE_PORT:-8500}"
exec bash examples/policy/run_angledreach_bg_distractor_model.sh
