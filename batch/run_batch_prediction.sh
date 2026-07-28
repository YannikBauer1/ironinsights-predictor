#!/bin/bash
# Run batch prediction for upcoming competitions (Megan worker / cron).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

python3 batch/batch_predict_and_update.py --since-last-run "$@"
