#!/usr/bin/env bash
# Stop compose stack; keep host deploy_data volumes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/deploy.sh" --down
