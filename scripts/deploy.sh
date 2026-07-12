#!/usr/bin/env bash
# One-click Linux deploy for preference platform (API + worker + SPA + crawler).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- defaults ----------
HOST_PORT="${HOST_PORT:-8080}"
PLATFORM_DATA_HOST="${PLATFORM_DATA_HOST:-$ROOT/deploy_data/platform_data}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-$ROOT/deploy_data/models}"
MEDIA_HOST_DIR="${MEDIA_HOST_DIR:-$ROOT/deploy_data/media}"
MODEL_FILE="${MODEL_FILE:-}"
WORKER_BATCH_SIZE="${WORKER_BATCH_SIZE:-8}"
WORKER_POLL_SECONDS="${WORKER_POLL_SECONDS:-5}"
CRAWLER_INTERVAL_SECONDS="${CRAWLER_INTERVAL_SECONDS:-21600}"
CRAWLER_START_PAGE="${CRAWLER_START_PAGE:-1}"
CRAWLER_MAX_PAGES="${CRAWLER_MAX_PAGES:-4}"
SKIP_BUILD="${SKIP_BUILD:-0}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-preference-platform}"

DO_BUILD=1
DO_UP=1
DO_DOWN=0
DO_LOGS=0
RECREATE=0

usage() {
  cat <<'EOF'
Usage: ./scripts/deploy.sh [options]

Options:
  --build-only     Build images only
  --down           Stop and remove containers (keeps host data dirs)
  --logs           Follow api+worker+crawler logs after start (or alone)
  --recreate       Force recreate containers
  --skip-build     Skip docker build (use existing image)
  -h, --help       Show this help

Environment:
  HOST_PORT              Host port (default 8080)
  PLATFORM_DATA_HOST     SQLite / snapshots / HF cache (default ./deploy_data/platform_data)
  MODEL_HOST_DIR         Model directory mounted at /data/models (default ./deploy_data/models)
  MEDIA_HOST_DIR         Media root mounted at /data/media (default ./deploy_data/media)
  MODEL_FILE             Optional host .pth to copy/link as best_model.pth
  WORKER_BATCH_SIZE      Default 8
  CRAWLER_INTERVAL_SECONDS  Crawl interval; 21600 = 6h, 0 = run once
  CRAWLER_START_PAGE     First list page (default 1)
  CRAWLER_MAX_PAGES      Pages per crawl (default 4)
  INGEST_API_KEY         If set when generating .env, used as-is
EOF
}

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) DO_UP=0; DO_BUILD=1; shift ;;
    --down) DO_DOWN=1; DO_UP=0; DO_BUILD=0; shift ;;
    --logs) DO_LOGS=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --skip-build) SKIP_BUILD=1; DO_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

# ---------- docker ----------
if ! command -v docker >/dev/null 2>&1; then
  die "docker not found. Install Docker Engine: https://docs.docker.com/engine/install/"
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  die "docker compose plugin not found. Install Docker Compose v2."
fi

export COMPOSE_PROJECT_NAME
export HOST_PORT PLATFORM_DATA_HOST MODEL_HOST_DIR MEDIA_HOST_DIR
export WORKER_BATCH_SIZE WORKER_POLL_SECONDS
export CRAWLER_INTERVAL_SECONDS CRAWLER_START_PAGE CRAWLER_MAX_PAGES

compose() { "${COMPOSE[@]}" "$@"; }

# ---------- port check ----------
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" 2>/dev/null | grep -q ":$port"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

# ---------- down ----------
if [[ "$DO_DOWN" -eq 1 ]]; then
  log "Stopping stack..."
  compose down --remove-orphans || true
  log "Stopped. Host data kept under: $PLATFORM_DATA_HOST"
  exit 0
fi

# ---------- logs only ----------
if [[ "$DO_LOGS" -eq 1 && "$DO_UP" -eq 0 && "$DO_BUILD" -eq 0 ]]; then
  compose logs -f api worker crawler
  exit 0
fi

# ---------- prepare dirs ----------
log "Preparing host directories..."
mkdir -p "$PLATFORM_DATA_HOST" "$MODEL_HOST_DIR" "$MEDIA_HOST_DIR" \
  "$PLATFORM_DATA_HOST/hf_cache" \
  "$PLATFORM_DATA_HOST/training_snapshots" \
  "$PLATFORM_DATA_HOST/candidates"

# Resolve absolute paths for compose volume mounts
PLATFORM_DATA_HOST="$(cd "$PLATFORM_DATA_HOST" && pwd)"
MODEL_HOST_DIR="$(cd "$MODEL_HOST_DIR" && pwd)"
MEDIA_HOST_DIR="$(cd "$MEDIA_HOST_DIR" && pwd)"
export PLATFORM_DATA_HOST MODEL_HOST_DIR MEDIA_HOST_DIR

# Optional model file
if [[ -n "$MODEL_FILE" ]]; then
  [[ -f "$MODEL_FILE" ]] || die "MODEL_FILE not found: $MODEL_FILE"
  target="$MODEL_HOST_DIR/best_model.pth"
  if [[ "$(cd "$(dirname "$MODEL_FILE")" && pwd)/$(basename "$MODEL_FILE")" != "$target" ]]; then
    log "Installing model -> $target"
    cp -f "$MODEL_FILE" "$target"
  fi
fi

if [[ ! -f "$MODEL_HOST_DIR/best_model.pth" ]]; then
  # Accept first .pth in model dir as default name
  first_pth="$(find "$MODEL_HOST_DIR" -maxdepth 1 -type f \( -name '*.pth' -o -name '*.pt' \) 2>/dev/null | head -n 1 || true)"
  if [[ -n "$first_pth" && ! -f "$MODEL_HOST_DIR/best_model.pth" ]]; then
    log "Linking $(basename "$first_pth") as best_model.pth"
    ln -sfn "$(basename "$first_pth")" "$MODEL_HOST_DIR/best_model.pth" 2>/dev/null \
      || cp -f "$first_pth" "$MODEL_HOST_DIR/best_model.pth"
  fi
fi

if [[ ! -f "$MODEL_HOST_DIR/best_model.pth" ]]; then
  warn "No model at $MODEL_HOST_DIR/best_model.pth — API will start but scoring needs a model."
  warn "Place a checkpoint there or re-run with MODEL_FILE=/path/to/model.pth"
fi

if [[ -z "$(find "$MEDIA_HOST_DIR" -type f 2>/dev/null | head -n 1)" ]]; then
  warn "Media directory is empty: $MEDIA_HOST_DIR"
  warn "Media directory is initially empty; the crawler will populate it after startup."
fi

# ---------- .env ----------
if [[ ! -f "$ROOT/.env" ]]; then
  log "Creating .env from .env.docker.example"
  [[ -f "$ROOT/.env.docker.example" ]] || die "missing .env.docker.example"
  cp "$ROOT/.env.docker.example" "$ROOT/.env"
  # Generate ingest key if still placeholder
  if grep -q 'replace-with-a-long-random-value' "$ROOT/.env" 2>/dev/null; then
    if [[ -n "${INGEST_API_KEY:-}" ]]; then
      key="$INGEST_API_KEY"
    elif command -v openssl >/dev/null 2>&1; then
      key="$(openssl rand -hex 24)"
    else
      key="$(tr -dc 'a-f0-9' </dev/urandom | head -c 48)"
    fi
    # portable sed
    if sed --version >/dev/null 2>&1; then
      sed -i "s/replace-with-a-long-random-value/$key/" "$ROOT/.env"
    else
      sed -i.bak "s/replace-with-a-long-random-value/$key/" "$ROOT/.env" && rm -f "$ROOT/.env.bak"
    fi
    log "Generated INGEST_API_KEY (also stored in .env)"
  fi
else
  log "Using existing .env"
fi

# Ensure container paths in .env (do not overwrite custom if already set correctly)
grep -q '^PLATFORM_DATA_DIR=' "$ROOT/.env" || echo 'PLATFORM_DATA_DIR=/data/platform_data' >>"$ROOT/.env"
grep -q '^ALLOWED_MEDIA_ROOTS=' "$ROOT/.env" || echo 'ALLOWED_MEDIA_ROOTS=/data/media' >>"$ROOT/.env"
grep -q '^DEFAULT_MODEL_PATH=' "$ROOT/.env" || echo 'DEFAULT_MODEL_PATH=/data/models/best_model.pth' >>"$ROOT/.env"

# Sync HOST_PORT into env file for human reference
if grep -q '^HOST_PORT=' "$ROOT/.env"; then
  if sed --version >/dev/null 2>&1; then
    sed -i "s/^HOST_PORT=.*/HOST_PORT=$HOST_PORT/" "$ROOT/.env"
  else
    sed -i.bak "s/^HOST_PORT=.*/HOST_PORT=$HOST_PORT/" "$ROOT/.env" && rm -f "$ROOT/.env.bak"
  fi
else
  echo "HOST_PORT=$HOST_PORT" >>"$ROOT/.env"
fi

# ---------- port ----------
if [[ "$DO_UP" -eq 1 ]]; then
  if port_in_use "$HOST_PORT"; then
    # Allow if our container already binds it
    if ! docker ps --format '{{.Names}} {{.Ports}}' | grep -q "preference-api.*$HOST_PORT"; then
      die "Port $HOST_PORT is already in use. Set HOST_PORT=xxxx and retry."
    fi
  fi
fi

# ---------- build ----------
if [[ "$SKIP_BUILD" -eq 1 ]]; then
  log "Skipping build (SKIP_BUILD=1)"
elif [[ "$DO_BUILD" -eq 1 ]]; then
  log "Building image (first run downloads PyTorch/deps — may take a long time)..."
  compose build
fi

# ---------- up ----------
if [[ "$DO_UP" -eq 1 ]]; then
  log "Starting api + worker + crawler..."
  if [[ "$RECREATE" -eq 1 ]]; then
    compose up -d --force-recreate --remove-orphans
  else
    compose up -d --remove-orphans
  fi

  log "Waiting for API health..."
  ok=0
  for i in $(seq 1 60); do
    # The container healthcheck verifies the same endpoint from inside the
    # API network namespace. This avoids false failures on hosts whose
    # loopback/port-forwarding policy prevents the deploy user from curling
    # the published port.
    if [[ "$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' preference-api 2>/dev/null || true)" == "healthy" ]]; then
      ok=1
      break
    fi
    sleep 2
  done
  if [[ "$ok" -ne 1 ]]; then
    warn "Health check timed out. Recent logs:"
    compose logs --tail=80 api worker crawler || true
    die "API not ready on port $HOST_PORT"
  fi

  # Detect LAN IP for display
  lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [[ -z "$lan_ip" ]] && lan_ip="<server-ip>"

  cat <<EOF

========================================
  Preference platform is up
========================================
  Local:    http://127.0.0.1:${HOST_PORT}/
  LAN:      http://${lan_ip}:${HOST_PORT}/

  First visit: open the URL → create admin (password ≥ 12 chars)

  Data:     ${PLATFORM_DATA_HOST}
  Models:   ${MODEL_HOST_DIR}
  Media:    ${MEDIA_HOST_DIR}

  Logs:     ./scripts/deploy.sh --logs
  Stop:     ./scripts/deploy.sh --down
  Requeue:  docker compose exec worker python -m platform_app.requeue_predictions

  Crawler:  runs automatically every ${CRAWLER_INTERVAL_SECONDS}s
            pages ${CRAWLER_START_PAGE}..$((CRAWLER_START_PAGE + CRAWLER_MAX_PAGES - 1))
            docker compose logs -f crawler

  Docs:     DEPLOY.md
========================================
EOF
fi

if [[ "$DO_LOGS" -eq 1 ]]; then
  compose logs -f api worker crawler
fi
