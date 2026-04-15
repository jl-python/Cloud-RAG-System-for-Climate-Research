#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Single-command runner for the Climate RAG system
# Cloud Computing Course Project
#
# Usage:
#   bash reproduce.sh
#
# Note: Ingestion is NOT run here — it takes ~2 hours.
#   Run manually: python3 data/ingestion.py --n 3000
#   Optional: set HF_TOKEN in .env for HuggingFace streaming (see .env.example).
# =============================================================================

set -e
set -m

# ── Colors ────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[reproduce.sh]${NC} $1"; }
warn() { echo -e "${YELLOW}[reproduce.sh]${NC} $1"; }
fail() { echo -e "${RED}[reproduce.sh] ERROR:${NC} $1"; exit 1; }

# ── Step 0: Check Python 3.12+ ───────────────────────────────
log "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 12 ]; then
  fail "Python 3.12+ required. You have Python $PYTHON_VERSION."
fi
log "Python $PYTHON_VERSION (ok)"

# ── Step 1: Check .env ───────────────────────────────────────
log "Checking .env file..."
if [ ! -f ".env" ]; then
  fail ".env not found. Run: cp .env.example .env and fill in your credentials."
fi

source .env
for var in DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD GEMINI_API_KEY BACKEND_URL; do
  if [ -z "${!var}" ]; then
    fail "Missing required .env variable: $var"
  fi
done
log ".env variables verified (ok)"

# ── Step 2: Virtual environment ───────────────────────────────
log "Setting up virtual environment..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  log "Created venv (ok)"
else
  warn "venv already exists, skipping creation."
fi

source venv/bin/activate
log "Virtual environment activated (ok)"

# ── Step 3: Install dependencies ─────────────────────────────
log "Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt
log "Dependencies installed (ok)"

# ── Step 4: Create output directories ────────────────────────
log "Creating output directories..."
mkdir -p artifacts logs tests data/checkpoints
log "Directories ready (ok)"

# ── Step 5: Test DB connection ────────────────────────────────
log "Testing database connection..."
python3 scripts/db_connect.py || fail "Database connection failed. Check your .env credentials."
log "Database connected (ok)"

# ── Step 6: Start backend ─────────────────────────────────────
log "Starting FastAPI backend on port 3001..."
if command -v lsof >/dev/null 2>&1 && lsof -i :3001 -sTCP:LISTEN -t >/dev/null 2>&1; then
  warn "Port 3001 already in use — killing existing process..."
  kill $(lsof -i :3001 -sTCP:LISTEN -t) 2>/dev/null || true
  sleep 1
fi

uvicorn backend.app:app --port 3001 >> logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > logs/backend.pid

# Wait for backend to be ready (model load can take 60s+ on first boot)
log "Waiting for backend to be ready (this may take a minute on first run)..."
for i in {1..45}; do
  if curl -s http://localhost:3001/health | grep -q "ok"; then
    log "Backend is up (ok)"
    break
  fi
  if [ $i -eq 45 ]; then
    kill -- -$BACKEND_PID 2>/dev/null || kill $BACKEND_PID 2>/dev/null
    fail "Backend did not start in time. Check logs/backend.log for errors."
  fi
  sleep 2
done

# ── Step 7: Run smoke tests ───────────────────────────────────
log "Running smoke tests..."
pytest tests/smoke_test.py -v 2>&1 | tee logs/smoke_test.log
log "Smoke tests passed (ok)"

# ── Step 8: Save artifacts ────────────────────────────────────
log "Saving run artifacts..."
pip freeze > artifacts/requirements_frozen.txt
echo "{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"python_version\": \"$PYTHON_VERSION\",
  \"backend_pid\": $BACKEND_PID
}" > artifacts/run_summary.json
log "Artifacts saved (ok)"

# ── Step 9: Start frontend ────────────────────────────────────
log "Starting Streamlit frontend on port 3000..."
streamlit run frontend/app.py --server.port 3000 >> logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > logs/frontend.pid

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Everything is running.${NC}"
echo -e "${GREEN} Frontend → http://localhost:3000${NC}"
echo -e "${GREEN} Backend  → http://localhost:3001${NC}"
echo -e "${GREEN} Logs     → logs/backend.log${NC}"
echo -e "${GREEN} Press Ctrl+C to stop both servers${NC}"
echo -e "${GREEN}============================================${NC}"

# ── Cleanup on exit ───────────────────────────────────────────
cleanup() {
  log "Shutting down servers..."
  [ -n "$BACKEND_PID" ]  && kill -- -$BACKEND_PID  2>/dev/null || true
  [ -n "$FRONTEND_PID" ] && kill -- -$FRONTEND_PID 2>/dev/null || true
  log "Done."
}
trap cleanup EXIT

wait $FRONTEND_PID