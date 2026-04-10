#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator (template)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
# Or locally:
#   chmod +x scripts/validate-submission.sh
#   ./scripts/validate-submission.sh <ping_url> [repo_dir]

set -uo pipefail

PING_URL=${1:-}
REPO_DIR=${2:-$(pwd)}

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

if ! command -v openenv >/dev/null 2>&1; then
  echo "openenv-core is not installed. Run: pip install openenv-core"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker first."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required."
  exit 1
fi

echo "[1/3] Pinging HF Space: $PING_URL/reset"
if ! curl -fsS "$PING_URL/reset" >/dev/null; then
  echo "HF Space ping failed. Ensure it returns HTTP 200 for /reset."
  exit 1
fi

echo "[2/3] openenv validate"
( cd "$REPO_DIR" && openenv validate )

echo "[3/3] Docker build"
( cd "$REPO_DIR" && docker build -t medtriage-openenv . )

echo "Validation complete."
