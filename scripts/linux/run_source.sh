#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "FaceTrackVR development environment is missing." >&2
    echo "Run: bash $REPO_ROOT/scripts/linux/install_source.sh" >&2
    exit 1
fi

cd "$REPO_ROOT/EyeTrackApp"
exec "$PYTHON" eyetrackapp.py "$@"
