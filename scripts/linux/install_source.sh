#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
BIN_LINK="${HOME}/.local/bin/facetrackvr"
DESKTOP_FILE="${HOME}/.local/share/applications/FaceTrackVR.desktop"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required for the source installation." >&2
    exit 1
fi

uv venv --python 3.14 --clear "$VENV_DIR"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    -r "$REPO_ROOT/scripts/requirements-linux-dev.txt"

mkdir -p "$(dirname "$BIN_LINK")" "$(dirname "$DESKTOP_FILE")"
ln -sf "$REPO_ROOT/scripts/linux/run_source.sh" "$BIN_LINK"
sed "s|REPO_DIR|$REPO_ROOT|g" \
    "$REPO_ROOT/scripts/linux/FaceTrackVR-source.desktop" > "$DESKTOP_FILE"
chmod +x "$REPO_ROOT/scripts/linux/run_source.sh" "$DESKTOP_FILE"
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$(dirname "$DESKTOP_FILE")" || true

echo "FaceTrackVR now launches from $REPO_ROOT"
