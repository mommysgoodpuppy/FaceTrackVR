#!/usr/bin/env bash
# FaceTrackVR user-local installer. Run from inside the extracted tarball:
#   ./install.sh          install/upgrade to ~/.local/opt/FaceTrackVR
#   ./install.sh --remove uninstall
set -euo pipefail

APP_DIR="${HOME}/.local/opt/FaceTrackVR"
BIN_LINK="${HOME}/.local/bin/facetrackvr"
DESKTOP_FILE="${HOME}/.local/share/applications/FaceTrackVR.desktop"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--remove" ]; then
    rm -rf "$APP_DIR"
    rm -f "$BIN_LINK" "$DESKTOP_FILE"
    echo "FaceTrackVR removed. (Compatible EyeTrackVR settings were kept.)"
    exit 0
fi

if [ ! -x "$HERE/facetrackvr" ]; then
    echo "ERROR: run this script from inside the extracted FaceTrackVR folder." >&2
    exit 1
fi

echo "Installing to $APP_DIR ..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR" "$(dirname "$BIN_LINK")" "$(dirname "$DESKTOP_FILE")"
cp -a "$HERE/." "$APP_DIR/"

ln -sf "$APP_DIR/facetrackvr" "$BIN_LINK"

sed "s|INSTALL_DIR|$APP_DIR|g" "$HERE/FaceTrackVR.desktop" > "$DESKTOP_FILE"
chmod +x "$DESKTOP_FILE" || true
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$(dirname "$DESKTOP_FILE")" || true

echo "Done. Launch 'FaceTrackVR' from your app menu, or run: $BIN_LINK"
echo "(Make sure ~/.local/bin is on your PATH to use the terminal command.)"
