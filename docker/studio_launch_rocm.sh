#!/usr/bin/env bash
# Default CMD of the ROCm Studio image (Dockerfile.studio-rocm): Studio alone, no
# supervisord. UNSLOTH_STUDIO_PASSWORD only sets the initial admin password, and
# `unsloth studio` exits 1 when handed one after that, so a `docker restart` with
# the variable still set (a Compose file) would never come back up. As
# studio_launch.sh does: the value goes to a root-only file that unsloth-studio-run
# reads while nothing is stored, and never into Studio's environment.
set -euo pipefail

unsloth-studio-home

INITIAL_FILE="${UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE:-/run/unsloth/studio-initial-password}"
rm -f "$INITIAL_FILE"
if unsloth-studio-run --stored; then
    NOTE="password set on an earlier boot (inside the container: unsloth studio reset-password)"
elif [[ -n "${UNSLOTH_STUDIO_PASSWORD:-}" ]]; then
    mkdir -p "$(dirname "$INITIAL_FILE")"
    (umask 077 && printf '%s' "$UNSLOTH_STUDIO_PASSWORD" > "$INITIAL_FILE")
    NOTE="password from UNSLOTH_STUDIO_PASSWORD env"
else
    NOTE="password generated on first boot; Unsloth Studio names the file it wrote in its log below"
fi
unset UNSLOTH_STUDIO_PASSWORD
echo "Unsloth Studio -> http://localhost:${UNSLOTH_STUDIO_PORT:-8000}   (user unsloth, ${NOTE})"
exec unsloth-studio-run
