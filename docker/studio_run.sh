#!/usr/bin/env bash
# supervisord's studio program. Applies the initial admin password only while none
# is stored: `unsloth studio` exits 1 when handed one afterwards, so a restart of the
# program (crash, unsloth-studio-update, docker restart) would park Studio in FATAL.
# The launcher leaves the value in a root-only file, never in supervisord's
# environment, and this decides at every spawn.
#
#   unsloth-studio-run             start Studio
#   unsloth-studio-run --stored    exit 0 when an admin password is stored
set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
INITIAL="${UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE:-/run/unsloth/studio-initial-password}"

password_stored() {
    # The admin row with must_change_password=0. A bare auth.db from an interrupted
    # first launch, or a seeded row nobody changed, still accepts an initial password.
    python3 - "${STUDIO_HOME}/auth/auth.db" <<'PY'
import sqlite3, sys
try:
    conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri = True)
    row = conn.execute(
        "SELECT must_change_password FROM auth_user WHERE username = 'unsloth'"
    ).fetchone()
except sqlite3.Error:
    row = None
sys.exit(0 if row is not None and not row[0] else 1)
PY
}

if [[ "${1:-}" == "--stored" ]]; then
    if password_stored; then exit 0; else exit 1; fi
fi

unset UNSLOTH_STUDIO_PASSWORD
if [[ -s "$INITIAL" ]] && ! password_stored; then
    UNSLOTH_STUDIO_PASSWORD="$(<"$INITIAL")"
    export UNSLOTH_STUDIO_PASSWORD
fi
exec "${STUDIO_HOME}/bin/unsloth" studio -H 0.0.0.0 -p 8000
