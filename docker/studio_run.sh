#!/usr/bin/env bash
# supervisord's studio program. Applies the initial admin password only while none
# is stored: `unsloth studio` exits 1 when handed one afterwards, so a restart of the
# program (crash, unsloth-studio-update, docker restart) would park Studio in FATAL.
# The launcher leaves the value in a root-only file, never in supervisord's
# environment, and this decides at every spawn.
#
#   unsloth-studio-run                start Studio
#   unsloth-studio-run --stored       exit 0 when an admin password is stored
#   unsloth-studio-run --initialized  exit 0 when the admin row is committed
set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
INITIAL="${UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE:-/run/unsloth/studio-initial-password}"

admin_initialized() {
    # The bootstrap file is written before the admin row commits, so an interrupted
    # first launch can leave a file whose password the next launch replaces.
    python3 - "${STUDIO_HOME}/auth/auth.db" <<'PY'
import sqlite3, sys
from urllib.parse import quote
try:
    # quoted: a ? or # in the path would otherwise end the URI early and point
    # both checks at a database that does not exist; read-only so a missing
    # auth.db is never created here
    conn = sqlite3.connect(f"file:{quote(sys.argv[1])}?mode=ro", uri = True)
    row = conn.execute("SELECT 1 FROM auth_user WHERE username = 'unsloth'").fetchone()
except sqlite3.Error:
    row = None
sys.exit(0 if row is not None else 1)
PY
}

password_stored() {
    # Stored = admin row with must_change_password=0; a row predating that column counts as
    # stored (the CLI migrates it with default 0).
    python3 - "${STUDIO_HOME}/auth/auth.db" <<'PY'
import sqlite3, sys
from urllib.parse import quote
try:
    # quoted: a ? or # in the path would otherwise end the URI early and point
    # both checks at a database that does not exist; read-only so a missing
    # auth.db is never created here
    conn = sqlite3.connect(f"file:{quote(sys.argv[1])}?mode=ro", uri = True)
    row = conn.execute("SELECT 1 FROM auth_user WHERE username = 'unsloth'").fetchone()
    if row is not None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(auth_user)")}
        if "must_change_password" in cols:
            row = conn.execute(
                "SELECT must_change_password FROM auth_user WHERE username = 'unsloth'"
            ).fetchone()
            row = None if row is None or row[0] else row
except sqlite3.Error:
    row = None
sys.exit(0 if row is not None else 1)
PY
}

case "${1:-}" in
    --stored)      if password_stored;   then exit 0; else exit 1; fi ;;
    --initialized) if admin_initialized; then exit 0; else exit 1; fi ;;
esac

unset UNSLOTH_STUDIO_PASSWORD
# Managed accounts' tools run as uid 0 and can read /run; delete the plaintext once stored.
if [[ -e "$INITIAL" ]] && password_stored; then
    rm -f "$INITIAL"
fi
if [[ -s "$INITIAL" ]] && ! password_stored; then
    # byte for byte: $(<file) would drop a trailing newline the CLI is meant to see
    IFS= read -r -d '' UNSLOTH_STUDIO_PASSWORD < "$INITIAL" || true
    export UNSLOTH_STUDIO_PASSWORD
    # The CLI stores it before binding. $$ becomes the CLI at the exec below: once it is
    # gone without storing (bad password, crash) the file stays for the respawn.
    _wait="${UNSLOTH_STUDIO_PASSWORD_WAIT:-600}"
    [[ "$_wait" =~ ^[0-9]+$ ]] || _wait=600
    (
        for (( _i = 0; _i < 10#$_wait; _i++ )); do
            if password_stored; then rm -f "$INITIAL"; exit 0; fi
            kill -0 "$$" 2>/dev/null || { password_stored && rm -f "$INITIAL"; exit 0; }
            sleep 1
        done
    ) < /dev/null > /dev/null 2>&1 &
fi
# Default -H 0.0.0.0 so a published -p is reachable; the container is the boundary.
# UNSLOTH_STUDIO_SECURE=1: --secure (Cloudflare only, loopback bind, fails closed).
# UNSLOTH_STUDIO_CLOUDFLARE=1: --cloudflare (link plus local port). Mutually exclusive;
# refused here so supervisord does not restart Studio forever.
STUDIO_ARGS=(-H 0.0.0.0 -p "${UNSLOTH_STUDIO_PORT:-8000}")
if [[ "${UNSLOTH_STUDIO_SECURE:-0}" == "1" && "${UNSLOTH_STUDIO_CLOUDFLARE:-0}" == "1" ]]; then
    echo "ERROR: set UNSLOTH_STUDIO_SECURE=1 or UNSLOTH_STUDIO_CLOUDFLARE=1, not both:" >&2
    echo "       --secure serves only the tunnel, --cloudflare serves the tunnel and the local port." >&2
    exit 2
elif [[ "${UNSLOTH_STUDIO_SECURE:-0}" == "1" ]]; then
    STUDIO_ARGS=(--secure -p "${UNSLOTH_STUDIO_PORT:-8000}")
elif [[ "${UNSLOTH_STUDIO_CLOUDFLARE:-0}" == "1" ]]; then
    STUDIO_ARGS+=(--cloudflare)
fi
exec "${STUDIO_HOME}/bin/unsloth" studio "${STUDIO_ARGS[@]}"
