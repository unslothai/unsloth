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
    # The admin row with must_change_password=0. A bare auth.db from an interrupted
    # first launch, or a seeded row nobody changed, still accepts an initial password.
    # A row from before that column existed counts as stored: the CLI migrates it
    # with default 0 and then rejects an initial password.
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
if [[ -s "$INITIAL" ]] && ! password_stored; then
    # byte for byte: $(<file) would drop a trailing newline the CLI is meant to see
    IFS= read -r -d '' UNSLOTH_STUDIO_PASSWORD < "$INITIAL" || true
    export UNSLOTH_STUDIO_PASSWORD
fi
# Exposure. The default stays -H 0.0.0.0 because that is what makes a published
# -p 8000:8000 reachable at all; the container is the boundary, the host's -p
# decides who sees it.
#
#   UNSLOTH_STUDIO_SECURE=1      --secure: a Cloudflare HTTPS link and nothing
#                                else. Studio forces a loopback bind itself, so
#                                -p 8000:8000 would publish a port nothing is
#                                listening on. Fails closed if the tunnel does
#                                not come up.
#   UNSLOTH_STUDIO_CLOUDFLARE=1  --cloudflare: the same public HTTPS link, with
#                                the local port still served, for a laptop that
#                                wants both.
#
# Mirrors UNSLOTH_JUPYTER_CLOUDFLARE, which JupyterLab has had all along. The two
# are mutually exclusive in the CLI, so refuse the pair here rather than let
# supervisord restart Studio forever on an argument error.
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
