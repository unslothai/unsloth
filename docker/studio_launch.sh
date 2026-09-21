#!/usr/bin/env bash
# Default CMD of the full Unsloth image (Dockerfile.studio).
#
# Bootstraps the three services managed by supervisord:
#   studio   port UNSLOTH_STUDIO_PORT (8000)   user unsloth; password from UNSLOTH_STUDIO_PASSWORD, or
#                        the generated one printed in `docker logs` (studio-password)
#   jupyter  port 8888   password from JUPYTER_PASSWORD, or a random one
#                        printed in `docker logs` when unset
#   sshd     port 22     key-only; enabled when PUBLIC_KEY / SSH_KEY is set
#
# Environment:
#   JUPYTER_PORT       Jupyter port inside the container       (default 8888)
#   JUPYTER_PASSWORD   Jupyter login password (unset: generated and printed)
#   UNSLOTH_STUDIO_PASSWORD  initial Studio admin password (unset: generated and
#                      printed); ignored once a password is stored
#   PUBLIC_KEY/SSH_KEY OpenSSH public key for root login; sshd stays disabled
#                      when neither is set (nothing to authenticate with --
#                      password login is never enabled for root)
#   UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY=1  exit 0 right after the settings checks, before
#                      anything is written (tests only)
set -euo pipefail

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export UNSLOTH_STUDIO_PORT="${UNSLOTH_STUDIO_PORT:-8000}"
# at most five significant digits, or bash arithmetic wraps 18446744073709551617 to 1
if ! [[ "$UNSLOTH_STUDIO_PORT" =~ ^0*[0-9]{1,5}$ ]] || (( 10#$UNSLOTH_STUDIO_PORT < 1 || 10#$UNSLOTH_STUDIO_PORT > 65535 )); then
    printf "\033[1;31mERROR:\033[0m UNSLOTH_STUDIO_PORT=%s is not a port number (1-65535).\n" "$UNSLOTH_STUDIO_PORT" >&2
    exit 1
fi
UNSLOTH_STUDIO_PORT=$(( 10#$UNSLOTH_STUDIO_PORT ))
if [[ -n "${SSH_KEY:-${PUBLIC_KEY:-}}" ]] && (( UNSLOTH_STUDIO_PORT == 22 )); then
    printf "\033[1;31mERROR:\033[0m UNSLOTH_STUDIO_PORT=22 is sshd's port inside the container when SSH_KEY or PUBLIC_KEY is set.\n" >&2
    exit 1
fi
# JupyterLab starts first and wins the bind, so Studio silently falls back to an unpublished
# port. Compared as traitlets does, via int(): whitespace, leading zeros, a leading + and
# underscores ("08000", " 8000", "8_000") all bind 8000.
jupyter_port_digits="${JUPYTER_PORT//[[:space:]_]/}"
jupyter_port_digits="${jupyter_port_digits#+}"
if [[ "$jupyter_port_digits" =~ ^[0-9]+$ ]] && (( 10#$jupyter_port_digits == UNSLOTH_STUDIO_PORT )); then
    printf "\033[1;31mERROR:\033[0m JUPYTER_PORT=%s is Unsloth Studio's port inside the container (UNSLOTH_STUDIO_PORT).\n" "$UNSLOTH_STUDIO_PORT" >&2
    printf "       Leave JupyterLab on 8888 and map the host side instead: -p 9000:8888\n" >&2
    exit 1
fi
export UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S="${UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S:-120}"
if ! [[ "$UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S" =~ ^[0-9]+$ ]]; then
    printf "\033[1;31mERROR:\033[0m UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S=%s is not a number of seconds.\n" "$UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S" >&2
    exit 1
fi
export UNSLOTH_STUDIO_STOP_WAIT_S=$(( 10#$UNSLOTH_STUDIO_SHUTDOWN_STOP_TIMEOUT_S + 30 ))
# tests run this on the host; everything past here writes /etc/profile.d, /root/.jupyter, /workspace
[[ "${UNSLOTH_STUDIO_LAUNCH_CHECK_ONLY:-0}" == 1 ]] && exit 0
export UNSLOTH_STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
export UNSLOTH_JUPYTER_CLOUDFLARE="${UNSLOTH_JUPYTER_CLOUDFLARE:-0}"

# SSH login shells lack the `docker run -e` vars. Secrets are excluded on purpose,
# and every value is shlex.quote()d because this file is sourced by every login shell.
python - > /etc/profile.d/unsloth_env.sh <<'PY' || true
import os, re, shlex
keep   = re.compile(r"^(HF_|CUDA_|NCCL_|JUPYTER_|UNSLOTH_|WANDB_|TRITON_)|^PATH$")
secret = re.compile(r"(_TOKEN|_API_KEY|_PASSWORD|_SECRET|_LICENSE)$")
for key, value in sorted(os.environ.items()):
    if keep.search(key) and not secret.search(key):
        print(f"export {key}={shlex.quote(value)}")
PY

# never store plaintext, and never a fixed default: unset means random, printed once
JUPYTER_CONFIG_DIR=/root/.jupyter
JUPYTER_NOTE="password from JUPYTER_PASSWORD env"
if [[ -f "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" ]]; then
    JUPYTER_NOTE="existing jupyter config reused"
else
    if [[ -z "${JUPYTER_PASSWORD:-}" ]]; then
        JUPYTER_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(12))')"
        JUPYTER_NOTE="generated password: ${JUPYTER_PASSWORD}"
    fi
    export JUPYTER_PASSWORD
    mkdir -p "${JUPYTER_CONFIG_DIR}"
    HASH=$(python - <<PY
from jupyter_server.auth import passwd
import os
print(passwd(os.environ["JUPYTER_PASSWORD"]))
PY
)
    cat > "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" <<EOF
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/workspace"
c.PasswordIdentityProvider.hashed_password = "${HASH}"
EOF
    # mirror unsloth_sync_notebooks.sh's gating, or the view points at a missing dir
    _root_dir="/workspace"
    _view_dir="${UNSLOTH_NOTEBOOKS_VIEW_DIR:-/workspace/Unsloth Notebooks}"
    if [[ "${UNSLOTH_SKIP_NOTEBOOK_VIEW:-0}" != "1" \
          && "${UNSLOTH_SKIP_NOTEBOOK_SYNC:-0}" != "1" \
          && "${_view_dir}" == "${_root_dir}/"* ]]; then
        _view_rel="${_view_dir#${_root_dir}/}"
        # default_url must be set on BOTH ServerApp and LabApp, or the lab app
        # overrides ServerApp back to /lab.
        #
        # repr() rather than interpolation into a heredoc: a double quote in the
        # path closed the string literal and made jupyter_lab_config.py a
        # SyntaxError, so the documented override stopped the service starting,
        # and a backslash silently changed the path. Both are legal POSIX
        # characters. Values arrive via the environment, like the password block
        # above, so the shell side needs no quoting either.
        UNSLOTH_VIEW_REL="${_view_rel}" UNSLOTH_VIEW_DIR="${_view_dir}" \
        python - >> "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" <<'PY'
import os
import urllib.parse
rel  = os.environ["UNSLOTH_VIEW_REL"]
view = os.environ["UNSLOTH_VIEW_DIR"]
# default_url is a URL, not a path: the default view directory has a space in it,
# and unencoded it lands in the banner Jupyter prints as
# "http://host:8888/lab/tree/Unsloth Notebooks", which is not copy-pasteable and
# is not a legal request target. Measured: curl refuses the raw form outright and
# gets 302 from the encoded one. quote() leaves "/" alone, so subdirectories keep working.
url = "/lab/tree/" + urllib.parse.quote(rel)
print(f"c.ServerApp.default_url = {url!r}")
print(f"c.LabApp.default_url = {url!r}")
print(f"c.ServerApp.preferred_dir = {view!r}")
PY
    fi
fi

PUBLIC_SSH_KEY="${SSH_KEY:-${PUBLIC_KEY:-}}"
export UNSLOTH_ENABLE_SSHD=false
if [[ -n "${PUBLIC_SSH_KEY}" ]] && command -v sshd >/dev/null 2>&1; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "${PUBLIC_SSH_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A
    mkdir -p /run/sshd
    export UNSLOTH_ENABLE_SSHD=true
fi

mkdir -p /workspace

# AGPLv3: refuse to start if the Unsloth attribution is stripped. The same checker
# runs at build time and as a jupyter_server extension.
if [[ "${UNSLOTH_SKIP_BRANDING_CHECK:-0}" != "1" ]]; then
    if ! /opt/unsloth-venv/bin/python -m unsloth_branding --verify; then
        echo "Refusing to start the container." >&2
        exit 1
    fi
fi

export UNSLOTH_JUPYTER_NOTE="${JUPYTER_NOTE}"  # for the ready summary (studio-password)
# UNSLOTH_STUDIO_PASSWORD only sets the initial admin password, and `unsloth studio`
# exits 1 when handed one after that. So it goes to a root-only file that
# unsloth-studio-run consumes while nothing is stored, and never into supervisord's
# environment, where every respawn of the studio program would see it again.
INITIAL_FILE="${UNSLOTH_STUDIO_INITIAL_PASSWORD_FILE:-/run/unsloth/studio-initial-password}"
rm -f "$INITIAL_FILE"
if unsloth-studio-run --stored; then
    STUDIO_NOTE="user unsloth, password set on an earlier boot"
    UNSLOTH_STUDIO_PASSWORD_STATE=stored
elif [[ -n "${UNSLOTH_STUDIO_PASSWORD:-}" ]]; then
    mkdir -p "$(dirname "$INITIAL_FILE")"
    (umask 077 && printf '%s' "$UNSLOTH_STUDIO_PASSWORD" > "$INITIAL_FILE")
    STUDIO_NOTE="user unsloth, password from UNSLOTH_STUDIO_PASSWORD env"
    UNSLOTH_STUDIO_PASSWORD_STATE=initial
else
    STUDIO_NOTE="user unsloth, generated password printed below once Unsloth Studio is up"
    UNSLOTH_STUDIO_PASSWORD_STATE=generated
fi
unset UNSLOTH_STUDIO_PASSWORD
export UNSLOTH_STUDIO_PASSWORD_STATE  # read by unsloth-studio-password
echo "Unsloth Studio  -> http://localhost:${UNSLOTH_STUDIO_PORT}   (${STUDIO_NOTE})"
echo "JupyterLab      -> http://localhost:${JUPYTER_PORT}   (${JUPYTER_NOTE})"
if [[ "${UNSLOTH_JUPYTER_CLOUDFLARE}" == "1" ]]; then
    echo "JupyterLab tunnel-> enabled; public trycloudflare URL appears below once it is up"
else
    echo "JupyterLab tunnel-> off (set UNSLOTH_JUPYTER_CLOUDFLARE=1 for a public link)"
fi
if [[ "${UNSLOTH_ENABLE_SSHD}" == "true" ]]; then
    echo "sshd            -> port 22 (key-only)"
fi

exec supervisord -c /etc/supervisor/supervisord.conf
