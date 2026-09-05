#!/usr/bin/env bash
# Default CMD of the full Unsloth image (Dockerfile.studio).
#
# Bootstraps the three services managed by supervisord:
#   studio   port 8000   user unsloth; password from UNSLOTH_STUDIO_PASSWORD, or
#                        the generated one printed in `docker logs` (studio-password)
#   jupyter  port 8888   password from JUPYTER_PASSWORD, or a random one
#                        printed in `docker logs` when unset
#   sshd     port 22     key-only; enabled when PUBLIC_KEY / SSH_KEY is set
#
# Environment:
#   JUPYTER_PORT       Jupyter port inside the container       (default 8888)
#   JUPYTER_PASSWORD   Jupyter login password (unset: generated and printed)
#   UNSLOTH_STUDIO_PASSWORD  Studio admin password (unset: generated and printed)
#   PUBLIC_KEY/SSH_KEY OpenSSH public key for root login; sshd stays disabled
#                      when neither is set (nothing to authenticate with --
#                      password login is never enabled for root)
set -euo pipefail

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
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
rel  = os.environ["UNSLOTH_VIEW_REL"]
view = os.environ["UNSLOTH_VIEW_DIR"]
print(f"c.ServerApp.default_url = {'/lab/tree/' + rel!r}")
print(f"c.LabApp.default_url = {'/lab/tree/' + rel!r}")
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

STUDIO_NOTE="user unsloth, password from UNSLOTH_STUDIO_PASSWORD env"
if [[ -z "${UNSLOTH_STUDIO_PASSWORD:-}" ]]; then
    if [[ -e "${UNSLOTH_STUDIO_HOME}/auth/auth.db" && ! -s "${UNSLOTH_STUDIO_HOME}/auth/.bootstrap_password" ]]; then
        STUDIO_NOTE="user unsloth, password set on an earlier boot"
    else
        STUDIO_NOTE="user unsloth, generated password printed below once Studio is up"
    fi
fi
echo "Unsloth Studio  -> http://localhost:8000   (${STUDIO_NOTE})"
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
