#!/usr/bin/env bash
# Default CMD of the full Unsloth image (Dockerfile.studio).
#
# Bootstraps the three services managed by supervisord:
#   studio   port 8000   first-boot admin password printed in `docker logs`
#   jupyter  port 8888   password from JUPYTER_PASSWORD, or a random one
#                        printed in `docker logs` when unset
#   sshd     port 22     key-only; enabled when PUBLIC_KEY / SSH_KEY is set
#
# Environment:
#   JUPYTER_PORT       Jupyter port inside the container       (default 8888)
#   JUPYTER_PASSWORD   Jupyter login password (unset: generated and printed)
#   PUBLIC_KEY/SSH_KEY OpenSSH public key for root login; sshd stays disabled
#                      when neither is set (nothing to authenticate with --
#                      password login is never enabled for root)
set -euo pipefail

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export UNSLOTH_STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
# Default off so supervisord's %(ENV_UNSLOTH_JUPYTER_CLOUDFLARE)s autostart gate
# resolves; set to 1 (docker run -e) to expose JupyterLab on a trycloudflare URL.
export UNSLOTH_JUPYTER_CLOUDFLARE="${UNSLOTH_JUPYTER_CLOUDFLARE:-0}"

# Make the runtime env visible to SSH sessions, which get a fresh login shell
# without the `docker run -e` vars. Secrets are excluded on purpose: tokens,
# API keys and passwords stay in process env only, never on disk where an
# SSH session (or anything reading /etc/profile.d) could pick them up.
# shlex.quote() each value: env vars can contain quotes, $, backticks etc,
# and this file is sourced by every login shell.
python - > /etc/profile.d/unsloth_env.sh <<'PY' || true
import os, re, shlex
keep   = re.compile(r"^(HF_|CUDA_|NCCL_|JUPYTER_|UNSLOTH_|WANDB_|TRITON_)|^PATH$")
secret = re.compile(r"(_TOKEN|_API_KEY|_PASSWORD|_SECRET|_LICENSE)$")
for key, value in sorted(os.environ.items()):
    if keep.search(key) and not secret.search(key):
        print(f"export {key}={shlex.quote(value)}")
PY

# --- Jupyter -----------------------------------------------------------------
# Hash the password with jupyter's own helper; never store the plaintext.
# No fixed default password: when JUPYTER_PASSWORD is unset we generate a
# random one and print it once in the boot banner (docker logs).
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
    # Land straight in the categorized notebook view, but only when it is enabled
    # AND lives under root_dir (so it is expressible as a /lab/tree path). Mirror
    # unsloth_sync_notebooks.sh's gating -- UNSLOTH_NOTEBOOKS_VIEW_DIR plus both
    # UNSLOTH_SKIP_NOTEBOOK_VIEW (no view built) and UNSLOTH_SKIP_NOTEBOOK_SYNC
    # (entrypoint skips sync entirely, so nothing under the view dir exists) -- so
    # a relocated, disabled, or unsynced view never points JupyterLab at a missing
    # dir; in those cases JupyterLab just opens on its default (/lab) over /workspace.
    _root_dir="/workspace"
    _view_dir="${UNSLOTH_NOTEBOOKS_VIEW_DIR:-/workspace/Unsloth Notebooks}"
    if [[ "${UNSLOTH_SKIP_NOTEBOOK_VIEW:-0}" != "1" \
          && "${UNSLOTH_SKIP_NOTEBOOK_SYNC:-0}" != "1" \
          && "${_view_dir}" == "${_root_dir}/"* ]]; then
        _view_rel="${_view_dir#${_root_dir}/}"
        # default_url must be set on BOTH ServerApp and LabApp -- the lab
        # extension app otherwise overrides ServerApp's value back to /lab.
        # preferred_dir points the file browser at that folder. A literal space
        # is URL-encoded to %20 in the redirect itself.
        cat >> "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" <<EOF
c.ServerApp.default_url = "/lab/tree/${_view_rel}"
c.LabApp.default_url = "/lab/tree/${_view_rel}"
c.ServerApp.preferred_dir = "${_view_dir}"
EOF
    fi
fi

# --- sshd (opt-in) -----------------------------------------------------------
# Enabled only when a public key is provided; root password login is never
# allowed. Cloud GPU platforms (e.g. runpod-style hosts) inject PUBLIC_KEY.
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

# --- Branding / AGPLv3 attribution integrity gate (whole container) -----------
# This image is built by Unsloth and ships under the GNU AGPLv3. Refuse to start
# anything if the Unsloth attribution (Help/About, spinning-logo splash, branded
# login, theme, AGPLv3 license text + source links) has been stripped or altered.
# The same checker also runs as a jupyter_server extension (refuses JupyterLab on
# its own) and at image-build time. Bypass for local development if ever needed:
# UNSLOTH_SKIP_BRANDING_CHECK=1 (intended for Unsloth's own debugging, not resale).
if [[ "${UNSLOTH_SKIP_BRANDING_CHECK:-0}" != "1" ]]; then
    if ! /opt/unsloth-venv/bin/python -m unsloth_branding --verify; then
        echo "Refusing to start the container." >&2
        exit 1
    fi
fi

echo "Unsloth Studio  -> http://localhost:8000   (first-boot password below)"
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
