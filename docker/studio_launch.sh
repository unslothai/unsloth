#!/usr/bin/env bash
# Default CMD of the full Unsloth image (Dockerfile.studio).
#
# Bootstraps the three services managed by supervisord:
#   studio   port 8000   first-boot admin password printed in `docker logs`
#   jupyter  port 8888   password from JUPYTER_PASSWORD (default: unsloth)
#   sshd     port 22     key-only; enabled when PUBLIC_KEY / SSH_KEY is set
#
# Environment:
#   JUPYTER_PORT       Jupyter port inside the container       (default 8888)
#   JUPYTER_PASSWORD   Jupyter login password                  (default unsloth)
#   PUBLIC_KEY/SSH_KEY OpenSSH public key for root login; sshd stays disabled
#                      when neither is set (nothing to authenticate with --
#                      password login is never enabled for root)
set -euo pipefail

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export UNSLOTH_STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"

# Make the runtime env visible to SSH sessions, which get a fresh login shell
# without the `docker run -e` vars. Same pattern as the production image.
printenv | grep -E '^(HF_|CUDA_|NCCL_|JUPYTER_|UNSLOTH_|WANDB_|PATH=|TRITON_)' | \
    sed 's/^\([^=]*\)=\(.*\)$/export \1="\2"/' > /etc/profile.d/unsloth_env.sh || true

# --- Jupyter -----------------------------------------------------------------
# Hash the password with jupyter's own helper; never store the plaintext.
JUPYTER_CONFIG_DIR=/root/.jupyter
if [[ ! -f "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" ]]; then
    mkdir -p "${JUPYTER_CONFIG_DIR}"
    HASH=$(python - <<PY
from jupyter_server.auth import passwd
import os
print(passwd(os.environ.get("JUPYTER_PASSWORD", "unsloth")))
PY
)
    cat > "${JUPYTER_CONFIG_DIR}/jupyter_lab_config.py" <<EOF
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/workspace"
c.PasswordIdentityProvider.hashed_password = "${HASH}"
EOF
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
echo "Unsloth Studio  -> http://localhost:8000   (first-boot password below)"
echo "JupyterLab      -> http://localhost:${JUPYTER_PORT}   (password: JUPYTER_PASSWORD env, default 'unsloth')"
if [[ "${UNSLOTH_ENABLE_SSHD}" == "true" ]]; then
    echo "sshd            -> port 22 (key-only)"
fi

exec supervisord -c /etc/supervisor/supervisord.conf
