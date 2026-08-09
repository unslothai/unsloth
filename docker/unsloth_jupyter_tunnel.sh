#!/usr/bin/env bash
# Optional public Cloudflare quick-tunnel for JupyterLab, mirroring the tunnel
# Studio creates for its own UI. Off by default. Two ways to use it:
#
#   * at run time:   docker run -e UNSLOTH_JUPYTER_CLOUDFLARE=1 ... unsloth/unsloth
#                    -> the https://<name>.trycloudflare.com URL is printed in
#                       `docker logs` once JupyterLab is up.
#   * on demand:     docker exec <container> unsloth-jupyter-tunnel --force
#
# The tunnel gives a public https URL that works from anywhere with no account
# or open inbound port. JupyterLab still requires its password, so the notebook
# is not open to the world; treat the URL as sensitive all the same.
set -u

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1
if [ "$FORCE" != "1" ] && [ "${UNSLOTH_JUPYTER_CLOUDFLARE:-0}" != "1" ]; then
    echo "[jupyter-tunnel] disabled (set UNSLOTH_JUPYTER_CLOUDFLARE=1, or run with --force)"
    exit 0
fi

PORT="${JUPYTER_PORT:-8888}"

echo "[jupyter-tunnel] waiting for JupyterLab on port ${PORT} ..."
for _ in $(seq 1 90); do
    if curl -fsS -o /dev/null "http://localhost:${PORT}/login" 2>/dev/null; then
        break
    fi
    sleep 2
done

# Reuse a cloudflared already on the host (Studio caches one for its own
# tunnel); otherwise fetch the static binary for this arch. No account needed.
CFD=""
for cand in \
    "${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}/bin/cloudflared" \
    /usr/local/bin/cloudflared \
    cloudflared; do
    if command -v "$cand" >/dev/null 2>&1; then CFD="$(command -v "$cand")"; break; fi
    [ -x "$cand" ] && { CFD="$cand"; break; }
done
if [ -z "$CFD" ]; then
    case "$(uname -m)" in
        x86_64|amd64) A=amd64;;
        aarch64|arm64) A=arm64;;
        *) A=amd64;;
    esac
    CFD=/usr/local/bin/cloudflared
    echo "[jupyter-tunnel] downloading cloudflared (${A}) ..."
    if ! curl -fsSL -o "$CFD" \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${A}"; then
        echo "[jupyter-tunnel] could not download cloudflared" >&2
        exit 1
    fi
    chmod +x "$CFD"
fi

echo "[jupyter-tunnel] starting Cloudflare quick-tunnel to JupyterLab (port ${PORT})."
echo "[jupyter-tunnel] the https://<name>.trycloudflare.com URL appears below; log in with your Jupyter password."
exec "$CFD" tunnel --no-autoupdate --url "http://localhost:${PORT}"
