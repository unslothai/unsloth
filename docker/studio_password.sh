#!/usr/bin/env bash
# One-shot supervisord program (studio-password): prints how to sign in to Studio,
# then a ready summary once Studio and JupyterLab answer. Studio itself only writes
# the generated first-boot password to a file, so without this the log names the
# file and nothing else, and nothing marks the point where the container is usable.
#
# Environment:
#   UNSLOTH_STUDIO_PASSWORD_STATE    from the launcher: initial | stored | generated
#   UNSLOTH_STUDIO_HOME              where Studio keeps auth/ (default /opt/unsloth-studio)
#   UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT Studio's own change-it-or-shut-down window, seconds
#   UNSLOTH_STUDIO_PASSWORD_WAIT     how long to wait for the file, seconds (default 600)
#   UNSLOTH_STUDIO_READY_WAIT        how long to wait for both services, seconds (default 900)
#   JUPYTER_PORT, UNSLOTH_JUPYTER_NOTE  from the launcher, for the summary
set -euo pipefail

AUTH_DIR="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}/auth"
FILE="${AUTH_DIR}/.bootstrap_password"
WAIT="${UNSLOTH_STUDIO_PASSWORD_WAIT:-600}"
READY_WAIT="${UNSLOTH_STUDIO_READY_WAIT:-900}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
STATE="${UNSLOTH_STUDIO_PASSWORD_STATE:-}"
[[ -z "$STATE" && -n "${UNSLOTH_STUDIO_PASSWORD:-}" ]] && STATE=initial

# Same rules as studio/backend/auth/bootstrap_timeout.py: unset/blank/malformed = 3600,
# <= 0 disables; whitespace stripped and single underscores allowed, as int() does.
raw="${UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT:-}"
raw="${raw#"${raw%%[![:space:]]*}"}"
raw="${raw%"${raw##*[![:space:]]}"}"
if [[ "$raw" =~ ^[+-]?[0-9]+(_[0-9]+)*$ ]]; then
    digits="${raw#[+-]}"
    digits="${digits//_/}"
    if [[ "$raw" == -* ]]; then TIMEOUT=0; else TIMEOUT=$(( 10#$digits )); fi
else
    TIMEOUT=3600
fi

# Mirrors _format_duration there: seconds under a minute, else minutes and a remainder.
plural() { echo "$1 $2$([[ "$1" == 1 ]] || echo s)"; }
duration() {
    local s=$1
    if (( s < 60 )); then plural "$s" second; return; fi
    local text; text="$(plural $(( s / 60 )) minute)"
    (( s % 60 )) && text="$text $(plural $(( s % 60 )) second)"
    echo "$text"
}

case "$STATE" in
    initial)
        STUDIO_LINE="username: unsloth   password: from UNSLOTH_STUDIO_PASSWORD" ;;
    stored)
        STUDIO_LINE="username: unsloth   password: already set on an earlier boot (inside the container: unsloth studio reset-password)" ;;
    *)
        deadline=$(( $(date +%s) + WAIT ))
        STUDIO_LINE=""
        # The CLI writes the file before committing the admin row, so wait for the row.
        while [[ ! -s "$FILE" ]] || ! unsloth-studio-run --initialized; do
            if (( $(date +%s) >= deadline )); then
                STUDIO_LINE="the first-boot password did not appear in ${WAIT}s; check the Unsloth Studio log above"
                break
            fi
            sleep 1
        done
        if [[ -z "$STUDIO_LINE" ]]; then
            # LF-terminated on every OS; strip only that, a password may end in a space
            password="$(tr -d '\r\n' < "$FILE")"
            note=""
            if (( TIMEOUT > 0 )); then
                note="   (change it on first sign-in: Unsloth Studio stops after $(duration "$TIMEOUT") with the default password; UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0 disables)"
            fi
            STUDIO_LINE="username: unsloth   password: ${password}${note}"
        fi ;;
esac
echo "Unsloth Studio login -> ${STUDIO_LINE}"

# Studio's socket opens before torch loads; wait for /api/health 200.
studio_ok=""; jupyter_ok=""
deadline=$(( $(date +%s) + READY_WAIT ))
while :; do
    if [[ -z "$studio_ok" ]] && curl -sf -o /dev/null --max-time 3 --noproxy '*' "http://127.0.0.1:${UNSLOTH_STUDIO_PORT:-8000}/api/health"; then
        studio_ok=1
    fi
    if [[ -z "$jupyter_ok" ]] && curl -sf -o /dev/null --max-time 3 --noproxy '*' "http://127.0.0.1:${JUPYTER_PORT}/login"; then
        jupyter_ok=1
    fi
    { [[ -n "$studio_ok" && -n "$jupyter_ok" ]] || (( $(date +%s) >= deadline )); } && break
    sleep 2
done

if [[ -n "$studio_ok" && -n "$jupyter_ok" ]]; then
    title="Unsloth container ready"
else
    title="Unsloth container: startup incomplete after ${READY_WAIT}s, see the log above"
fi
studio_text="not answering"
[[ -n "$studio_ok" ]] && studio_text="$STUDIO_LINE"
jupyter_text="not answering"
[[ -n "$jupyter_ok" ]] && jupyter_text="${UNSLOTH_JUPYTER_NOTE:-password from JUPYTER_PASSWORD env}"
# Colour the summary so it stands out from the startup noise; NO_COLOR turns it off.
if [[ -n "${NO_COLOR:-}" ]]; then
    _b=""; _g=""; _y=""; _r=""
else
    _b=$'\033[1m'; _g=$'\033[1;32m'; _y=$'\033[1;33m'; _r=$'\033[0m'
fi
rule="$(printf '=%.0s' $(seq 1 72))"
printf '%s\n' "${_g}${rule}${_r}"
printf '  %s\n' "${_g}${title}${_r}"
printf '  Unsloth     %s   %s\n' "${_b}http://localhost:${UNSLOTH_STUDIO_PORT:-8000}${_r}" "${_y}${studio_text}${_r}"
printf '  JupyterLab  %s   %s\n' "${_b}http://localhost:${JUPYTER_PORT}${_r}" "${_y}${jupyter_text}${_r}"
printf '  %s\n' "Ports are the container's: use the host side of your -p flags, or an SSH tunnel to a remote host."
printf '%s\n' "${_g}${rule}${_r}"
