#!/usr/bin/env bash
# Prints the Studio admin credential into `docker logs` (supervisord program
# studio-password, one shot). Studio itself only writes the generated first-boot
# password to a file, so without this the log names the file and nothing else.
#
# Environment:
#   UNSLOTH_STUDIO_PASSWORD          set: Studio applied it, nothing to print
#   UNSLOTH_STUDIO_HOME              where Studio keeps auth/ (default /opt/unsloth-studio)
#   UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT Studio's own change-it-or-shut-down window, seconds
#   UNSLOTH_STUDIO_PASSWORD_WAIT     how long to wait for the file, seconds (default 600)
set -euo pipefail

if [[ -n "${UNSLOTH_STUDIO_PASSWORD:-}" ]]; then
    echo "Unsloth Studio login -> username: unsloth   password: from UNSLOTH_STUDIO_PASSWORD"
    exit 0
fi

AUTH_DIR="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}/auth"
FILE="${AUTH_DIR}/.bootstrap_password"
WAIT="${UNSLOTH_STUDIO_PASSWORD_WAIT:-600}"
TIMEOUT="${UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT:-3600}"
# An auth.db with no bootstrap file means the password was changed on an earlier
# boot (Studio deletes the file then), so there is nothing to wait ten minutes for.
[[ -e "${AUTH_DIR}/auth.db" && ! -s "$FILE" ]] && WAIT=$(( WAIT < 30 ? WAIT : 30 ))

deadline=$(( $(date +%s) + WAIT ))
while [[ ! -s "$FILE" ]]; do
    if (( $(date +%s) >= deadline )); then
        if [[ -e "${AUTH_DIR}/auth.db" ]]; then
            echo "Unsloth Studio login -> username: unsloth   password: already set on an earlier boot" \
                 "(inside the container: unsloth studio reset-password)"
        else
            echo "Unsloth Studio login -> the first-boot password did not appear in ${WAIT}s; check the studio log above"
        fi
        exit 0
    fi
    sleep 1
done

# LF-terminated on every OS; strip only that, a password may end in a space
password="$(tr -d '\r\n' < "$FILE")"
note=""
if [[ "$TIMEOUT" =~ ^[0-9]+$ ]] && (( TIMEOUT > 0 )); then
    note="   (change it on first sign-in: Studio stops after $(( TIMEOUT / 60 )) min with the default password; UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0 disables)"
fi
echo "Unsloth Studio login -> username: unsloth   password: ${password}${note}"
