#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Writes (and with --enable/--start, loads) Studio's optional launchd user agent. Must stay bash 3.2 compatible.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/ai.unsloth.studio.server.plist.in"
LABEL="ai.unsloth.studio.server"
MARKER="<!-- unsloth-studio-managed-launchd -->"

_UNSLOTH_EXE=""
_STUDIO_HOME=""
_HOST="${UNSLOTH_LAUNCHD_HOST:-127.0.0.1}"
_PORT="${UNSLOTH_LAUNCHD_PORT:-8888}"
_DO_ENABLE=false
_DO_START=false

_usage() {
    cat <<'EOF'
Usage: install_user_agent.sh --unsloth-exe PATH [options]

Options:
  --unsloth-exe PATH   Absolute path to the unsloth CLI (required)
  --studio-home PATH   Sets UNSLOTH_STUDIO_HOME in the agent (custom installs)
  --host HOST          Bind address (default: 127.0.0.1; UNSLOTH_LAUNCHD_HOST overrides;
                       use 0.0.0.0 for LAN / all-interfaces)
  --port PORT          Listen port (default: 8888; UNSLOTH_LAUNCHD_PORT overrides)
  --enable             Write the agent and clear any launchctl disable, so it starts at login
  --start              Also (re)load it now (implies --enable)
  -h, --help           Show this help
EOF
}

# sed, not ${v//&/...}: bash 5.2 reads & in the replacement as the match, bash 3.2 keeps quotes that would stop it.
_xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --unsloth-exe) _UNSLOTH_EXE="${2:-}"; shift 2 ;;
        --studio-home) _STUDIO_HOME="${2:-}"; shift 2 ;;
        --host) _HOST="${2:-}"; shift 2 ;;
        --port) _PORT="${2:-}"; shift 2 ;;
        --enable) _DO_ENABLE=true; shift ;;
        --start) _DO_ENABLE=true; _DO_START=true; shift ;;
        -h|--help) _usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            _usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$_UNSLOTH_EXE" ]; then
    echo "ERROR: --unsloth-exe is required." >&2
    exit 2
fi
if [ ! -x "$_UNSLOTH_EXE" ]; then
    echo "ERROR: unsloth executable not found or not executable: $_UNSLOTH_EXE" >&2
    exit 1
fi
case "$_PORT" in
    ''|*[!0-9]*|0) echo "ERROR: --port must be a positive integer." >&2; exit 2 ;;
esac
if [ -z "$_HOST" ]; then
    echo "ERROR: --host must not be empty." >&2
    exit 2
fi
for _v in "$_HOST" "$_STUDIO_HOME" "$_UNSLOTH_EXE" "${HOME:-}"; do
    case "$_v" in
        *[[:cntrl:]]*) echo "ERROR: control characters are not allowed in agent values: $_v" >&2; exit 2 ;;
    esac
done
case "${HOME:-}" in
    /*) ;;
    *) echo "ERROR: HOME must be an absolute path." >&2; exit 1 ;;
esac
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: agent template missing: $TEMPLATE" >&2
    exit 1
fi
if [ "$(uname -s 2>/dev/null || true)" != "Darwin" ]; then
    echo "ERROR: launchd user agent install is supported on macOS only." >&2
    exit 1
fi
_uid="$(id -u)"
_domain="gui/$_uid"
if [ "$_DO_ENABLE" = true ]; then
    if ! command -v launchctl >/dev/null 2>&1; then
        echo "ERROR: launchctl not found; launchd is required." >&2
        exit 1
    fi
    # Over SSH with nobody logged in at the console there is no GUI domain to load into.
    if ! launchctl print "$_domain" >/dev/null 2>&1; then
        echo "ERROR: no GUI login session for this user (SSH only?); log in on the Mac once, then re-run." >&2
        exit 1
    fi
fi

_UNSLOTH_EXE="$(CDPATH= cd -P -- "$(dirname "$_UNSLOTH_EXE")" && pwd -P)/$(basename "$_UNSLOTH_EXE")"
_env_home=""
if [ -n "$_STUDIO_HOME" ]; then
    mkdir -p "$_STUDIO_HOME"
    _STUDIO_HOME="$(CDPATH= cd -P -- "$_STUDIO_HOME" && pwd -P)"
    _env_home="$_STUDIO_HOME"
    _run_home="$_STUDIO_HOME"
else
    _run_home="$HOME/.unsloth/studio"
fi
# launchd does not create WorkingDirectory and refuses to start without it.
mkdir -p "$_run_home"

# launchd starts jobs with only /usr/bin:/bin:/usr/sbin:/sbin on PATH.
_exe_dir="$(dirname "$_UNSLOTH_EXE")"
_path="$_exe_dir:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# setup.sh keeps a system Node (e.g. nvm) instead of a managed one; Studio finds it on PATH.
_node="$(command -v node 2>/dev/null || true)"
case "$_node" in
    *[[:cntrl:]]*) ;;
    /*) _path="$_exe_dir:$(dirname "$_node"):${_path#"$_exe_dir:"}" ;;
esac

# launchd scans the login account's home at login, not a redirected $HOME.
_account_home="$(dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2>/dev/null | awk '{print $2}' || true)"
case "$_account_home" in /*) ;; *) _account_home="$HOME" ;; esac
_agent_dir="$_account_home/Library/LaunchAgents"
mkdir -p "$_agent_dir"
_plist_path="$_agent_dir/$LABEL.plist"
if [ -e "$_plist_path" ] && [ "$(sed -n 2p "$_plist_path" 2>/dev/null)" != "$MARKER" ]; then
    echo "ERROR: $_plist_path exists and was not written by Unsloth; leaving it alone." >&2
    exit 1
fi
_tmp="$(mktemp "$_agent_dir/.$LABEL.XXXXXX")"
trap 'rm -f "$_tmp"' EXIT
while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
        *"<string>@@UNSLOTH_EXE@@</string>") printf '    <string>%s</string>\n' "$(_xml_escape "$_UNSLOTH_EXE")" ;;
        *"<string>@@HOST@@</string>") printf '    <string>%s</string>\n' "$(_xml_escape "$_HOST")" ;;
        *"<string>@@PORT@@</string>") printf '    <string>%s</string>\n' "$_PORT" ;;
        *"<string>@@PATH@@</string>") printf '    <string>%s</string>\n' "$(_xml_escape "$_path")" ;;
        @@ENVIRONMENT_LINES@@)
            if [ -n "$_env_home" ]; then
                printf '    <key>UNSLOTH_STUDIO_HOME</key>\n    <string>%s</string>\n' "$(_xml_escape "$_env_home")"
            fi
            # launchd hands the job the account's HOME; Studio resolves its llama.cpp dir from the one it was installed under.
            if [ "$HOME" != "$_account_home" ]; then
                printf '    <key>HOME</key>\n    <string>%s</string>\n' "$(_xml_escape "$HOME")"
            fi
            ;;
        *"<string>@@STUDIO_HOME@@</string>") printf '  <string>%s</string>\n' "$(_xml_escape "$_run_home")" ;;
        *) printf '%s\n' "$_line" ;;
    esac
done < "$TEMPLATE" > "$_tmp"
chmod 0644 "$_tmp"
mv "$_tmp" "$_plist_path"

if [ "$_DO_ENABLE" != true ]; then
    printf '%s\n' "$_plist_path"
    exit 0
fi

# A plist left after a failed load would still start at next login.
_fail_load() {
    rm -f "$_plist_path"
    echo "ERROR: $1; removed $_plist_path so it does not start at next login." >&2
    exit 1
}

# Clears a `launchctl disable` override, which would otherwise keep it from loading at login.
launchctl enable "$_domain/$LABEL" || _fail_load "launchctl enable $_domain/$LABEL failed"
if [ "$_DO_START" = true ]; then
    if launchctl print "$_domain/$LABEL" >/dev/null 2>&1; then
        launchctl bootout "$_domain/$LABEL" >/dev/null 2>&1 || true
        # bootstrap fails ("5: Input/output error") until the old job unloads; wait past ExitTimeOut (120s).
        _waited=0
        while launchctl print "$_domain/$LABEL" >/dev/null 2>&1; do
            if [ "$_waited" -ge 130 ]; then
                _fail_load "$LABEL did not unload within 130s; try: launchctl bootout $_domain/$LABEL"
            fi
            sleep 1
            _waited=$((_waited + 1))
        done
    fi
    launchctl bootstrap "$_domain" "$_plist_path" || _fail_load "launchctl bootstrap $_domain failed"
fi
printf '%s\n' "$_plist_path"
