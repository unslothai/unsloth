#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Install one agent CLI for the Local Agent Guides CI; recipes mirror start.py's install_hint.
# Usage: agent-guides-install.sh <agent>
set -uo pipefail

AGENT="${1:?usage: agent-guides-install.sh <agent>}"
mkdir -p logs
LOG="logs/install-${AGENT}.log"

install_fail() {
  echo "::error::[agent install failed] agent=${AGENT}: $* (class (b): the agent CLI did not install; not a server or guide problem)." >&2
  echo "---- tail $LOG ----" >&2
  tail -60 "$LOG" 2>/dev/null || true
  exit 1
}

npm_retry() {
  local i
  for i in 1 2 3; do
    if npm install -g "$@" >> "$LOG" 2>&1; then
      return 0
    fi
    echo "[install] npm install -g $* attempt $i failed; backing off $((i * 10))s" | tee -a "$LOG"
    sleep "$((i * 10))"
  done
  return 1
}

# A `?--flag` option is optional and passed only while the installer still parses it as a case label;
# vendors treat unknown options as fatal.
installer_args() {
  local script="$1"; shift
  local arg flag
  for arg in "$@"; do
    if [[ "$arg" == \?* ]]; then
      flag="${arg#\?}"
      if grep -qE -- "^[[:space:]]*\(?([^[:space:]#|()]+\|)*${flag}(\|[^[:space:]|()]+)*\)" "$script"; then
        printf '%s\n' "$flag"
      else
        echo "[install] the installer no longer takes $flag; installing without it" | tee -a "$LOG" >&2
      fi
    else
      printf '%s\n' "$arg"
    fi
  done
}

# Download fully before executing so a truncated installer never runs.
curl_bash() {
  local url="$1"; shift
  local i tmp
  local arg
  local -a args
  tmp="$(mktemp)"
  for i in 1 2 3; do
    if curl -fsSL --retry 3 --retry-delay 5 "$url" -o "$tmp" 2>>"$LOG"; then
      args=()
      # A read loop: macOS ships bash 3.2, which lacks the array-reading builtins.
      while IFS= read -r arg; do
        args+=("$arg")
      done < <(installer_args "$tmp" "$@")
      if bash "$tmp" ${args[@]+"${args[@]}"} >> "$LOG" 2>&1; then
        rm -f "$tmp"
        return 0
      fi
    fi
    echo "[install] curl|bash $url attempt $i failed; backing off $((i * 10))s" | tee -a "$LOG"
    sleep "$((i * 10))"
  done
  rm -f "$tmp"
  return 1
}

echo "[install] agent=$AGENT (log=$LOG)"
case "$AGENT" in
  claude)
    curl_bash "https://claude.ai/install.sh" || install_fail "claude installer failed"
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    ;;
  codex)
    npm_retry "@openai/codex" || install_fail "npm install -g @openai/codex failed"
    ;;
  opencode)
    case "${OPENCODE_CHANNEL:-stable}" in
      stable) package="opencode-ai" ;;
      v2)
        package="@opencode-ai/cli@beta"
        if latest_bin="$(npm view @opencode-ai/cli@latest bin --json 2>>"$LOG")" \
            && grep -q '"opencode2"' <<<"$latest_bin"; then
          package="@opencode-ai/cli@latest"
        fi
        ;;
      *) install_fail "unknown OpenCode channel '${OPENCODE_CHANNEL}'" ;;
    esac
    npm_retry "$package" || install_fail "npm install -g $package failed"
    ;;
  openclaw)
    # Prefer npm in CI; fall back to the start.py curl installer.
    if ! npm_retry "openclaw@latest"; then
      curl_bash "https://openclaw.ai/install.sh" || install_fail "openclaw install failed (npm + curl)"
      echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    fi
    ;;
  hermes)
    curl_bash "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh" \
      --non-interactive '?--skip-setup' '?--skip-browser' '?--no-skills' \
      || install_fail "hermes installer failed"
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    ;;
  pi)
    # --ignore-scripts matches start.py's install_hint; the old @mariozechner scope is frozen.
    npm_retry --ignore-scripts "@earendil-works/pi-coding-agent" \
      || install_fail "npm install -g --ignore-scripts @earendil-works/pi-coding-agent failed"
    ;;
  dsh)
    npm_retry "@deepseek-ai/dsh" || install_fail "npm install -g @deepseek-ai/dsh failed"
    ;;
  vibe)
    PATH="$HOME/.local/bin:$PATH" curl_bash "https://mistral.ai/vibe/install.sh" \
      || install_fail "vibe installer failed"
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    ;;
  *)
    install_fail "unknown agent '$AGENT'"
    ;;
esac

echo "[install] OK for $AGENT"
