#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Install one coding-agent CLI for the Local Agent Guides CI. Isolated as
# failure class (b) "agent package install failed": npm/curl flakiness here
# is the single biggest source of false reds, so installs retry with
# backoff and the only ::error:: this script can emit is class (b). The
# install recipes mirror the install_hint strings in
# unsloth_cli/commands/start.py at HEAD.
#
# Usage: agent-guides-install.sh <agent>
#   agent in: claude codex hermes openclaw opencode pi
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

# npm registry flakiness is common in CI; retry 3x with linear backoff.
# Extra npm flags may precede the package (e.g. npm_retry --ignore-scripts pkg).
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

# curl|bash installers, retried at the curl layer. We download to a temp file
# first and only execute on a fully successful fetch, so a truncated download
# (network hiccup mid-stream) can never run a half-written installer.
curl_bash() {
  local url="$1"; shift
  local i tmp
  tmp="$(mktemp)"
  for i in 1 2 3; do
    if curl -fsSL --retry 3 --retry-delay 5 "$url" -o "$tmp" 2>>"$LOG" \
        && bash "$tmp" "$@" >> "$LOG" 2>&1; then
      rm -f "$tmp"
      return 0
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
    # start.py install_hint: curl -fsSL https://claude.ai/install.sh | bash
    curl_bash "https://claude.ai/install.sh" || install_fail "claude installer failed"
    # The installer drops the binary under ~/.local/bin.
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    ;;
  codex)
    # start.py install_hint: npm install -g @openai/codex
    npm_retry "@openai/codex" || install_fail "npm install -g @openai/codex failed"
    ;;
  opencode)
    # start.py install_hint: npm install -g opencode-ai
    npm_retry "opencode-ai" || install_fail "npm install -g opencode-ai failed"
    ;;
  openclaw)
    # start.py install_hint: curl -fsSL https://openclaw.ai/install.sh | bash
    # npm is the more deterministic path in CI and matches the agent's docs;
    # fall back to the start.py curl installer if the npm tag is missing.
    if ! npm_retry "openclaw@latest"; then
      curl_bash "https://openclaw.ai/install.sh" || install_fail "openclaw install failed (npm + curl)"
      echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    fi
    ;;
  hermes)
    # start.py install_hint:
    #   curl -fsSL .../NousResearch/hermes-agent/main/scripts/install.sh | bash
    curl_bash "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh" \
      --non-interactive --skip-setup --skip-browser --no-skills \
      || install_fail "hermes installer failed"
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
    ;;
  pi)
    # start.py install_hint: npm install -g --ignore-scripts @earendil-works/pi-coding-agent
    # (--ignore-scripts matches Pi's documented recipe; exercising the exact hint
    # catches guide drift). The CLI moved from the now-deprecated @mariozechner
    # scope to @earendil-works (the old scope is frozen, so installing it would
    # test a stale Pi against the API).
    npm_retry --ignore-scripts "@earendil-works/pi-coding-agent" \
      || install_fail "npm install -g --ignore-scripts @earendil-works/pi-coding-agent failed"
    ;;
  *)
    install_fail "unknown agent '$AGENT'"
    ;;
esac

echo "[install] OK for $AGENT"
