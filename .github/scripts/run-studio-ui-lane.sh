#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# One lane of the Windows Chat UI Tests job, start to finish: boot an Unsloth on
# this lane's own port and own UNSLOTH_STUDIO_HOME, rotate the bootstrap
# password, drive its Playwright suites, stop the server.
#
# Usage:  run-studio-ui-lane.sh chat|extra
#
# Why this exists. The job ran five Playwright suites one after another for
# ~788s of a 20.6 minute job, 88% of it in steps over 60s, and it was the last
# finisher on every commit where no macOS job ran. The suites are disjoint --
# each boots its own server and drives its own browser -- and the ports were
# ALREADY distinct (18895/18896/18897/18899). The only thing forcing sequence
# was shared state, exactly as #9158 found for the three Linux indicator
# engines.
#
# Two lanes rather than five. windows-latest is 4 vCPU / 16 GB and these lanes
# load a real GGUF through llama-server, whose turn latency already needed a
# 540s budget on this image. Five concurrent servers would contend for the same
# four cores and turn a timing-sensitive suite into a flaky one. Two splits the
# work near evenly -- 375s against 413s -- so the lane wall is ~413s, and the
# theoretical floor with the 287s indicator suite in it is 287s. Three lanes buy
# ~126s for a third model-loading server; not worth it here.
#
#   chat  : chat UI (18896)  -> loaded-models indicator, Edge (18899)
#   extra : Compare/Recipes/Export/Settings + update banner (18897)
#           -> Edge permission controls (18895)
#
# Within a lane the order and the environment of every suite is unchanged from
# when they were steps, including `extra` reusing one server for both of its
# Playwright runs and passing the same STUDIO_OLD_PW / STUDIO_NEW_PW to each.
#
# THE STATE SPLIT. Per lane:
#
#   port                 -- one each, so both servers coexist. Already true.
#   UNSLOTH_STUDIO_HOME  -- one each. boot-studio-api-only.sh wipes
#       $home/auth so the boot mints a fresh .bootstrap_password, then this
#       script reads that file back. On a shared home the two lanes race
#       destructively: one wipes the password the other just minted and is
#       about to log in with.
#   health probe tmp     -- one each; the helper defaults to a single path.
#   artifact + log paths -- one each, so a failure in one lane is still
#       readable after the other has written its own.
#
# UNSLOTH_STUDIO_HOME is the CLI's INSTALL root, not just a data root, so a bare
# empty directory is not usable: unsloth_cli/commands/studio.py resolves
# $UNSLOTH_STUDIO_HOME/unsloth_studio/Scripts/python.exe and exits "Unsloth
# Unsloth not set up. Run install.sh first." before binding a port. Each lane
# home therefore links the one venv install.ps1 already built and owns only the
# mutable state beside it.
#
# And one thing the Linux precedent did not have to handle. Setting the variable
# makes the root CUSTOM, and _ensure_studio_env_exported() then points
# UNSLOTH_LLAMA_CPP_PATH at $UNSLOTH_STUDIO_HOME/llama.cpp rather than the
# legacy ~/.unsloth/llama.cpp (studio.py, `if _is_legacy`). These lanes load a
# real model, so a lane whose home has no llama.cpp would fail to load rather
# than fall back. It is set explicitly below, which the CLI honours because that
# export is a truthy-check and not an unconditional assignment.

set -euo pipefail

LANE="${1:?usage: $0 chat|extra}"

case "$LANE" in
  chat)  PORT=18896 ;;
  extra) PORT=18897 ;;
  *) echo "run-studio-ui-lane.sh: unknown lane '$LANE'" >&2; exit 2 ;;
esac

installed_home="${UNSLOTH_INSTALLED_STUDIO_HOME:-$HOME/.unsloth/studio}"
# Windows install; Linux path kept so the script is runnable off-CI.
if [ -x "$installed_home/unsloth_studio/Scripts/python.exe" ]; then
  installed_py="$installed_home/unsloth_studio/Scripts/python.exe"
elif [ -x "$installed_home/unsloth_studio/bin/python" ]; then
  installed_py="$installed_home/unsloth_studio/bin/python"
else
  echo "::error::no studio venv under $installed_home; nothing for lane $LANE to link"
  ls -la "$installed_home" 2>/dev/null || true
  exit 1
fi
echo "[lane $LANE] linking venv from $installed_py"

home="${GITHUB_WORKSPACE:-$PWD}/.studio-lane-$LANE"
mkdir -p "$home"

# A junction, not `ln -s`. Under MSYS `ln -s` COPIES by default unless
# MSYS=winsymlinks:nativestrict, and a native symlink needs Developer Mode or
# admin; either way a per-lane copy of the venv would cost more than the
# parallelism saves. `mklink /J` is a directory junction, needs no privilege,
# and is what the filesystem gives you for free.
if [ ! -e "$home/unsloth_studio" ]; then
  if [ "${OS:-}" = "Windows_NT" ]; then
    cmd //c mklink //J "$(cygpath -w "$home/unsloth_studio")" \
                       "$(cygpath -w "$installed_home/unsloth_studio")" >/dev/null
  else
    ln -sfn "$installed_home/unsloth_studio" "$home/unsloth_studio"
  fi
fi

export UNSLOTH_STUDIO_HOME="$home"
# See the header: a custom root would otherwise send this to $home/llama.cpp.
export UNSLOTH_LLAMA_CPP_PATH="${UNSLOTH_LLAMA_CPP_PATH:-$HOME/.unsloth/llama.cpp}"

server_log="logs/studio-lane-$LANE.log"
mkdir -p logs

boot() {
  local port="$1" log="$2"
  # `env -u GITHUB_ENV` is load-bearing, not tidiness. boot-studio-api-only.sh
  # appends the pid to $GITHUB_ENV when it is set and only falls back to stdout
  # when it is not -- and inside a step it is always set. Two concurrent lanes
  # would then both append LANE_PID= to the one file the runner reads after the
  # step, which is both a lost pid and the very kind of shared mutable state
  # this split exists to remove. Unset for this call, the pid comes back on
  # stdout where a per-lane shell can read it. Same reason the password rotation
  # below is inline rather than a `>> $GITHUB_ENV` step of its own.
  env -u GITHUB_ENV bash .github/scripts/boot-studio-api-only.sh \
    --port "$port" --log "$log" --pid-var "LANE_PID" > "logs/boot-$LANE.out" 2>&1
  LANE_PID="$(sed -n 's/^LANE_PID=//p' "logs/boot-$LANE.out" | tail -1)"
  cat "logs/boot-$LANE.out"
  [ -n "$LANE_PID" ] || { echo "::error::lane $LANE: boot returned no pid"; return 1; }
  bash .github/scripts/wait-for-health.sh --port "$port" --log "$log" \
    --tmp "logs/health-$LANE.json"
}

stop() {
  [ -n "${LANE_PID:-}" ] || return 0
  kill "$LANE_PID" 2>/dev/null || true
  sleep 2
  LANE_PID=""
}
trap stop EXIT

# Rotated per lane, from THIS lane's home. Every existing call site reads
# ~/.unsloth/studio/auth/.bootstrap_password directly, which is the legacy path
# and would hand a lane the other lane's password.
mint() {
  STUDIO_OLD_PW="$(cat "$home/auth/.bootstrap_password")"
  STUDIO_NEW_PW="CIUi-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
  STUDIO_NEW2_PW="CIUi-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
  echo "::add-mask::$STUDIO_OLD_PW"
  echo "::add-mask::$STUDIO_NEW_PW"
  echo "::add-mask::$STUDIO_NEW2_PW"
  export STUDIO_OLD_PW STUDIO_NEW_PW STUDIO_NEW2_PW
}

# windows-latest is 4 vCPU / 16 GB and gemma-3-270m turn latency under
# llama-server's CPU backend already crowded the 180s default on this image
# before anything ran beside it. Unchanged from the steps these replace.
export STUDIO_UI_STRICT=1
export STUDIO_UI_TURN_TIMEOUT_MS=540000

if [ "$LANE" = "chat" ]; then
  boot "$PORT" "$server_log"
  mint
  mkdir -p logs/playwright
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright \
    python tests/studio/playwright_chat_ui.py
  # chat-ui ends with a Shutdown click; this is belt-and-suspenders.
  stop

  # Real Edge, which is also the WebView2 the desktop app embeds on Windows.
  bash .github/scripts/run-studio-indicator-browser.sh 18899 chromium msedge
else
  boot "$PORT" "$server_log"
  mint
  mkdir -p logs/playwright_extra
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright_extra \
    python tests/studio/playwright_extra_ui.py

  # The same layout regression the Linux job runs, on the platform whose
  # scrollbars actually take width and whose font metrics are its own: the
  # card's floor and the action row's wrap threshold are both measurements, and
  # a measurement taken on one platform is a guess on the others. Same server
  # and same credentials it had as a step.
  mkdir -p logs/playwright_update_banner
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright_update_banner \
    python tests/studio/playwright_update_banner_layout.py
  stop

  bash .github/scripts/run-studio-permission-browser.sh 18895 chromium msedge
fi

echo "[lane $LANE] done"
