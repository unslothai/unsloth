#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Run an apt-using command with a per-attempt timeout, retries and apt fail-fast settings.
# Waits for and clears all four apt locks an orphaned attempt may hold.
# Usage:
#   bash .github/scripts/retry-with-apt-lock.sh apt-get update
#   bash .github/scripts/retry-with-apt-lock.sh python -m playwright install --with-deps chromium
# Environment:
#   RETRY_ATTEMPTS         attempts before giving up   (default 3)
#   RETRY_ATTEMPT_TIMEOUT  seconds per attempt         (default 480)
#   APT_ACQUIRE_TIMEOUT    seconds apt waits on a stalled transfer (default 20)
#   APT_ACQUIRE_RETRIES    apt's own internal retries  (default 3)
# No `set -e`: this script reads exit codes itself.
set -uo pipefail

ATTEMPTS="${RETRY_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${RETRY_ATTEMPT_TIMEOUT:-480}"
# dpkg's two locks cover install, lists covers update, archives covers the download cache.
APT_LOCKS="/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock"
APT_TIMEOUT="${APT_ACQUIRE_TIMEOUT:-20}"
APT_RETRIES="${APT_ACQUIRE_RETRIES:-3}"
APT_CONF="/etc/apt/apt.conf.d/99-unsloth-ci-fail-fast"

if [ "$#" -eq 0 ]; then
  echo "::error::retry-with-apt-lock.sh needs a command to run" >&2
  exit 2
fi

have_fuser() { command -v fuser > /dev/null 2>&1; }

# Applied globally so `playwright install --with-deps`, which calls apt indirectly, fails fast too.
configure_apt_fail_fast() {
  [ -d /etc/apt/apt.conf.d ] || return 0
  conf="Acquire::Retries \"${APT_RETRIES}\";
Acquire::http::Timeout \"${APT_TIMEOUT}\";
Acquire::https::Timeout \"${APT_TIMEOUT}\";
Acquire::ftp::Timeout \"${APT_TIMEOUT}\";"
  if ! printf '%s\n' "$conf" | sudo tee "$APT_CONF" > /dev/null 2>&1; then
    echo "::warning::could not write ${APT_CONF}; apt keeps its 120s idle timeout"
    return 0
  fi
  echo "apt configured to fail fast: ${APT_TIMEOUT}s transfer timeout, ${APT_RETRIES} internal retries"
}

# `fuser` on an absent file is an error, so missing lock files are skipped.
held_apt_locks() {
  held=""
  for lock in $APT_LOCKS; do
    [ -e "$lock" ] || continue
    if sudo fuser "$lock" > /dev/null 2>&1; then
      held="$held $lock"
    fi
  done
  printf '%s' "$held"
}

release_apt_locks() {
  if ! have_fuser; then
    echo "::warning::fuser unavailable; cannot wait on the apt locks, retrying blind"
    sleep 15
    return 0
  fi
  for _ in $(seq 1 24); do
    holding="$(held_apt_locks)"
    [ -n "$holding" ] || return 0
    sleep 5
  done
  holding="$(held_apt_locks)"
  [ -n "$holding" ] || return 0
  echo "::warning::still held after 120s, terminating the holders:${holding}"
  for lock in $holding; do
    sudo fuser -k "$lock" > /dev/null 2>&1 || true
  done
  sleep 5
}

configure_apt_fail_fast

for attempt in $(seq 1 "$ATTEMPTS"); do
  rc=0
  # `|| rc=$?` stays exempt from any caller -e.
  timeout --signal=TERM --kill-after=30 "$ATTEMPT_TIMEOUT" "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    [ "$attempt" -gt 1 ] && echo "::notice::succeeded on attempt ${attempt}"
    exit 0
  fi

  # 124 is timeout's kill and 137 is SIGKILL after --kill-after; anything else is the command.
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    reason="did not finish within ${ATTEMPT_TIMEOUT}s"
  else
    reason="exited with status ${rc}"
  fi
  echo "::warning::attempt ${attempt}/${ATTEMPTS} of '$*' ${reason}"

  [ "$attempt" -ge "$ATTEMPTS" ] && break
  release_apt_locks
done

echo "::error::'$*' failed ${ATTEMPTS} times; the attempt warnings above say which way each one went"
exit 1
