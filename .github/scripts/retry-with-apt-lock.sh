#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Run a command that talks to apt, bounded per attempt and retried.
#
# GitHub's Azure apt mirror stalls on these runners. Observed repeatedly in one
# day: `playwright install --with-deps` sat in apt's download loop three times,
# and `Linux deps` (a bare `apt-get update && apt-get install`) sat there for 28
# minutes before anyone noticed. An unbounded apt step is not slow, it is silent:
# it spends the job's whole timeout-minutes, GitHub scores the result as
# "cancelled" rather than a failure, prints no reason, and skips every step after
# it. A shard then reports nothing about its subject because a mirror hiccuped.
#
# Two things make the retry actually work, and both were learned the hard way:
#
#   1. The dpkg lock. apt runs as root, so killing the attempt leaves the
#      apt-get child alive holding /var/lib/dpkg/lock-frontend, and the next
#      attempt dies two seconds later with "Could not get lock". A retry that
#      cannot succeed is worse than none: it buries the real reason under a
#      second, different failure. So wait for the lock, then take it -- the
#      holder is our own orphan and the runner is disposable.
#
#   2. `set -e`. GitHub runs `run:` blocks as `bash -e`, so a bare failing
#      command aborts the step then and there. A retry loop written as
#      `timeout ... ; rc=$?` never reaches its second attempt: the step exits 124
#      with no output at all, which is exactly how the first version of this
#      shipped and why it is a script now rather than eight inline copies.
#
# Usage:
#   bash .github/scripts/retry-with-apt-lock.sh apt-get update
#   bash .github/scripts/retry-with-apt-lock.sh apt-get install -y foo bar
#   bash .github/scripts/retry-with-apt-lock.sh python -m playwright install --with-deps chromium
#
# Environment:
#   RETRY_ATTEMPTS         attempts before giving up   (default 3)
#   RETRY_ATTEMPT_TIMEOUT  seconds per attempt         (default 480)
#
# Deliberately no `set -e`: this script reads exit codes itself, and -e would
# abort it on the very first failing attempt -- the bug described above.
set -uo pipefail

ATTEMPTS="${RETRY_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${RETRY_ATTEMPT_TIMEOUT:-480}"
DPKG_LOCK="/var/lib/dpkg/lock-frontend"

if [ "$#" -eq 0 ]; then
  echo "::error::retry-with-apt-lock.sh needs a command to run" >&2
  exit 2
fi

# Best effort: a runner without fuser still retries, it just cannot wait on the
# lock. Reported once rather than silently, so a retry that fails on a held lock
# is explicable.
have_fuser() { command -v fuser > /dev/null 2>&1; }

release_dpkg_lock() {
  if ! have_fuser; then
    echo "::warning::fuser unavailable; cannot wait on ${DPKG_LOCK}, retrying blind"
    sleep 15
    return 0
  fi
  for _ in $(seq 1 24); do
    sudo fuser "$DPKG_LOCK" > /dev/null 2>&1 || return 0
    sleep 5
  done
  echo "::warning::${DPKG_LOCK} still held after 120s; terminating the holder"
  sudo fuser -k "$DPKG_LOCK" > /dev/null 2>&1 || true
  sleep 5
}

for attempt in $(seq 1 "$ATTEMPTS"); do
  rc=0
  # `|| rc=$?` keeps this exempt from any -e a caller may have set, and records
  # the status instead of ending the script.
  timeout --signal=TERM --kill-after=30 "$ATTEMPT_TIMEOUT" "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    [ "$attempt" -gt 1 ] && echo "::notice::succeeded on attempt ${attempt}"
    exit 0
  fi

  # 124 is timeout's own "I killed it"; 137 is SIGKILL landing after
  # --kill-after. Anything else is the command itself refusing, and saying "did
  # not finish" about a two-second exit sends the next reader looking for a
  # stall that never happened.
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    reason="did not finish within ${ATTEMPT_TIMEOUT}s"
  else
    reason="exited with status ${rc}"
  fi
  echo "::warning::attempt ${attempt}/${ATTEMPTS} of '$*' ${reason}"

  [ "$attempt" -ge "$ATTEMPTS" ] && break
  release_dpkg_lock
done

echo "::error::'$*' failed ${ATTEMPTS} times; the attempt warnings above say which way each one went"
exit 1
