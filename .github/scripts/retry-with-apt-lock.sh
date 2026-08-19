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
#   3. apt's own timeouts, which is the part that makes the retry meaningful
#      rather than merely survivable. Reading the logs of the four stalls above:
#
#        04:47:02  Get:5 https://archive.ubuntu.com/ubuntu noble-security InRelease [126 kB]
#        05:16:29  ##[error]The operation was canceled.
#
#      Twenty-nine minutes of silence, mid-fetch of a 126 kB index file. Two of
#      the four hung on that same file. What happened before it is the other half
#      of the story: azure.archive.ubuntu.com was `Ign:`d four times over thirty
#      seconds, so apt had already failed over through /etc/apt/apt-mirrors.txt to
#      the public archive, which is not provisioned for this fleet.
#
#      apt did not consider any of that an error. `Acquire::http::Timeout`
#      defaults to 120s AND is an idle timeout, so a connection that is open and
#      trickling never trips it -- apt will wait out the heat death of the
#      universe for a socket that is technically still alive. Cutting the timeout
#      to 20s with internal retries turns a 29-minute hang into a ~1 minute
#      failure, which matters for a reason beyond speed: the wall-clock kill below
#      is what orphans the dpkg lock in the first place, so an apt that fails on
#      its own is an apt we never have to kill.
#
#      Deliberately NOT pinning a mirror. The evidence does not support it: in one
#      stall Azure was dead and the public archive hung, in another Azure was
#      serving fine and the transfer stalled at a 13.6 MB package. Neither is
#      reliably better, and the mirrorlist failover is already the right mechanism
#      -- it just needs to be allowed to give up.
#
# Usage:
#   bash .github/scripts/retry-with-apt-lock.sh apt-get update
#   bash .github/scripts/retry-with-apt-lock.sh apt-get install -y foo bar
#   bash .github/scripts/retry-with-apt-lock.sh python -m playwright install --with-deps chromium
#
# Environment:
#   RETRY_ATTEMPTS         attempts before giving up   (default 3)
#   RETRY_ATTEMPT_TIMEOUT  seconds per attempt         (default 480)
#   APT_ACQUIRE_TIMEOUT    seconds apt waits on a stalled transfer (default 20)
#   APT_ACQUIRE_RETRIES    apt's own internal retries  (default 3)
#
# Deliberately no `set -e`: this script reads exit codes itself, and -e would
# abort it on the very first failing attempt -- the bug described above.
set -uo pipefail

ATTEMPTS="${RETRY_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${RETRY_ATTEMPT_TIMEOUT:-480}"
DPKG_LOCK="/var/lib/dpkg/lock-frontend"
APT_TIMEOUT="${APT_ACQUIRE_TIMEOUT:-20}"
APT_RETRIES="${APT_ACQUIRE_RETRIES:-3}"
APT_CONF="/etc/apt/apt.conf.d/99-unsloth-ci-fail-fast"

if [ "$#" -eq 0 ]; then
  echo "::error::retry-with-apt-lock.sh needs a command to run" >&2
  exit 2
fi

# Best effort: a runner without fuser still retries, it just cannot wait on the
# lock. Reported once rather than silently, so a retry that fails on a held lock
# is explicable.
have_fuser() { command -v fuser > /dev/null 2>&1; }

# Make apt give up on a dead transfer instead of waiting on it forever. Written
# before the first attempt rather than baked into the runner image so it applies
# to `playwright install --with-deps` too, which shells out to apt without ever
# naming it. Best effort throughout: a runner where this cannot be written still
# runs the command, just without the fast failure.
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

configure_apt_fail_fast

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
