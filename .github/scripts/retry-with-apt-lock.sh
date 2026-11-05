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

# GitHub's Ubuntu images send apt through mirror+file:$APT_MIRRORS with azure.archive.ubuntu.com
# first. apt fails over only on errors: a mirror trickling at a few KB/s (runner-images #14823)
# never trips Acquire::*::Timeout, so every retry downloads from the same slow node.
# This measures each mirror once and puts the fastest first; a failed attempt demotes the leader.
APT_MIRRORS="${APT_MIRRORS_FILE:-/etc/apt/apt-mirrors.txt}"
MIRROR_PROBE_SECONDS="${APT_MIRROR_PROBE_SECONDS:-5}"

mirror_urls() {
  grep -vE '^[[:space:]]*(#|$)' "$APT_MIRRORS" 2> /dev/null | awk '{print $1}'
}

# bytes/s fetched from a mirror in MIRROR_PROBE_SECONDS; 0 when unreachable. curl prints the rate
# even when --max-time cuts the transfer, which is exactly the slow case being measured.
probe_mirror() {
  speed="$(curl -sS -o /dev/null --max-time "$MIRROR_PROBE_SECONDS" -w '%{speed_download}' \
    "${1%/}/dists/${2}/main/binary-${3}/Packages.gz" 2> /dev/null)"
  speed="${speed%%.*}"
  case "$speed" in '' | *[!0-9]*) speed=0 ;; esac
  printf '%s' "$speed"
}

# Rewrites APT_MIRRORS with the given URLs in order, priority:1 first.
write_mirrors() {
  body=""
  n=0
  for url in "$@"; do
    n=$((n + 1))
    body="${body}${url}	priority:${n}
"
  done
  if ! printf '%s' "$body" | sudo tee "$APT_MIRRORS" > /dev/null 2>&1; then
    echo "::warning::could not rewrite ${APT_MIRRORS}; apt keeps its mirror order"
    return 1
  fi
}

# Once per job: later helper calls in the same job reuse the order (and any demotion).
MIRROR_MARKER="${RUNNER_TEMP:-/tmp}/unsloth-apt-mirror-ranked"

prefer_fast_apt_mirror() {
  [ "${APT_PREFER_FAST_MIRROR:-1}" = "1" ] || return 0
  [ -f "$APT_MIRRORS" ] && command -v curl > /dev/null 2>&1 || return 0
  [ -e "$MIRROR_MARKER" ] && return 0
  : > "$MIRROR_MARKER" 2> /dev/null || true
  urls="$(mirror_urls)"
  [ -n "$urls" ] || return 0
  codename="$(. /etc/os-release 2> /dev/null && printf '%s' "${VERSION_CODENAME:-}")"
  arch="$(dpkg --print-architecture 2> /dev/null || echo amd64)"
  [ -n "$codename" ] || return 0
  # The global archive carries every amd64 pocket, so it is a valid fallback beside any
  # *.archive.ubuntu.com mirror. ports.ubuntu.com (arm64) has no such twin and is left alone.
  if printf '%s\n' "$urls" | grep -q 'archive\.ubuntu\.com/ubuntu' \
    && ! printf '%s\n' "$urls" | grep -qE '^https?://archive\.ubuntu\.com/ubuntu/?$'; then
    urls="$(printf '%s\nhttp://archive.ubuntu.com/ubuntu/' "$urls")"
  fi
  ranked=""
  for url in $urls; do
    speed="$(probe_mirror "$url" "$codename" "$arch")"
    echo "apt mirror ${url}: $((speed / 1024)) KB/s"
    ranked="${ranked}${speed} ${url}
"
  done
  # shellcheck disable=SC2046  # one URL per word
  write_mirrors $(printf '%s' "$ranked" | sort -k1,1nr -s | awk '{print $2}') \
    && echo "apt mirror order: $(mirror_urls | tr '\n' ' ')"
}

# After a failed attempt the leading mirror goes last, so the retry does not reuse it.
demote_apt_mirror() {
  [ "${APT_PREFER_FAST_MIRROR:-1}" = "1" ] && [ -f "$APT_MIRRORS" ] || return 0
  urls="$(mirror_urls)"
  [ "$(printf '%s\n' "$urls" | wc -l)" -gt 1 ] || return 0
  first="$(printf '%s\n' "$urls" | head -n 1)"
  # shellcheck disable=SC2046
  write_mirrors $(printf '%s\n' "$urls" | tail -n +2) "$first" \
    && echo "apt mirror ${first} moved last for the retry"
}

# A timed-out attempt can leave apt-get running under sudo (timeout signals sudo, not its
# grandchildren), still holding the dpkg lock and still downloading from the slow mirror.
stop_orphaned_apt() {
  sudo pkill -TERM -x apt-get > /dev/null 2>&1 || true
  sudo pkill -TERM -x apt > /dev/null 2>&1 || true
  sleep 2
  sudo dpkg --configure -a > /dev/null 2>&1 || true
}

prefer_fast_apt_mirror
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
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    stop_orphaned_apt
  fi
  demote_apt_mirror
  release_apt_locks
done

echo "::error::'$*' failed ${ATTEMPTS} times; the attempt warnings above say which way each one went"
exit 1
