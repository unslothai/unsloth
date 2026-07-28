#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Simulate a virgin developer machine on a GitHub-hosted runner. Two modes, because
# "the tool is absent" and "the installer never called the tool" cannot be simulated
# by the same mechanism:
#
#   mask   Make the toolchain genuinely ABSENT: scrub PATH to OS defaults and (with
#          --remove) move the real toolchain aside, so `command -v git` correctly
#          FAILS, as on a clean Mac. A failing "poison shim" would do the opposite --
#          `command -v` finds it and reports the tool as present -- so no shims here.
#   trace  Leave the toolchain working but route it through logging wrappers that log
#          the call then exec the real binary, proving whether the installer ever
#          REACHES for a compiler/git without changing behaviour.
#
# Writes shell exports to $CLEAN_ENV_FILE (default ./clean-machine.env) to `source`;
# nothing is exported globally, so other steps keep a normal environment.
#
# Usage:
#   bash .github/scripts/clean-machine-env.sh mask [--remove]
#   bash .github/scripts/clean-machine-env.sh trace
#   source ./clean-machine.env
set -uo pipefail

MODE="${1:-}"
REMOVE=0
[ "${2:-}" = "--remove" ] && REMOVE=1

case "$MODE" in
  mask|trace) ;;
  *) echo "usage: $0 {mask|trace} [--remove]" >&2; exit 2 ;;
esac

OS="$(uname -s)"
WORK="${CLEAN_MACHINE_DIR:-$PWD/.clean-machine}"
ENV_FILE="${CLEAN_ENV_FILE:-$PWD/clean-machine.env}"
TRACE="$WORK/tool-invocations.log"
BIN="$WORK/bin"
RESTORE="$WORK/restore.sh"
mkdir -p "$BIN"
: > "$TRACE"
: > "$ENV_FILE"
printf '#!/usr/bin/env bash\n# Undo clean-machine-env.sh --remove. Safe to run twice.\nset -uo pipefail\n' > "$RESTORE"
chmod +x "$RESTORE"

# The toolchain we care about: a consumer install must need none of it.
TOOLS="xcode-select xcrun clang clang++ cc c++ gcc g++ git cmake make brew ninja cargo rustc"

note() { echo "[clean-machine] $*"; }

# ── PATH scrub ────────────────────────────────────────────────────────────────
# Keep only OS-default system dirs: drops Homebrew, the hosted Python toolcache,
# setup-* shims, pipx, cargo and every other preinstalled developer dir.
scrub_path() {
  local keep out=""
  if [ "$OS" = "Darwin" ]; then
    keep="/usr/bin:/bin:/usr/sbin:/sbin"
  else
    keep="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  fi
  local IFS=":"
  for d in $keep; do
    [ -d "$d" ] && out="${out:+$out:}$d"
  done
  echo "$out"
}

# ── mask ──────────────────────────────────────────────────────────────────────
if [ "$MODE" = "mask" ]; then
  NEWPATH="$(scrub_path)"
  {
    echo "export PATH='$NEWPATH'"
    # UNSET, not a fake path: `xcode-select -p` honours DEVELOPER_DIR and prints it
    # verbatim with exit 0, so a nonexistent dir makes the probe SUCCEED. On a clean
    # Mac it is unset and the missing xcode_select_link is what makes the probe fail.
    echo "unset DEVELOPER_DIR || true"
    echo "unset SDKROOT CC CXX CFLAGS CXXFLAGS LDFLAGS CMAKE_GENERATOR CMAKE_PREFIX_PATH || true"
    echo "export HOMEBREW_NO_AUTO_UPDATE=1"
    echo "export UNSLOTH_CLEAN_MACHINE=1"
  } >> "$ENV_FILE"

  if [ "$REMOVE" = "1" ] && [ "$OS" = "Darwin" ]; then
    # Best-effort real removal; each step is independent and recorded in restore.sh
    # so an `if: always()` step can put the runner back. xcode_select_link is exactly
    # what `xcode-select -p` reads, so removing it reproduces a virgin Mac's gate.
    # `xcode-select --reset` is NOT enough: it can reselect a full Xcode.app.
    if [ -e /var/db/xcode_select_link ]; then
      if sudo rm -f /var/db/xcode_select_link 2>/dev/null; then
        note "removed /var/db/xcode_select_link"
        echo "sudo xcode-select --switch /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not remove /var/db/xcode_select_link"
      fi
    fi
    # Moving the CLT dir aside turns /usr/bin/{cc,clang,git} into dead shims, so the
    # run also proves the install needs no compiler at all.
    if [ -d /Library/Developer/CommandLineTools ]; then
      if sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools.masked 2>/dev/null; then
        note "moved CommandLineTools aside"
        echo "sudo mv /Library/Developer/CommandLineTools.masked /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move CommandLineTools"
      fi
    fi
    # Xcode.app must go too: with the link removed AND CommandLineTools moved,
    # `xcode-select -p` still does not fail, it falls through to the image's Xcode
    # bundle (observed: /Applications/Xcode_16.4.app/Contents/Developer), which
    # re-arms /usr/bin/{git,cc} and silently un-cleans the machine. A rename is
    # instant regardless of bundle size: same filesystem, no copy.
    for app in /Applications/Xcode*.app; do
      [ -d "$app" ] || continue
      if sudo mv "$app" "${app}.masked" 2>/dev/null; then
        note "moved $(basename "$app") aside"
        echo "sudo mv '${app}.masked' '$app' 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move $app"
      fi
    done
    for brewdir in /opt/homebrew /usr/local/Homebrew; do
      if [ -d "$brewdir" ]; then
        if sudo mv "$brewdir" "${brewdir}.masked" 2>/dev/null; then
          note "moved $brewdir aside"
          echo "sudo mv '${brewdir}.masked' '$brewdir' 2>/dev/null || true" >> "$RESTORE"
        else
          note "WARN could not move $brewdir"
        fi
      fi
    done
  fi
fi

# ── trace ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "trace" ]; then
  for tool in $TOOLS; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [ -n "$real" ] || continue
    # Logs the call then execs the REAL binary: behaviour unchanged, so the trace
    # answers "did the installer reach for this?" honestly.
    cat > "$BIN/$tool" <<WRAP
#!/bin/sh
printf '%s\t%s\n' "$tool" "\$*" >> "$TRACE"
exec "$real" "\$@"
WRAP
    chmod +x "$BIN/$tool"
  done
  {
    echo "export PATH='$BIN:$PATH'"
    echo "export UNSLOTH_TOOL_TRACE='$TRACE'"
    echo "export UNSLOTH_CLEAN_MACHINE=trace"
  } >> "$ENV_FILE"
fi

note "mode=$MODE remove=$REMOVE"
note "env file: $ENV_FILE"
note "trace:    $TRACE"
note "restore:  $RESTORE"
