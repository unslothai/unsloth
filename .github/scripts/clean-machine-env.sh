#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Simulate a virgin developer machine on a GitHub-hosted runner, so the installer
# is exercised the way a real user's brand-new Mac / PC exercises it.
#
# Two modes, because "the tool is absent" and "the installer never called the tool"
# CANNOT be simulated by the same mechanism:
#
#   mask   Make the toolchain genuinely ABSENT. Scrubs PATH down to the OS
#          defaults and (with --remove) moves the real toolchain aside. After
#          this, `command -v git` correctly FAILS, which is what a clean Mac does.
#          A failing "poison shim" on PATH would do the opposite -- `command -v`
#          finds it and reports the tool as present -- so shims are NOT used here.
#
#   trace  Leave the toolchain working, but route it through logging wrappers that
#          record the invocation and then exec the real binary. Proves whether the
#          installer ever REACHES for a compiler/git, without changing behaviour.
#
# Writes shell exports to $CLEAN_ENV_FILE (default ./clean-machine.env) for the
# caller to `source`. Nothing is exported globally, so other workflow steps
# (checkout, upload-artifact) keep a normal environment.
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
# Keep only OS-default system dirs. Drops Homebrew, the hosted Python toolcache,
# setup-* shims, pipx, cargo, and every other preinstalled developer dir.
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
    # DEVELOPER_DIR must be UNSET, not pointed at a fake path: `xcode-select -p`
    # honours DEVELOPER_DIR and prints it verbatim with exit 0, so setting it to a
    # nonexistent dir makes the probe SUCCEED -- the exact opposite of a clean Mac,
    # where DEVELOPER_DIR is unset and the missing /var/db/xcode_select_link is what
    # makes `xcode-select -p` fail.
    echo "unset DEVELOPER_DIR || true"
    echo "unset SDKROOT CC CXX CFLAGS CXXFLAGS LDFLAGS CMAKE_GENERATOR CMAKE_PREFIX_PATH || true"
    echo "export HOMEBREW_NO_AUTO_UPDATE=1"
    echo "export UNSLOTH_CLEAN_MACHINE=1"
  } >> "$ENV_FILE"

  if [ "$REMOVE" = "1" ] && [ "$OS" = "Darwin" ]; then
    # Best-effort real removal. Each step is independent and recorded in
    # restore.sh so an `if: always()` step can put the runner back.
    # /var/db/xcode_select_link is exactly what `xcode-select -p` reads, so
    # removing it reproduces a virgin Mac's gate precisely. `xcode-select --reset`
    # is NOT enough: it can reselect a full Xcode.app.
    if [ -e /var/db/xcode_select_link ]; then
      if sudo rm -f /var/db/xcode_select_link 2>/dev/null; then
        note "removed /var/db/xcode_select_link"
        echo "sudo xcode-select --switch /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not remove /var/db/xcode_select_link"
      fi
    fi
    # Moving the CLT dir aside turns /usr/bin/{cc,clang,git} into dead shims, so
    # the run also proves the install needs no compiler at all.
    if [ -d /Library/Developer/CommandLineTools ]; then
      if sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools.masked 2>/dev/null; then
        note "moved CommandLineTools aside"
        echo "sudo mv /Library/Developer/CommandLineTools.masked /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move CommandLineTools"
      fi
    fi
    # Xcode.app must go too. With the select link removed AND CommandLineTools moved,
    # `xcode-select -p` does not fail -- it falls through to whatever Xcode bundle the
    # runner image ships (observed: /Applications/Xcode_16.4.app/Contents/Developer),
    # which re-arms /usr/bin/git and /usr/bin/cc and silently un-cleans the machine.
    # A rename is instant regardless of bundle size: same filesystem, no copy.
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
    # Wrapper logs the call then execs the REAL binary, so behaviour is unchanged
    # and the trace answers "did the installer reach for this?" honestly.
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
