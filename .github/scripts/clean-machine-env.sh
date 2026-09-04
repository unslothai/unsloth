#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Simulate a virgin developer machine on a GitHub-hosted runner. Two modes, because
# "the tool is absent" and "the installer never called the tool" need different
# mechanisms:
#
#   mask   Make the toolchain genuinely ABSENT: scrub PATH to OS defaults and (with
#          --remove) move the real toolchain aside so `command -v git` correctly
#          FAILS. Deliberately no general "poison shims": a failing shim is still FOUND
#          by `command -v`, which reports the tool as present, the opposite of clean.
#          macOS has one observation-only exception: install_name_tool gets a logging
#          sentinel because the installer must shadow Apple's dialog-producing shim and
#          never uses this command to decide whether a dependency is installed.
#   trace  Leave the toolchain working behind wrappers that log the call then exec
#          the real binary, answering whether the installer ever REACHES for a
#          compiler/git without changing behaviour.
#
# Writes shell exports to $CLEAN_ENV_FILE (default ./clean-machine.env) to `source`;
# nothing is exported globally, so other steps keep a normal environment.
#
# Usage:
#   bash .github/scripts/clean-machine-env.sh mask [--remove]
#   bash .github/scripts/clean-machine-env.sh trace
#   source ./clean-machine.env
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_NAME_TOOL_HELPER="$SCRIPT_DIR/clean-machine-install-name-tool.sh"

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
# cctools binaries are included because their /usr/bin shims can trigger the same
# developer-tools dialog. uv's exact optional install_name_tool self-ID patch is
# observed separately and narrowly allow-listed by clean-machine-assert.sh.
TOOLS="xcode-select xcrun clang clang++ cc c++ gcc g++ git cmake make brew ninja cargo rustc
install_name_tool lipo otool objdump vtool strip nm"

note() { echo "[clean-machine] $*"; }

# Move a path aside and record the reverse in restore.sh. PATH scrubbing only HIDES
# these -- uv, the py launcher and framework lookups find them anyway -- so absence has
# to be real. The restore line is guarded: the install may have recreated the path, and
# an unguarded `mv` would bury the original inside it.
mask_aside() {
  local src="$1" dst="${2:-$1.masked}" as=""
  [ -e "$src" ] || return 0
  [ -w "$(dirname "$src")" ] || as="sudo"
  if $as mv "$src" "$dst" 2>/dev/null; then
    note "moved $src aside"
    printf "[ -e '%s' ] || %s mv '%s' '%s' 2>/dev/null || true\n" "$src" "$as" "$dst" "$src" >> "$RESTORE"
  else
    note "WARN could not move $src"
  fi
}

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
  if [ "$OS" = "Darwin" ]; then
    # Do not execute /usr/bin/install_name_tool as a self-test on a CLT-free Mac: that
    # is the GUI prompt this lane exists to prevent. This sentinel is ahead of /usr/bin,
    # logs argv with explicit argc and hex-encoded argument boundaries, and fails without
    # touching a dylib. install.sh's still-more-local uv guard must win over it.
    bash "$INSTALL_NAME_TOOL_HELPER" write sentinel "$BIN/install_name_tool"
    NEWPATH="$BIN:$NEWPATH"
  fi
  {
    echo "export PATH='$NEWPATH'"
    # UNSET, not a fake path: `xcode-select -p` honours DEVELOPER_DIR and prints it
    # verbatim with exit 0, so a nonexistent dir makes the probe SUCCEED. On a clean
    # Mac it is unset and the missing xcode_select_link is what makes the probe fail.
    echo "unset DEVELOPER_DIR || true"
    echo "unset SDKROOT CC CXX CFLAGS CXXFLAGS LDFLAGS CMAKE_GENERATOR CMAKE_PREFIX_PATH || true"
    echo "export HOMEBREW_NO_AUTO_UPDATE=1"
    echo "export UNSLOTH_CLEAN_MACHINE=1"

    echo "export UNSLOTH_TOOL_TRACE='$TRACE'"
  } >> "$ENV_FILE"

  if [ "$REMOVE" = "1" ] && [ "$OS" = "Darwin" ]; then
    # Best effort, each step independent and recorded in restore.sh so an `if: always()`
    # step can put the runner back. `xcode-select -p` reads xcode_select_link, so
    # removing it reproduces a virgin Mac's gate; `--reset` can reselect Xcode.app.
    # Captured now, re-selected LAST: restore.sh runs in order, and a --switch emitted here
    # would name a directory the later lines have not moved back yet, fail, and be
    # swallowed, leaving the link unrestored while the step reported success.
    _orig_dev=""
    if [ -e /var/db/xcode_select_link ]; then
      _orig_dev="$(xcode-select -p 2>/dev/null || true)"
      if sudo rm -f /var/db/xcode_select_link 2>/dev/null; then
        note "removed /var/db/xcode_select_link (was: ${_orig_dev:-unset})"
      else
        note "WARN could not remove /var/db/xcode_select_link"
        _orig_dev=""
      fi
    fi
    # Moving the CLT dir aside turns /usr/bin/{cc,clang,git} into dead shims, proving
    # the install needs no compiler at all.
    if [ -d /Library/Developer/CommandLineTools ]; then
      if sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools.masked 2>/dev/null; then
        note "moved CommandLineTools aside"
        echo "sudo mv /Library/Developer/CommandLineTools.masked /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move CommandLineTools"
      fi
    fi
    # Xcode.app too: with the link removed AND CommandLineTools moved, `xcode-select -p`
    # still succeeds via the image's Xcode bundle (observed:
    # /Applications/Xcode_16.4.app/Contents/Developer), which re-arms /usr/bin/{git,cc}.
    for app in /Applications/Xcode*.app; do
      [ -d "$app" ] || continue
      if sudo mv "$app" "${app}.masked" 2>/dev/null; then
        note "moved $(basename "$app") aside"
        echo "sudo mv '${app}.masked' '$app' 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move $app"
      fi
    done
    # After both directory restores above, so the path it names is back. Still `|| true`:
    # the runner is ephemeral and a failed re-selection must not fail an otherwise green
    # job, but it can no longer fail for the trivial reason of running too early.
    if [ -n "$_orig_dev" ]; then
      echo "sudo xcode-select --switch '$_orig_dev' 2>/dev/null || true" >> "$RESTORE"
    fi
    # /usr/local EXISTS on a factory-fresh Mac (a SIP-exempt firmlink) but is empty, so
    # empty it rather than remove it. Before the Homebrew block, so /usr/local/Homebrew
    # is stashed once, with one restore line, in the right order.
    if [ -d /usr/local ]; then
      STASH="$WORK/usr-local"
      mkdir -p "$STASH"
      for entry in /usr/local/* /usr/local/.[!.]*; do
        [ -e "$entry" ] || continue
        base="$(basename "$entry")"
        if sudo mv "$entry" "$STASH/$base" 2>/dev/null; then
          note "emptied /usr/local/$base"
          printf "[ -e '/usr/local/%s' ] || sudo mv '%s/%s' '/usr/local/%s' 2>/dev/null || true\n" \
            "$base" "$STASH" "$base" "$base" >> "$RESTORE"
        else
          note "WARN could not move $entry"
        fi
      done
    fi
    # The hosted toolcache and the python.org framework are what a PATH scrub cannot
    # reach: uv discovers interpreters by probing well-known locations.
    mask_aside "${AGENT_TOOLSDIRECTORY:-$HOME/hostedtoolcache}"
    mask_aside /Library/Frameworks/Python.framework
    # A virgin $HOME has none of these, and a populated uv/pip cache can satisfy a
    # resolution that would fail on a user's machine.
    for d in .cargo .rustup .nvm .rbenv .pyenv .local .cache \
             Library/Caches/uv Library/Caches/pip Library/Caches/Homebrew; do
      mask_aside "$HOME/$d"
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

  if [ "$REMOVE" = "1" ] && [ "$OS" = "Linux" ]; then
    # A hosted Linux runner keeps git, gcc, cmake and make in /usr/bin, which the PATH
    # scrub must keep, so move the resolved binaries aside (recorded in restore.sh).
    # Versioned siblings like gcc-11 survive; a consumer install invokes the unsuffixed
    # names, which is what `absent` checks.
    for tool in $TOOLS; do
      # Repeated: the same name can sit in /usr/bin and /usr/local/bin, and moving only
      # the first leaves the second on PATH.
      for _ in 1 2 3 4; do
        real="$(command -v "$tool" 2>/dev/null || true)"
        [ -n "$real" ] && [ -e "$real" ] || break
        if sudo mv "$real" "$real.masked" 2>/dev/null; then
          note "moved $real aside"
          echo "sudo mv '$real.masked' '$real' 2>/dev/null || true" >> "$RESTORE"
        else
          note "WARN could not move $real"
          break
        fi
      done
    done
  fi
fi

# ── trace ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "trace" ]; then
  for tool in $TOOLS; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [ -n "$real" ] || continue
    # Logs then execs the REAL binary, so behaviour is unchanged and the trace answers
    # "did the installer reach for this?" honestly. install_name_tool needs preserved
    # argument boundaries via hex so the assertion can require exact -id PATH PATH argv.
    if [ "$tool" = "install_name_tool" ]; then
      bash "$INSTALL_NAME_TOOL_HELPER" write passthrough "$BIN/$tool" "$real"
    else
      cat > "$BIN/$tool" <<WRAP
#!/bin/sh
printf '%s\t%s\n' "$tool" "\$*" >> "$TRACE"
exec "$real" "\$@"
WRAP
    fi
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
