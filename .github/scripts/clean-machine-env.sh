#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Simulate a virgin developer machine on a hosted runner.
#   mask   Make the toolchain truly absent (scrubbed PATH, --remove moves it aside); no failing
#          shims, since `command -v` would still find them.
#   trace  Wrap tools to log each call then exec the real binary.
# Writes exports to $CLEAN_ENV_FILE (default ./clean-machine.env) for `source`.
# Usage: bash .github/scripts/clean-machine-env.sh mask [--remove] | trace
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
: > "$TRACE.git-cwd"
: > "$ENV_FILE"
printf '#!/usr/bin/env bash\n# Undo clean-machine-env.sh --remove. Safe to run twice.\nset -uo pipefail\n' > "$RESTORE"
chmod +x "$RESTORE"

# cctools are included because their /usr/bin shims can trigger the developer-tools dialog.
TOOLS="xcode-select xcrun clang clang++ cc c++ gcc g++ git cmake make brew ninja cargo rustc
install_name_tool lipo otool objdump vtool strip nm"

note() { echo "[clean-machine] $*"; }

# PATH scrubbing only hides tools (uv and framework lookups still find them), so move them aside.
# The restore line is guarded: the install may recreate the path, and a bare mv would bury the original in it.
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

# Keep only OS-default system dirs.
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

if [ "$MODE" = "mask" ]; then
  NEWPATH="$(scrub_path)"
  if [ "$OS" = "Darwin" ]; then
    # Never run /usr/bin/install_name_tool on a CLT-free Mac: it opens the GUI prompt.
    # install.sh's more local uv guard must win over this sentinel.
    bash "$INSTALL_NAME_TOOL_HELPER" write sentinel "$BIN/install_name_tool"
    NEWPATH="$BIN:$NEWPATH"
  fi
  {
    echo "export PATH='$NEWPATH'"
    # Unset, not faked: `xcode-select -p` prints DEVELOPER_DIR verbatim with exit 0 even if missing.
    echo "unset DEVELOPER_DIR || true"
    echo "unset SDKROOT CC CXX CFLAGS CXXFLAGS LDFLAGS CMAKE_GENERATOR CMAKE_PREFIX_PATH || true"
    echo "export HOMEBREW_NO_AUTO_UPDATE=1"
    echo "export UNSLOTH_CLEAN_MACHINE=1"

    echo "export UNSLOTH_TOOL_TRACE='$TRACE'"
  } >> "$ENV_FILE"

  if [ "$REMOVE" = "1" ] && [ "$OS" = "Darwin" ]; then
    # Original selection is re-selected LAST in restore.sh, after the directories it names are back.
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
    if [ -d /Library/Developer/CommandLineTools ]; then
      if sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools.masked 2>/dev/null; then
        note "moved CommandLineTools aside"
        echo "sudo mv /Library/Developer/CommandLineTools.masked /Library/Developer/CommandLineTools 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move CommandLineTools"
      fi
    fi
    # Xcode.app too, or `xcode-select -p` still succeeds via the image's Xcode bundle.
    for app in /Applications/Xcode*.app; do
      [ -d "$app" ] || continue
      if sudo mv "$app" "${app}.masked" 2>/dev/null; then
        note "moved $(basename "$app") aside"
        echo "sudo mv '${app}.masked' '$app' 2>/dev/null || true" >> "$RESTORE"
      else
        note "WARN could not move $app"
      fi
    done
    if [ -n "$_orig_dev" ]; then
      echo "sudo xcode-select --switch '$_orig_dev' 2>/dev/null || true" >> "$RESTORE"
    fi
    # /usr/local exists but is empty on a fresh Mac, so empty it rather than remove it.
    # Must run before the Homebrew block so /usr/local/Homebrew is stashed once.
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
    # uv probes well-known interpreter locations that a PATH scrub cannot hide.
    mask_aside "${AGENT_TOOLSDIRECTORY:-$HOME/hostedtoolcache}"
    mask_aside /Library/Frameworks/Python.framework
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
    # Versioned siblings like gcc-11 survive; installs invoke the unsuffixed names.
    for tool in $TOOLS; do
      # Repeated: the same name can exist in several PATH dirs.
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

if [ "$MODE" = "trace" ]; then
  for tool in $TOOLS; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [ -n "$real" ] || continue
    # install_name_tool argv is hex-encoded so the assertion can require exact argument boundaries.
    if [ "$tool" = "install_name_tool" ]; then
      bash "$INSTALL_NAME_TOOL_HELPER" write passthrough "$BIN/$tool" "$real"
    else
      # git records its cwd: notools must see that `submodule update` ran inside uv's checkout.
      cwd_line=""
      [ "$tool" = "git" ] && cwd_line="printf '%s\t%s\n' \"\$PWD\" \"\$*\" >> '$TRACE.git-cwd'"
      cat > "$BIN/$tool" <<WRAP
#!/bin/sh
printf '%s\t%s\n' "$tool" "\$*" >> "$TRACE"
$cwd_line
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
