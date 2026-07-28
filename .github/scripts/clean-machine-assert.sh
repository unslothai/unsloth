#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assert the clean-machine contract after an install attempt.
#
#   absent   The toolchain really was absent for the whole run. Guards against a
#            leg that "passed" only because masking silently failed, or because
#            the installer quietly installed Xcode CLT behind our back.
#   notools  The trace recorded no compiler/git/brew invocation (trace mode).
#   nobuild  The install log shows no source build (no sdist, no cmake, no
#            "Building wheel"). This is the wheels-only contract.
#
# Usage: bash .github/scripts/clean-machine-assert.sh absent notools nobuild
set -uo pipefail

LOG="${INSTALL_LOG:-logs/install.log}"
TRACE="${UNSLOTH_TOOL_TRACE:-}"
rc=0

fail() { echo "::error::$*"; rc=1; }
ok()   { echo "[assert] OK  $*"; }

for check in "$@"; do
  case "$check" in

    absent)
      # Deliberately NOT a `command -v` check. On a real virgin Mac /usr/bin/git and
      # /usr/bin/cc EXIST as Xcode CLT stubs, so `command -v git` SUCCEEDS -- running
      # it is what fails ("xcrun: error: invalid active developer path"). Asserting on
      # `command -v` would therefore be unfaithful and would fail on a correctly masked
      # runner. The honest invariant is: the tool must not WORK.
      if xcode-select -p >/dev/null 2>&1; then
        fail "xcode-select -p still resolves to $(xcode-select -p 2>/dev/null); not a clean Mac"
      else
        ok "xcode-select -p fails (the gate a virgin Mac hits)"
      fi
      for tool in git cc clang cmake; do
        command -v "$tool" >/dev/null 2>&1 || { ok "$tool not on PATH"; continue; }
        if "$tool" --version >/dev/null 2>&1; then
          fail "toolchain still usable: '$tool --version' succeeded ($(command -v "$tool")); masking failed"
        else
          ok "$tool present but non-functional (CLT stub), as on a clean Mac"
        fi
      done
      # brew is a plain binary with no stub, so absence from PATH is the right test.
      if command -v brew >/dev/null 2>&1; then
        fail "Homebrew still on PATH at $(command -v brew); masking failed"
      else
        ok "brew absent"
      fi
      ;;

    notools)
      if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
        fail "notools requested but no trace file (\$UNSLOTH_TOOL_TRACE=$TRACE)"
      else
        # git is legitimate under --local (it installs unsloth-zoo from a git URL);
        # UNSLOTH_ALLOW_TOOLS lets that leg allow-list it explicitly.
        allow="${UNSLOTH_ALLOW_TOOLS:-}"
        hits=""
        while IFS=$'\t' read -r tool rest; do
          [ -n "$tool" ] || continue
          case " $allow " in *" $tool "*) continue ;; esac
          # `xcode-select -p` ASKS whether a toolchain is selected; it cannot build
          # anything. The installer has to ask in order to tell the user whether a
          # source build is available, and the whole point of the fix is that it then
          # carries on without one. Treating the question as toolchain USE would fail
          # the very leg that proves the toolchain was never used. `--install`, which
          # pops the CLT installer, stays a hit.
          if [ "$tool" = "xcode-select" ]; then
            case "$rest" in
              -p|--print-path|-v|--version|"") continue ;;
            esac
          fi
          hits="$hits $tool"
        done < "$TRACE"
        if [ -n "$hits" ]; then
          fail "installer invoked toolchain:$(echo "$hits" | tr ' ' '\n' | sort -u | tr '\n' ' ')"
          echo "---- tool trace ----"; sort -u "$TRACE" | head -50
        else
          ok "no compiler/git/brew invocation recorded"
        fi
      fi
      ;;

    nobuild)
      # "Built an sdist" is NOT the same as "needed a compiler". Four packages on the
      # macOS path are sdist-only PURE PYTHON projects that build fine with no
      # toolchain (verified by resolving each against cp313/macos-arm64):
      #   openai-whisper, argbind, randomname  -- no version ever ships a wheel
      #   antlr4-python3-runtime==4.9.3        -- pinned below the 4.13.2 wheel
      # Failing on those would be a false alarm, so the contract asserted here is
      # "nothing that needs a COMPILER was built", with that allowlist subtracted.
      # UNSLOTH_ALLOW_SDIST can extend it.
      _allow="openai-whisper argbind randomname antlr4-python3-runtime ${UNSLOTH_ALLOW_SDIST:-}"
      if [ ! -f "$LOG" ]; then
        fail "nobuild requested but $LOG is missing"
      else
        _built="$(grep -oiE "building wheel for [a-z0-9._-]+" "$LOG" 2>/dev/null \
                  | sed -E 's/.* for //' | tr 'A-Z' 'a-z' | sort -u || true)"
        _bad=""
        for pkg in $_built; do
          case " $_allow " in *" $pkg "*) continue ;; esac
          _bad="$_bad $pkg"
        done
        if [ -n "$_bad" ]; then
          fail "built from source:$_bad -- these must resolve to wheels on a clean machine"
        else
          [ -n "$_built" ] && say_built="$(echo "$_built" | tr '\n' ' ')" || say_built="none"
          ok "no non-allowlisted source build (built: $say_built)"
        fi
        # Independent of package names: a compiler error means a toolchain was needed.
        if grep -qiE "error: command '(cc|gcc|clang|cl)' failed|no such file or directory: 'cc'|clang: error|cargo: not found|error: linker \`cc\` not found" "$LOG"; then
          fail "compiler invocation appears in the install log"
          grep -iE "error: command '(cc|gcc|clang|cl)' failed|clang: error" "$LOG" | head -10
        fi
      fi
      ;;

    *)
      fail "unknown check '$check'"
      ;;
  esac
done

exit "$rc"
