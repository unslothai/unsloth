#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assert the clean-machine contract after an install attempt.
#
#   absent   The toolchain really was absent for the whole run. Catches a leg that
#            "passed" only because masking silently failed, or because the installer
#            quietly installed Xcode CLT behind our back.
#   notools  The trace recorded no compiler/git/brew invocation (trace mode).
#   nobuild  The wheels-only contract: no "Building wheel" from pip, no
#            "Building <pkg>==<ver>" from uv. Needs UNSLOTH_VERBOSE=1, or
#            run_install_cmd (install.sh:193-243) discards uv's output on success
#            and there is nothing to read.
#   macho    Every Mach-O under $MACHO_ROOT is the host architecture and is signed.
#            Closes the Rosetta 2 gap, the one divergence masking cannot reproduce.
#
# Usage: bash .github/scripts/clean-machine-assert.sh absent notools nobuild macho
set -uo pipefail

LOG="${INSTALL_LOG:-logs/install.log}"
TRACE="${UNSLOTH_TOOL_TRACE:-}"
rc=0

fail() { echo "::error::$*"; rc=1; }
ok()   { echo "[assert] OK  $*"; }

for check in "$@"; do
  case "$check" in

    absent)
      # Deliberately NOT `command -v`: on a virgin Mac /usr/bin/{git,cc} EXIST as CLT
      # stubs, so `command -v` succeeds and only RUNNING them fails ("invalid active
      # developer path"). The honest invariant is: must not WORK.
      if xcode-select -p >/dev/null 2>&1; then
        fail "xcode-select -p still resolves to $(xcode-select -p 2>/dev/null); not a clean Mac"
      else
        ok "xcode-select -p fails (the gate a virgin Mac hits)"
      fi
      for tool in git cc clang cmake; do
        command -v "$tool" >/dev/null 2>&1 || { ok "$tool not on PATH"; continue; }
        if "$tool" --version >/dev/null 2>&1; then
          # On Intel runners /usr/bin/git is not CLT-provided and survives their
          # removal, so no masking can take it away. cc and clang do become stubs and
          # the macOS consumer path needs no git, so report rather than fail.
          case " ${UNSLOTH_CLEAN_ALLOW_WORKING:-} " in
            *" $tool "*)
              echo "[assert] NOTE $tool still works ($(command -v "$tool")); allowed on this runner"
              continue
              ;;
          esac
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
        # git is legitimate under --local (unsloth-zoo comes from a git URL), so that
        # leg allow-lists it via UNSLOTH_ALLOW_TOOLS.
        allow="${UNSLOTH_ALLOW_TOOLS:-}"
        hits=""
        while IFS=$'\t' read -r tool rest; do
          [ -n "$tool" ] || continue
          case " $allow " in *" $tool "*) continue ;; esac
          # `xcode-select -p` only ASKS whether a toolchain is selected; the installer
          # has to ask, and the fix is that it carries on without one. Counting the
          # question as USE would fail the very leg proving the toolchain went
          # untouched. `--install`, which pops the CLT installer, stays a hit.
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
      # "Built an sdist" is NOT "needed a compiler", so the contract is "nothing
      # needing a COMPILER was built". Every name below was checked against its
      # actual sdist: setuptools.build_meta backend, no ext_modules, not one
      # .c/.cpp/.pyx/.rs file, so its PEP 517 build is a pure-Python copy step.
      #   openai-whisper, argbind, randomname  -- no version ever ships a wheel
      #   antlr4-python3-runtime==4.9.3        -- pinned below the 4.13.2 wheel
      #   triton-kernels  -- requirements/triton-kernels.txt pins a git URL under
      #     the triton repo's python/triton_kernels subdirectory: 75 Python files,
      #     a four-line pyproject.toml, no setup.py, kernels compiled at runtime.
      #     A direct URL the installer names itself, not something resolution
      #     chose, and only the Linux legs reach it (install_python_stack.py skips
      #     the step on Windows and macOS).
      # UNSLOTH_ALLOW_SDIST extends the allowlist.
      #
      # Lowercased and underscore-folded on both sides: a distribution name and the
      # name uv prints can disagree on the separator (requirement triton_kernels vs
      # build line triton-kernels), and a one-spelling allowlist silently misses.
      _allow="$(printf '%s' "openai-whisper argbind randomname antlr4-python3-runtime triton-kernels ${UNSLOTH_ALLOW_SDIST:-}" | tr 'A-Z_' 'a-z-')"
      if [ ! -f "$LOG" ]; then
        fail "nobuild requested but $LOG is missing"
      else
        # uv does NOT use pip's phrasing: it prints `Building <name>==<version>` to
        # stderr (astral-sh/uv#11165), so the pip-only pattern left _built empty on
        # every uv source build. Match both. Requiring `==` or ` @ ` after the name
        # keeps this off the installer's own lowercase "building frontend..."
        # progress text. Strip ANSI first so a coloured run (FORCE_COLOR) parses.
        #
        # `Building <name> @ file://...` is dropped first: a local-path build is
        # something the caller pointed at (install.sh --local, or the
        # UNSLOTH_CI_SOURCE_OVERLAY editable overlay), never a dependency resolution
        # chose. Index dependencies always print `<name>==<version>`, so no signal is
        # lost: a genuine sdist from PyPI is still caught, including one named unsloth.
        _esc=$(printf '\033')
        _built="$(sed -E "s/${_esc}\[[0-9;]*[A-Za-z]//g" "$LOG" 2>/dev/null \
                  | grep -viE "building [a-z0-9._-]+ @ file://" \
                  | grep -oiE "building wheel for [a-z0-9._-]+|building [a-z0-9._-]+(==| @ )" \
                  | tr 'A-Z' 'a-z' \
                  | sed -E -e 's/^building wheel for //' -e 's/^building //' -e 's/(==| @ )$//' \
                  | tr '_' '-' \
                  | sort -u || true)"
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

    macho)
      # The one thing masking cannot reproduce: Rosetta 2 is preinstalled on hosted
      # runners and absent from a factory-fresh Mac, so an x86_64-only payload runs
      # green here and dies with "bad CPU type in executable" for the user. Assert the
      # architecture rather than hope the runner lacks Rosetta.
      # `lipo` is an xcrun shim and is gone after masking, so read `file -b`, exactly
      # as the desktop lane does. Keyed off `uname -m`, since macos-15-intel is x86_64.
      root="${MACHO_ROOT:-${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth}}"
      want="$(uname -m)"
      [ "$want" = "aarch64" ] && want=arm64
      if [ ! -d "$root" ]; then
        fail "macho requested but $root does not exist"
      else
        n=0 bad_arch="" unsigned=""
        while IFS= read -r f; do
          desc="$(file -b "$f" 2>/dev/null || true)"
          case "$desc" in *Mach-O*) ;; *) continue ;; esac
          n=$((n + 1))
          # Substring, not equality: a universal binary lists every slice it carries,
          # and one that includes the host arch is fine.
          case "$desc" in
            *"$want"*) ;;
            *) bad_arch="$bad_arch $f [$desc]" ;;
          esac
          # arm64 only: AMFI SIGKILLs unsigned code there ("Killed: 9"), while x86_64
          # loads it happily, so an unsigned x86_64 payload is not the same defect.
          # Ad-hoc is enough, which is what the linker emits by default.
          if [ "$want" = "arm64" ] && ! codesign -v "$f" >/dev/null 2>&1; then
            unsigned="$unsigned $f"
          fi
        done < <(find "$root" -type f \( -perm -u+x -o -name '*.dylib' -o -name '*.so' -o -name '*.node' \) 2>/dev/null)
        if [ "$n" = "0" ]; then
          # An empty scan reads exactly like a clean one, so the check would pass on a
          # wrong root and prove nothing.
          fail "no Mach-O found under $root; the arch/signature assertion proved nothing"
        elif [ -n "$bad_arch" ]; then
          fail "Mach-O is not $want, so it runs here only under Rosetta 2, which a fresh Mac does not have:$bad_arch"
        elif [ -n "$unsigned" ]; then
          fail "unsigned Mach-O, which AMFI kills on arm64:$unsigned"
        else
          ok "$n Mach-O files under $root are $want$([ "$want" = arm64 ] && echo ' and signed')"
        fi
      fi
      ;;

    *)
      fail "unknown check '$check'"
      ;;
  esac
done

exit "$rc"
