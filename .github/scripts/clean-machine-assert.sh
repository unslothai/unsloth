#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Assert the clean-machine contract after an install attempt.
#
#   absent   The toolchain really was absent for the whole run. Catches a leg that
#            "passed" because masking silently failed, or because the installer
#            quietly installed Xcode CLT behind our back.
#   notools       The trace recorded no compiler/git/brew invocation (trace mode),
#                 except uv's exact optional libpython self-ID operation.
#   nodylibtool   No install_name_tool invocation escaped the CLT-absent guard.
#   dylibpatch    A CLT-present control observed only exact libpython self-ID patches.
#   nobuild  Wheels-only: no "Building wheel" from pip, no "Building <pkg>==<ver>"
#            from uv. Needs UNSLOTH_VERBOSE=1, or run_install_cmd
#            (install.sh:193-243) discards uv's output on success.
#   macho    Every Mach-O under $MACHO_ROOT is the host architecture, and every
#            Mach-O MAIN EXECUTABLE is signed. Closes the Rosetta 2 gap, the one
#            divergence masking cannot reproduce.
#
# Usage: bash .github/scripts/clean-machine-assert.sh absent nodylibtool notools dylibpatch nobuild macho
set -uo pipefail

LOG="${INSTALL_LOG:-logs/install.log}"
TRACE="${UNSLOTH_TOOL_TRACE:-}"
rc=0

fail() { echo "::error::$*"; rc=1; }
ok()   { echo "[assert] OK  $*"; }

_decode_trace_arg() { # encoded, destination variable
  _encoded=$1
  case "$_encoded" in h*) _hex=${_encoded#h} ;; *) return 1 ;; esac
  case "$_hex" in *[!0123456789abcdef]* ) return 1 ;; esac
  [ $(( ${#_hex} % 2 )) -eq 0 ] || return 1
  _decoded=""
  while [ -n "$_hex" ]; do
    _rest=${_hex#??}
    _pair=${_hex%"$_rest"}
    _hex=$_rest
    printf -v _byte '%b' "\\x$_pair"
    _decoded+=$_byte
  done
  printf -v "$2" '%s' "$_decoded"
}

_is_uv_libpython_self_id_patch() { # argc, operation, source, destination, extra
  [ "$1" = "3" ] && [ "$2" = "-id" ] && [ -n "$3" ] && [ "$3" = "$4" ] \
    && [ -z "$5" ] || return 1
  _patch_name=${3##*/}
  case "$_patch_name" in libpython*.dylib) ;; *) return 1 ;; esac
  _patch_dir=${3%/*}

  if [ -n "${UV_PYTHON_INSTALL_DIR:-}" ]; then
    _patch_root=${UV_PYTHON_INSTALL_DIR%/}
  else
    # Default uv data locations end in uv/python; this fallback keeps the assertion
    # useful outside CI, where UV_PYTHON_INSTALL_DIR is normally unset.
    case "$3" in */uv/python/*) ;; *) return 1 ;; esac
    _patch_root=${3%%/uv/python/*}/uv/python
  fi

  # Resolve both directories physically before comparing them. A lexical shell glob
  # would accept "$root/x/../../outside/..." (and symlink escapes) because * spans '/'.
  _patch_root=$(CDPATH= cd "$_patch_root" 2>/dev/null && pwd -P) || return 1
  _patch_dir=$(CDPATH= cd "$_patch_dir" 2>/dev/null && pwd -P) || return 1
  case "$_patch_dir" in "$_patch_root"/*/lib) ;; *) return 1 ;; esac
  _patch_install=${_patch_dir#"$_patch_root"/}
  _patch_install=${_patch_install%/lib}
  [ -n "$_patch_install" ] || return 1
  case "$_patch_install" in */*) return 1 ;; esac
  return 0
}

for check in "$@"; do
  case "$check" in

    absent)
      # NOT `command -v`: on a virgin Mac /usr/bin/{git,cc} EXIST as CLT stubs, so it
      # succeeds and only RUNNING them fails. The invariant is: must not WORK.
      if xcode-select -p >/dev/null 2>&1; then
        fail "xcode-select -p still resolves to $(xcode-select -p 2>/dev/null); not a clean Mac"
      else
        ok "xcode-select -p fails (the gate a virgin Mac hits)"
      fi
      # The whole set clean-machine-env.sh moves aside, not the four it used to check: that
      # helper warns and carries on when a move fails, so a surviving gcc -- which
      # install.sh probes to decide build-essential is available -- passed unnoticed.
      for tool in git cc clang cmake gcc g++ make ninja cargo rustc; do
        command -v "$tool" >/dev/null 2>&1 || { ok "$tool not on PATH"; continue; }
        if "$tool" --version >/dev/null 2>&1; then
          # Intel runners' /usr/bin/git is not CLT-provided, so masking cannot take it
          # away; cc/clang do become stubs and macOS needs no git, so report, not fail.
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

    nodylibtool)
      if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
        fail "nodylibtool requested but no trace file (\$UNSLOTH_TOOL_TRACE=$TRACE)"
      else
        _dylib_hits=0
        while IFS=$'\t' read -r tool _rest; do
          [ "$tool" = "install_name_tool" ] && _dylib_hits=$((_dylib_hits + 1))
        done < "$TRACE"
        if [ "$_dylib_hits" -ne 0 ]; then
          fail "install_name_tool escaped the CLT-absent uv guard ($_dylib_hits invocation(s))"
          grep '^install_name_tool[[:space:]]' "$TRACE" | head -20 || true
        else
          ok "install_name_tool was never reached on the CLT-absent path"
        fi
      fi
      ;;

    dylibpatch)
      if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
        fail "dylibpatch requested but no trace file (\$UNSLOTH_TOOL_TRACE=$TRACE)"
      else
        _dylib_hits=0
        _dylib_bad=0
        while IFS=$'\t' read -r tool argc operation_encoded source_encoded destination_encoded extra; do
          [ "$tool" = "install_name_tool" ] || continue
          _dylib_hits=$((_dylib_hits + 1))
          operation=""; source=""; destination=""
          if ! _decode_trace_arg "$operation_encoded" operation \
             || ! _decode_trace_arg "$source_encoded" source \
             || ! _decode_trace_arg "$destination_encoded" destination \
             || ! _is_uv_libpython_self_id_patch "$argc" "$operation" "$source" "$destination" "$extra"; then
            _dylib_bad=$((_dylib_bad + 1))
            echo "::error::invalid install_name_tool trace record: $tool argc=$argc"
          fi
        done < "$TRACE"
        if [ "$_dylib_hits" -eq 0 ]; then
          fail "CLT-present control recorded no install_name_tool patch; managed Python may have been reused"
        elif [ "$_dylib_bad" -ne 0 ]; then
          fail "$_dylib_bad of $_dylib_hits install_name_tool invocation(s) were not exact libpython self-ID patches"
        else
          ok "all $_dylib_hits install_name_tool invocation(s) were exact libpython self-ID patches"
        fi
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
        while IFS=$'\t' read -r tool argc_or_rest arg1 arg2 arg3 extra; do
          [ -n "$tool" ] || continue
          # This optional uv operation is the only permitted developer-tool use. Keep it
          # structural rather than name-only: arbitrary install_name_tool calls still fail.
          if [ "$tool" = "install_name_tool" ]; then
            operation=""; source=""; destination=""
            if _decode_trace_arg "$arg1" operation \
               && _decode_trace_arg "$arg2" source \
               && _decode_trace_arg "$arg3" destination \
               && _is_uv_libpython_self_id_patch "$argc_or_rest" "$operation" "$source" "$destination" "$extra"; then
              continue
            fi
            hits="$hits $tool"
            continue
          fi
          case " $allow " in *" $tool "*) continue ;; esac
          # `xcode-select -p` only ASKS whether a toolchain is selected and the fix is
          # carrying on without one, so it is not USE. `--install` stays a hit.
          if [ "$tool" = "xcode-select" ]; then
            case "$argc_or_rest" in
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
      # "Built an sdist" is NOT "needed a compiler". Every name was verified against its
      # own sdist: setuptools.build_meta, no ext_modules, no .c/.cpp/.pyx/.rs, so the
      # PEP 517 build is a pure-Python copy step.
      #   openai-whisper, argbind, randomname  -- no version ever ships a wheel
      #   antlr4-python3-runtime==4.9.3        -- pinned below the 4.13.2 wheel
      #   triton-kernels  -- a git URL under the triton repo's python/triton_kernels
      #     subdir: 75 Python files, no setup.py, kernels compiled at runtime. Named by
      #     the installer, not chosen by resolution, and only the Linux legs reach it.
      #   diffusers  -- pinned to a source archive because MiniMax-H3 support is not in
      #     any release yet. Plain setuptools, no ext_modules, and the tree has zero
      #     .c/.cpp/.pyx/.rs/.cu files, so the PEP 517 build is a pure-Python copy.
      #     REMOVE this entry once a diffusers release carries H3 and the requirement
      #     goes back to a version specifier.
      # UNSLOTH_ALLOW_SDIST extends it. Lowercased and underscore-folded on both sides:
      # the distribution name and the name uv prints can differ on the separator.
      _allow="$(printf '%s' "openai-whisper argbind randomname antlr4-python3-runtime triton-kernels diffusers ${UNSLOTH_ALLOW_SDIST:-}" | tr 'A-Z_' 'a-z-')"
      if [ ! -f "$LOG" ]; then
        fail "nobuild requested but $LOG is missing"
      else
        # uv prints `Building <name>==<ver>`, pip `Building wheel for <name>`
        # (astral-sh/uv#11165), so match both; the `==` / ` @ ` keeps this off the
        # installer's own "building frontend..." text, and ANSI is stripped so a
        # coloured run parses. `Building <name> @ file://` is dropped -- a local-path
        # build is one the caller pointed at (--local, the overlay), never one
        # resolution chose, while index deps always print `<name>==<ver>`, so a genuine
        # PyPI sdist is still caught, including one named unsloth.
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
      # The one thing masking cannot reproduce: Rosetta 2 ships on hosted runners and
      # not on a factory-fresh Mac, so an x86_64-only payload runs green here and dies
      # with "bad CPU type in executable" for the user. `lipo` is an xcrun shim and gone
      # after masking, so read `file -Lb`, keyed off `uname -m` (macos-15-intel is x86_64).
      #
      # SCOPE: all of $MACHO_ROOT, .venv_t5_510/_530/_550 sidecars included -- payload,
      # not scratch (setup.sh:579-581 creates them, transformers_version.py:338-348 puts
      # them on sys.path). Any exclusion must be a named path rule, never a narrowed find.
      root="${MACHO_ROOT:-${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth}}"
      want="$(uname -m)"
      [ "$want" = "aarch64" ] && want=arm64
      if [ ! -d "$root" ]; then
        fail "macho requested but $root does not exist"
      else
        # SCOPE, part 2: the two payloads the install RUNS ON live outside $root. `uv
        # venv` links <venv>/bin/python at its base interpreter and the find below has no
        # -L, so the interpreter that ran every step is invisible to it; the uv that
        # fetched it lands in $HOME/.local/bin. Both are what Rosetta 2 hides: an x86_64
        # one runs green here and dies on the factory-fresh Mac this stands in for.
        # -L follows that symlink; -maxdepth keeps this a bin/ lookup, not a second walk
        # of site-packages through the venv's lib64 link -- depth 4 covers
        # <root>/unsloth_studio, the .venv_t5_* sidecars and <root>/studio/unsloth_studio.
        # In a variable so it can be counted separately: uv alone would satisfy $nout.
        base_py="$(find -L "$root" -maxdepth 4 -type f -path '*/bin/python' 2>/dev/null)"
        _macho_targets() {
          find "$root" -type f \( -perm -u+x -o -name '*.dylib' -o -name '*.so' -o -name '*.node' \) 2>/dev/null
          [ -n "$base_py" ] && printf '%s\n' "$base_py"
          for _uv in "$HOME/.local/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
            [ -n "$_uv" ] && [ -f "$_uv" ] && printf '%s\n' "$_uv"
          done
        }
        n=0 nexe=0 nout=0 nbase=0 bad_arch="" unsigned="" broken=""
        while IFS= read -r f; do
          # -L: find printed the SYMLINK path for <venv>/bin/python, and plain `file` does
          # not dereference, so it answered "symbolic link to ..." and the Mach-O test
          # below dropped the very interpreter this scan exists to check.
          desc="$(file -Lb "$f" 2>/dev/null || true)"
          case "$desc" in *Mach-O*) ;; *) continue ;; esac
          n=$((n + 1))
          case "$f" in "$root"/*) ;; *) nout=$((nout + 1)) ;; esac
          # Classified, not merely found: an entry `file` could not read is invisible here.
          case "
$base_py
" in *"
$f
"*) nbase=$((nbase + 1)) ;; esac
          # Substring, not equality: a universal binary lists every slice it carries,
          # and one that includes the host arch is fine.
          case "$desc" in
            *"$want"*) ;;
            *) bad_arch="$bad_arch $f [$desc]" ;;
          esac

          # Signature: MAIN EXECUTABLES ONLY. Asserting it on every Mach-O failed the
          # mask/pipe leg on 29 ordinary PyPI extension modules plus libportaudio.dylib:
          # MH_BUNDLE/MH_DYLIB images dlopen'd without library validation ship unsigned,
          # and that run had already imported them with the installer exiting 0. macOS
          # enforces on main executables and gatekept .app bundles. Key off the filetype
          # `file` reports, not the path (a .so may be either); the library veto is second
          # so a mixed-type fat file counts as a library, and substring tests are
          # order-independent (Apple prints `executable arm64`, GNU `arm64 executable`).
          _is_exe=0
          case "$desc" in *executable*) _is_exe=1 ;; esac
          case "$desc" in *"shared library"*|*bundle*) _is_exe=0 ;; esac
          # Named rule so a failure says which path matched; the filetype test covers it.
          case "$f" in *.app/Contents/MacOS/*) _is_exe=1 ;; esac
          [ "$_is_exe" = 1 ] && nexe=$((nexe + 1))

          # arm64 only: the kernel refuses to exec an unsigned arm64 main binary
          # ("Killed: 9"); x86_64 execs it happily, so it is not the same defect.
          if [ "$want" = "arm64" ] && [ "$_is_exe" = 1 ]; then
            # Ad-hoc counts as signed (arm64 linkers seal ad-hoc by default): the test is
            # "has a verifying seal", not "has an identity", which spctl/--strict demand.
            if ! codesign -v "$f" >/dev/null 2>&1; then
              # Nothing to verify and a seal that does not match differ. Captured, not
              # piped: `codesign -dvv` exits non-zero on an unsigned file, and under
              # pipefail that would be the pipeline's status even on a match.
              _sig="$(codesign -dvv "$f" 2>&1 || true)"
              case "$_sig" in
                *"not signed at all"*) unsigned="$unsigned $f" ;;
                *)                     broken="$broken $f" ;;
              esac
            fi
          fi
        done < <(_macho_targets | sort -u)
        if [ "$n" = "0" ]; then
          # An empty scan reads exactly like a clean one, so a wrong root would pass.
          fail "no Mach-O found under $root; the arch/signature assertion proved nothing"
        elif [ "$nbase" = "0" ]; then
          fail "no */bin/python under $root was classified as Mach-O, so the venv's base interpreter went unchecked (found: ${base_py:-none})"
        elif [ "$nout" = "0" ]; then
          # install.sh always bootstraps uv into $HOME/.local/bin, so zero hits outside
          # $root means the extra scan matched nothing and uv's arch went unproven.
          fail "no Mach-O outside $root was scanned, so uv and the venv's base interpreter escaped the check"
        elif [ -n "$bad_arch" ]; then
          fail "Mach-O is not $want, so it runs here only under Rosetta 2, which a fresh Mac does not have:$bad_arch"
        elif [ -n "$unsigned" ]; then
          fail "unsigned Mach-O main executable, which arm64 macOS refuses to exec:$unsigned"
        elif [ -n "$broken" ]; then
          fail "Mach-O main executable carries a signature that does not verify:$broken"
        else
          ok "$n Mach-O files under $root, plus uv and the venv's base interpreter, are $want$([ "$want" = arm64 ] && echo "; all $nexe main executable(s) signed")"
        fi
      fi
      ;;

    *)
      fail "unknown check '$check'"
      ;;
  esac
done

exit "$rc"
