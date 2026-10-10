#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Assert the clean-machine contract after an install attempt.
#   absent       the toolchain really was absent for the whole run
#   notools      the trace shows no compiler/git/brew use beyond the allowed uv operations
#   nodylibtool  no install_name_tool call escaped the CLT-absent guard
#   dylibpatch   a CLT-present control saw only exact libpython self-ID patches
#   nobuild      no source builds; needs UNSLOTH_VERBOSE=1 or uv output is discarded
#   macho        every Mach-O is the host arch and every main executable is signed
# Usage: bash .github/scripts/clean-machine-assert.sh absent nodylibtool notools dylibpatch nobuild macho
set -uo pipefail

LOG="${INSTALL_LOG:-logs/install.log}"
TRACE="${UNSLOTH_TOOL_TRACE:-}"
rc=0

fail() { echo "::error::$*"; rc=1; }
ok()   { echo "[assert] OK  $*"; }

_decode_trace_arg() {
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

# Allowed git+ remotes from the requirement files in $UNSLOTH_ALLOW_GIT_FROM.
_allowed_git_remotes() {
  for _req in ${UNSLOTH_ALLOW_GIT_FROM:-}; do
    [ -f "$_req" ] || { echo "::error::UNSLOTH_ALLOW_GIT_FROM names a missing file: $_req" >&2; continue; }
    # The revision follows the LAST @, so ssh://git@host/repo.git@SHA keeps its user.
    sed -n \
      -e 's/^[^#]*git+\([a-z][a-z0-9+.-]*:\/\/[^#[:space:]]*\)@[^@\/#[:space:]]*\([#[:space:]].*\)\{0,1\}$/\1/p' \
      -e 't' \
      -e 's/^[^#]*git+\([a-z][a-z0-9+.-]*:\/\/[^#[:space:]]*\).*/\1/p' "$_req"
  done | sed 's/\.git$//' | sort -u
}

_allowed_git_pins() {
  for _req in ${UNSLOTH_ALLOW_GIT_FROM:-}; do
    [ -f "$_req" ] || continue
    sed -n 's/^[^#]*git+[a-z][a-z0-9+.-]*:\/\/[^#[:space:]]*@\([0-9a-f]\{40\}\)\([#[:space:]].*\)\{0,1\}$/\1/p' "$_req"
  done | sort -u
}

_git_line_remotes() {
  printf '%s\n' "$1" | grep -oE '[a-z][a-z0-9+.-]*://[^[:space:]]+|[[:alnum:]_.-]+@[[:alnum:].-]+:[^[:space:]]+' \
    | sed 's/\.git$//'
}

# True only if every run of these args was inside one of uv's checkouts.
_ran_in_uv_checkout() {
  _ran_under "$1" "${UV_CACHE_DIR%/}/git-v0/checkouts/"
}

_ran_under() {
  [ -n "${UV_CACHE_DIR:-}" ] && [ -f "$TRACE.git-cwd" ] || return 1
  _seen=false
  while IFS=$'\t' read -r _cwd _args; do
    [ "$_args" = "$1" ] || continue
    _seen=true
    case "$_cwd" in "$2"?*) ;; *) return 1 ;; esac
    case "$_cwd" in *..*) return 1 ;; esac
  done < "$TRACE.git-cwd"
  [ "$_seen" = true ]
}

_in_checkout_of() {
  _ran_in_uv_checkout "$1" || return 1
  while IFS=$'\t' read -r _cwd _args; do
    [ "$_args" = "$1" ] || continue
    _short=${_cwd##*/}
    [ ${#_short} -ge 7 ] || return 1
    case "$_short" in *[!0-9a-f]*) return 1 ;; esac
    case "$2" in "$_short"*) ;; *) return 1 ;; esac
  done < "$TRACE.git-cwd"
}

# A remoteless git line is allowed only in uv's own cache-checkout shapes, with no network access.
_is_uv_git_cache_op() {
  _cache="${UV_CACHE_DIR%/}/git-v0"
  case "$1" in
    init|rev-parse|"rev-parse "*) _ran_under "$1" "$_cache/" ; return ;;
    "submodule update --recursive --init") _ran_in_uv_checkout "$1"; return ;;
    "reset --hard "*)
      _commit=${1#reset --hard }
      case "$_commit" in *[!0-9a-f]*|"") return 1 ;; esac
      [ ${#_commit} -eq 40 ] || return 1
      printf '%s\n' "$(_allowed_git_pins)" | grep -qxF -- "$_commit" || return 1
      _in_checkout_of "$1" "$_commit"
      return ;;
    "clone --local "*)
      [ -n "${UV_CACHE_DIR:-}" ] || return 1
      set -f; set -- $1; set +f
      [ $# -eq 4 ] || return 1
      case "$3" in "$_cache"/db/*) ;; *) return 1 ;; esac
      case "$4" in "$_cache"/checkouts/*) ;; *) return 1 ;; esac
      case "$3$4" in *..*) return 1 ;; esac
      return 0 ;;
  esac
  return 1
}

_is_allowed_git_remote() {
  [ -n "$1" ] && printf '%s\n' "$2" | grep -qxF -- "${1%.git}"
}

# A git line naming a remote must match uv's exact fetch or submodule argv shapes against
# an allowed remote, parsed as git would read it.
_is_allowed_remote_git_line() {
  _line=$1
  _allowed=$2
  set -f
  set -- $1
  set +f
  _origin=""
  if [ "${1:-}" = "-c" ]; then
    case "${2:-}" in remote.origin.url=*) _origin=${2#remote.origin.url=}; shift 2 ;; *) return 1 ;; esac
  fi
  case "${1:-}" in
    fetch)
      [ -z "$_origin" ] || return 1
      shift
      _repo=""
      for _word in "$@"; do
        if [ -z "$_repo" ]; then
          case "$_word" in
            --tags|--force|--update-head-ok|--no-tags|--quiet) continue ;;
            --depth=[0-9]*) case "${_word#--depth=}" in *[!0-9]*) return 1 ;; esac; continue ;;
            -*) return 1 ;;
          esac
          _repo=$_word
          continue
        fi
        case "$_word" in
          -*|*://*|*@*:*) return 1 ;;
          ?*:refs/*) ;;
          *) return 1 ;;
        esac
      done
      _is_allowed_git_remote "$_repo" "$_allowed"
      return ;;
    submodule)
      _is_allowed_git_remote "$_origin" "$_allowed" || return 1
      [ "${2:-}" = "update" ] || return 1
      shift 2
      for _word in "$@"; do
        case "$_word" in --init|--recursive) ;; *) return 1 ;; esac
      done
      _ran_in_uv_checkout "$_line"
      return ;;
  esac
  return 1
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
    case "$3" in */uv/python/*) ;; *) return 1 ;; esac
    _patch_root=${3%%/uv/python/*}/uv/python
  fi

  # Resolve physically: a lexical glob would accept ../ and symlink escapes.
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
      # NOT `command -v`: on a virgin Mac /usr/bin/{git,cc} exist as CLT stubs that fail when run.
      if xcode-select -p >/dev/null 2>&1; then
        fail "xcode-select -p still resolves to $(xcode-select -p 2>/dev/null); not a clean Mac"
      else
        ok "xcode-select -p fails (the gate a virgin Mac hits)"
      fi
      # Check the full set clean-machine-env.sh moves aside; it only warns on a failed move.
      for tool in git cc clang cmake gcc g++ make ninja cargo rustc; do
        command -v "$tool" >/dev/null 2>&1 || { ok "$tool not on PATH"; continue; }
        if "$tool" --version >/dev/null 2>&1; then
          # Intel's /usr/bin/git is not CLT-provided, so report it rather than fail.
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
        allow="${UNSLOTH_ALLOW_TOOLS:-}"
        # Git is allowed only for uv fetching the requirement files' pinned git+ remotes.
        allowed_remotes=$(_allowed_git_remotes)
        git_fetched_allowed=false
        if [ -n "$allowed_remotes" ]; then
          while IFS=$'\t' read -r tool rest; do
            [ "$tool" = "git" ] || continue
            case "$rest" in fetch\ *) ;; *) continue ;; esac
            if _is_allowed_remote_git_line "$rest" "$allowed_remotes"; then
              git_fetched_allowed=true
            fi
          done < "$TRACE"
        fi
        hits=""
        while IFS=$'\t' read -r tool argc_or_rest arg1 arg2 arg3 extra; do
          [ -n "$tool" ] || continue
          # The only permitted developer-tool use; keep it structural.
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
          if [ "$tool" = "git" ] && [ -n "$allowed_remotes" ]; then
            git_rest="$argc_or_rest"
            for _field in "$arg1" "$arg2" "$arg3" "$extra"; do
              [ -n "$_field" ] && git_rest="$git_rest	$_field"
            done
            if [ "$git_rest" = "--version" ]; then
              continue
            fi
            if [ -n "$(_git_line_remotes "$git_rest")" ]; then
              _is_allowed_remote_git_line "$git_rest" "$allowed_remotes" && continue
            elif [ "$git_fetched_allowed" = true ] && _is_uv_git_cache_op "$git_rest"; then
              continue
            fi
          fi
          # `xcode-select -p` only asks; `--install` is still a hit.
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
      # Pure-Python sdists (no compiled extensions); keep in sync with assert-nobuild.ps1.
      # Drop diffusers once a release has the wheel pin. UNSLOTH_ALLOW_SDIST extends the list.
      _allow="$(printf '%s' "openai-whisper argbind randomname antlr4-python3-runtime triton-kernels diffusers ${UNSLOTH_ALLOW_SDIST:-}" | tr 'A-Z_' 'a-z-')"
      if [ ! -f "$LOG" ]; then
        fail "nobuild requested but $LOG is missing"
      else
        # Match both pip and uv build lines; `==`/` @ ` skips frontend text. Local file:// builds are ignored.
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
        if grep -qiE "error: command '(cc|gcc|clang|cl)' failed|no such file or directory: 'cc'|clang: error|cargo: not found|error: linker \`cc\` not found" "$LOG"; then
          fail "compiler invocation appears in the install log"
          grep -iE "error: command '(cc|gcc|clang|cl)' failed|clang: error" "$LOG" | head -10
        fi
      fi
      ;;

    macho)
      # Rosetta 2 hides x86_64-only payloads on runners. Scan all of $MACHO_ROOT; any exclusion
      # must be a named path rule, never a narrowed find.
      root="${MACHO_ROOT:-${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth}}"
      want="$(uname -m)"
      [ "$want" = "aarch64" ] && want=arm64
      if [ ! -d "$root" ]; then
        fail "macho requested but $root does not exist"
      else
        # The venv's base interpreter and uv live outside $root, so check them separately.
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
          # -L: plain `file` reports the symlink, not the Mach-O.
          desc="$(file -Lb "$f" 2>/dev/null || true)"
          case "$desc" in *Mach-O*) ;; *) continue ;; esac
          n=$((n + 1))
          case "$f" in "$root"/*) ;; *) nout=$((nout + 1)) ;; esac
          case "
$base_py
" in *"
$f
"*) nbase=$((nbase + 1)) ;; esac
          # Substring: a universal binary lists every slice.
          case "$desc" in
            *"$want"*) ;;
            *) bad_arch="$bad_arch $f [$desc]" ;;
          esac

          # Signature check applies to main executables only; dylibs and bundles legitimately ship unsigned.
          _is_exe=0
          case "$desc" in *executable*) _is_exe=1 ;; esac
          case "$desc" in *"shared library"*|*bundle*) _is_exe=0 ;; esac
          case "$f" in *.app/Contents/MacOS/*) _is_exe=1 ;; esac
          [ "$_is_exe" = 1 ] && nexe=$((nexe + 1))

          # arm64 only: the kernel refuses unsigned arm64 main binaries, x86_64 does not.
          if [ "$want" = "arm64" ] && [ "$_is_exe" = 1 ]; then
            # Ad-hoc counts as signed.
            if ! codesign -v "$f" >/dev/null 2>&1; then
              # Captured, not piped: codesign -dvv exits non-zero on unsigned files, breaking pipefail.
              _sig="$(codesign -dvv "$f" 2>&1 || true)"
              case "$_sig" in
                *"not signed at all"*) unsigned="$unsigned $f" ;;
                *)                     broken="$broken $f" ;;
              esac
            fi
          fi
        done < <(_macho_targets | sort -u)
        if [ "$n" = "0" ]; then
          fail "no Mach-O found under $root; the arch/signature assertion proved nothing"
        elif [ "$nbase" = "0" ]; then
          fail "no */bin/python under $root was classified as Mach-O, so the venv's base interpreter went unchecked (found: ${base_py:-none})"
        elif [ "$nout" = "0" ]; then
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
