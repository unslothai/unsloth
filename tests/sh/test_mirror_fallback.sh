#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Mirror fallback in install.sh / studio/setup.sh: the env vars each probe outcome exports, under
# dash (install.sh's sh) and bash with setup.sh's strict flags, against a curl stub.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"

_block() { sed -n '/^# ── BEGIN mirror fallback/,/^# ── END mirror fallback ──$/p' "$1"; }

_WORK=$(mktemp -d)
trap 'rm -rf "$_WORK"' EXIT
_block "$INSTALL_SH" > "$_WORK/block.sh"
if [ "$(_block "$SETUP_SH")" = "$(cat "$_WORK/block.sh")" ]; then
    ok "setup.sh carries the same mirror fallback block as install.sh"
else
    bad "setup.sh mirror fallback block drifted from install.sh"
fi
# Inside a heredoc the block is written to a file instead of defining _mirror_fallback.
_heredoc_block_lines() { awk '
    tag != "" { if ($0 == tag || (dash && $0 ~ "^\t*" tag "$")) tag = ""; else if (/^[ \t]*# ── BEGIN mirror fallback/) print NR; next }
    !/^[ \t]*#/ && match($0, /(^|[^<])<<-?[ \t]*["\047]?[A-Za-z_][A-Za-z_0-9]*["\047]?([ \t|;&)>]|$)/) {
        t = substr($0, RSTART, RLENGTH); sub(/^[^<]*<</, "", t); dash = (t ~ /^-/)
        gsub(/^-?[ \t]*["\047]?|["\047]?[ \t|;&)>]*$/, "", t); tag = t }' "$1"; }
for _f in "$INSTALL_SH" "$SETUP_SH"; do
    assert_eq "${_f##*/} defines the mirror fallback outside every heredoc" "" "$(_heredoc_block_lines "$_f")"
done

# The stub answers a 1 MiB range probe from MOCK_<HOST>: fast, slow (<1 MiB/s), blocked or "<code> <bytes/s>"; "a|b" is a, then b.
mkdir -p "$_WORK/bin"
cat > "$_WORK/bin/curl" <<'EOF'
#!/bin/sh
for _a in "$@"; do _url="$_a"; done
echo "$_url" >> "$MOCK_LOG"
case "$_url" in
    https://tuna.mirrors.cernet.edu.cn/pypi/web/simple/*) _h=CERNETPYPIINDEX ;; https://tuna.mirrors.cernet.edu.cn/pytorch/whl/cpu/torch/) _h=CERNETTORCHINDEX ;;
    https://tuna.mirrors.cernet.edu.cn/pytorch/*) _h=CERNETTORCH ;; https://registry.npmmirror.com/-/binary/node/*) _h=NPMMIRRORNODE ;;
    https://tuna.mirrors.cernet.edu.cn/*) _h=CERNET ;; https://registry.npmmirror.com/*) _h=NPMMIRROR ;;
    https://pypi.org/*) _h=PYPIINDEX ;; https://download.pytorch.org/*) _h=TORCHINDEX ;;
    https://files.pythonhosted.org/*) _h=PYPI ;; https://download-r2.pytorch.org/*) _h=TORCH ;;
    https://nodejs.org/*) _h=NODE ;; https://registry.npmjs.org/*) _h=NPM ;; *) _h=ASTRAL ;;
esac
eval "_r=\${MOCK_$_h:-fast}"
if [ -f "$MOCK_LOG.$_h" ]; then _r=${_r#*|}; else : > "$MOCK_LOG.$_h"; _r=${_r%%|*}; fi
# The mirrors redirect, so a probe that does not follow redirects or fetches the whole file is wrong.
case " $* " in *" -sL "*" -r 0-"*) ;; *) printf '000 0.000'; exit 2 ;; esac
case " $* " in *" --max-time 1.5 "*) [ ! -f "$MOCK_LOG.busy" ] || : > "$MOCK_LOG.overlap"; : > "$MOCK_LOG.busy"; sleep 0.05; rm -f "$MOCK_LOG.busy" ;; esac
[ -z "${MOCK_OLD_CURL:-}" ] || case " $* " in *" 1.5 "*) exit 2 ;; esac
case "$_r" in
    fast) printf '206 4000000.000' ;;
    slow) printf '206 300000.000'; exit 28 ;;
    blocked) printf '000 0.000'; exit 7 ;;
    *) printf '%s.000' "$_r" ;;
esac
EOF
chmod +x "$_WORK/bin/curl"

_VARS="UV_DEFAULT_INDEX UV_INDEX UV_INDEX_STRATEGY PIP_INDEX_URL PIP_EXTRA_INDEX_URL UNSLOTH_PYTORCH_MIRROR UNSLOTH_NODE_MIRROR UNSLOTH_NPM_REGISTRY UV_PYTHON_INSTALL_MIRROR UNSLOTH_UV_WHEEL_MIRROR"

# _run <shell> [VAR=value ...]: prints "VAR=value" for every exported var in $_VARS, then the step lines.
_run() {
    _shell="$1"; shift
    rm -f "$_WORK/curl.log".*
    : > "$_WORK/curl.log"
    mkdir -p "$_WORK/home"
    case "$_shell" in
        dash) _flags="set -e" ;;
        *) _flags="set -euo pipefail" ;;
    esac
    env -i PATH="$_WORK/bin:/usr/bin:/bin" HOME="$_WORK/home" MOCK_LOG="$_WORK/curl.log" "$@" \
        "$_shell" -c "$_flags
C_WARN=; step() { echo \"STEP \$2\"; }; substep() { echo \"SUBSTEP \$1\"; }
. '$_WORK/block.sh'
_mirror_fallback \${_MF_ARGS:-}
eval \"\${_AFTER:-}\"
for _v in $_VARS; do eval \"[ -z \\\"\\\${\$_v+x}\\\" ] || echo \\\"\$_v=\\\$\$_v\\\"\"; done"
}

M=https://tuna.mirrors.cernet.edu.cn
for SH in dash bash; do
    command -v "$SH" >/dev/null 2>&1 || { echo "  SKIP: $SH not installed"; continue; }
    out=$(_run "$SH")
    assert_eq "[$SH] fast hosts export and print nothing" "" "$out"
    assert_eq "[$SH] fast hosts never touch a mirror" "" "$(grep -E 'cernet|npmmirror' "$_WORK/curl.log" || true)"
    assert_eq "[$SH] defaults are timed one at a time" "no" "$([ -f "$_WORK/curl.log.overlap" ] && echo overlap || echo no)"
    assert_contains "[$SH] a curl that rejects the probe counts as no answer" "$(_run "$SH" MOCK_OLD_CURL=1 MOCK_PYPI=blocked)" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    out=$(_run "$SH" MOCK_PYPI=blocked)
    assert_contains "[$SH] blocked pypi: uv default index is the mirror" "$out" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    assert_contains "[$SH] blocked pypi: pip index is the mirror" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    assert_not_contains "[$SH] blocked pypi: no unreachable second uv index" "$out" "UV_INDEX="
    assert_not_contains "[$SH] blocked pypi: no unsafe-first-match against a dead index" "$out" "UV_INDEX_STRATEGY"
    assert_not_contains "[$SH] blocked pypi: no pip extra index" "$out" "PIP_EXTRA_INDEX_URL"
    assert_contains "[$SH] blocked pypi: says so with both speeds" "$out" "STEP PyPI is blocked (0 KB/s, mirror 3906 KB/s); using $M/pypi/web/simple"
    assert_contains "[$SH] a switch names the opt-out" "$out" "SUBSTEP Set UNSLOTH_MIRROR_FALLBACK=0"
    out=$(_run "$SH" MOCK_PYPI=slow _AFTER='echo "SPARE $_UNSLOTH_MIRROR_SPARE"')
    assert_eq "[$SH] slow pypi: the mirror is the only uv and pip index" "UV_DEFAULT_INDEX=$M/pypi/web/simple PIP_INDEX_URL=$M/pypi/web/simple" "$(echo "$out" | grep -E '^(UV_|PIP_)' | paste -sd' ' -)"
    assert_contains "[$SH] slow pypi: pypi.org, which still answers, is spared behind the mirror for what it has not synced" "$out" " unsynced|UV_DEFAULT_INDEX=https://pypi.org/simple|UV_INDEX=$M/pypi/web/simple|UV_INDEX_STRATEGY=unsafe-first-match|PIP_EXTRA_INDEX_URL=https://pypi.org/simple|PIP_INDEX_URL=$M/pypi/web/simple"
    assert_eq "[$SH] slow pypi: raced again beside the mirror" "2" "$(grep -c files.pythonhosted "$_WORK/curl.log")"
    assert_contains "[$SH] a user index strategy is kept" "$(_run "$SH" MOCK_PYPI=slow UV_INDEX_STRATEGY=first-index _AFTER='echo "SPARE $_UNSLOTH_MIRROR_SPARE"')" "unsynced|UV_DEFAULT_INDEX=https://pypi.org/simple|UV_INDEX=$M/pypi/web/simple|UV_INDEX_STRATEGY=first-index|"
    assert_not_contains "[$SH] blocked pypi: nothing is spared behind the mirror" "$(_run "$SH" MOCK_PYPI=blocked _AFTER='echo "SPARE $_UNSLOTH_MIRROR_SPARE"')" "unsynced"
    assert_eq "[$SH] a mirror slower than the slow default changes nothing" "" "$(_run "$SH" MOCK_PYPI=slow MOCK_CERNET="206 200000")"
    out=$(_run "$SH" MOCK_PYPI=blocked MOCK_CERNET=blocked)
    assert_eq "[$SH] a dead mirror changes and prints nothing" "" "$out"
    assert_eq "[$SH] a default back above 1 MiB/s in the race is kept" "" "$(_run "$SH" MOCK_PYPI="slow|206 2000000" MOCK_CERNET="206 3000000")"
    assert_contains "[$SH] a redirect that stalls counts as blocked" "$(_run "$SH" MOCK_PYPI="302 0")" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    assert_contains "[$SH] an HTTP error in the race weighs as 0 B/s" "$(_run "$SH" MOCK_PYPI="slow|404 5000000" MOCK_CERNET="206 400000")" "STEP PyPI is blocked (0 KB/s, mirror 390 KB/s)"
    out=$(_run "$SH" MOCK_PYPI="404 0")
    assert_eq "[$SH] a default answering an HTTP error is kept, without timing the mirror" "" "$out$(grep -F "$M" "$_WORK/curl.log" || true)"
    out=$(_run "$SH" MOCK_TORCH=blocked MOCK_NODE=slow MOCK_NPM=blocked)
    for _kv in "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "UNSLOTH_NODE_MIRROR=https://registry.npmmirror.com/-/binary/node" "UNSLOTH_NPM_REGISTRY=https://registry.npmmirror.com"; do assert_contains "[$SH] mirror $_kv" "$out" "$_kv"; done
    assert_not_contains "[$SH] only slow hosts switch" "$out" "PIP_INDEX_URL"
    assert_eq "[$SH] each mirror tree is timed once" "3" "$(grep -E '^https://(tuna[.]mirrors[.]cernet[.]edu[.]cn|registry[.]npmmirror[.]com)/' "$_WORK/curl.log" | grep -cE '[.](t?gz|whl)$')"
    out=$(_run "$SH" MOCK_TORCH=blocked MOCK_NODE=blocked MOCK_NPMMIRRORNODE=blocked)
    assert_contains "[$SH] torch waits on CERNET's torch tree ..." "$out" "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"
    assert_not_contains "[$SH] ... and node on its node tree" "$out" "UNSLOTH_NODE_MIRROR"
    assert_contains "[$SH] an unreachable pypi.org means blocked mode" "$(_run "$SH" MOCK_PYPIINDEX=blocked)" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    out=$(_run "$SH" MOCK_PYPI=blocked MOCK_TORCH=blocked MOCK_ASTRAL=blocked MOCK_CERNETPYPIINDEX=blocked)
    assert_not_contains "[$SH] a mirror whose index does not answer is not used ..." "$out" "PIP_INDEX_URL"
    assert_contains "[$SH] ... while a mirror with its own live index is" "$out" "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"
    assert_contains "[$SH] ... and the uv wheel, which skips the index, still is" "$out" "UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web"
    assert_eq "[$SH] a mirror answering a redirect or an HTTP error is not used" "" "$(_run "$SH" MOCK_TORCH=blocked MOCK_CERNETTORCH="302 900000" MOCK_NODE=blocked MOCK_NPMMIRRORNODE="403 900000")"
    assert_not_contains "[$SH] a dead CERNET torch index keeps torch" "$(_run "$SH" MOCK_TORCH=blocked MOCK_CERNETTORCHINDEX=blocked)" "UNSLOTH_PYTORCH_MIRROR"
    assert_contains "[$SH] an unreachable torch index switches torch" "$(_run "$SH" MOCK_TORCHINDEX=blocked)" "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"
    out=$(_run "$SH" MOCK_PYPI=slow MOCK_CERNET=slow MOCK_NPM=slow MOCK_NPMMIRROR="206 400000")
    assert_contains "[$SH] each host is weighed against its own mirror" "$out" "UNSLOTH_NPM_REGISTRY=https://registry.npmmirror.com"
    assert_not_contains "[$SH] ... and a mirror no faster is not used" "$out" "PIP_INDEX_URL"
    out=$(_run "$SH" MOCK_PYPI=blocked UV_INDEX_URL=https://corp.example/simple)
    assert_not_contains "[$SH] user uv index env: uv untouched" "$out" "UV_DEFAULT_INDEX"
    assert_contains "[$SH] user uv index env: pip still falls back" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    mkdir -p "$_WORK/cfg/uv"
    for _toml in '[[index]]\nurl = "https://corp.example/simple"' 'index = [{ url = "https://corp.example/simple", default = true }]' 'pip.index-url = "https://corp.example/simple"'; do
        printf "$_toml\n" > "$_WORK/cfg/uv/uv.toml"
        assert_not_contains "[$SH] uv.toml $_toml: uv untouched" "$(_run "$SH" MOCK_PYPI=blocked XDG_CONFIG_HOME="$_WORK/cfg")" "UV_DEFAULT_INDEX"
    done
    rm -rf "$_WORK/cfg"
    mkdir -p "$_WORK/proj/src"
    printf '[project]\nname = "p"\n\n[[tool.uv.index]]\nurl = "https://corp.example/simple"\ndefault = true\n' > "$_WORK/proj/pyproject.toml"
    out=$(cd "$_WORK/proj/src" && _run "$SH" MOCK_PYPI=blocked)
    assert_not_contains "[$SH] a parent pyproject.toml [tool.uv] index: uv untouched" "$out" "UV_DEFAULT_INDEX"
    assert_contains "[$SH] a parent pyproject.toml [tool.uv] index: pip still falls back" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    printf 'native-tls = true\n' > "$_WORK/proj/src/uv.toml"
    assert_contains "[$SH] the nearest uv.toml hides a parent pyproject.toml" "$(cd "$_WORK/proj/src" && _run "$SH" MOCK_PYPI=blocked)" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    rm -rf "$_WORK/proj"
    mkdir -p "$_WORK/home/.pip"
    printf '[global]\nindex-url = https://corp.example/simple\n' > "$_WORK/home/.pip/pip.conf"
    out=$(_run "$SH" MOCK_PYPI=blocked)
    assert_not_contains "[$SH] pip.conf index: pip untouched" "$out" "PIP_INDEX_URL"
    printf '[global]\ntimeout = 60\n' > "$_WORK/home/.pip/pip.conf"
    out=$(_run "$SH" MOCK_PYPI=blocked)
    assert_contains "[$SH] pip.conf without an index still falls back" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    rm -rf "$_WORK/home/.pip"
    mkdir -p "$_WORK/venv"
    printf '[global]\nindex-url = https://corp.example/simple\n' > "$_WORK/venv/pip.conf"
    out=$(_run "$SH" MOCK_PYPI=blocked VENV_DIR="$_WORK/venv")
    assert_not_contains "[$SH] the Studio venv's pip.conf index: pip untouched" "$out" "PIP_INDEX_URL"
    assert_contains "[$SH] the Studio venv's pip.conf index: uv still falls back" "$out" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    rm -rf "$_WORK/venv"
    out=$(_run "$SH" MOCK_NODE=blocked _MF_ARGS=spare _AFTER='echo "SPARE $_UNSLOTH_MIRROR_SPARE"')
    assert_eq "[$SH] spare only: no host is probed" "" "$(cat "$_WORK/curl.log")"
    assert_contains "[$SH] spare only: every host gets its retry" "$out" "SPARE pypi|UV_DEFAULT_INDEX=$M/pypi/web/simple|PIP_INDEX_URL=$M/pypi/web/simple torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl node|"
    assert_contains "[$SH] spare only: the probe still runs later" "$(_run "$SH" MOCK_NODE=blocked _MF_ARGS=spare _AFTER=_mirror_fallback)" "UNSLOTH_NODE_MIRROR=https://registry.npmmirror.com/-/binary/node"
    out=$(_run "$SH" MOCK_PYPI=blocked UV_DEFAULT_INDEX=https://a.example/simple PIP_INDEX_URL=https://a.example/simple)
    assert_eq "[$SH] both tools configured: PyPI never probed" "" "$(grep -E 'files.pythonhosted|pypi.org' "$_WORK/curl.log" || true)"
    out=$(_run "$SH" MOCK_ASTRAL=slow)
    assert_contains "[$SH] slow releases.astral.sh: uv's Python builds come from npmmirror" "$out" "UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone"
    assert_contains "[$SH] slow releases.astral.sh: the pinned uv comes from the PyPI mirror" "$out" "UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web"
    assert_eq "[$SH] releases.astral.sh is timed once for uv and Python" "2" "$(grep -c releases.astral.sh "$_WORK/curl.log")"
    out=$(_run "$SH" MOCK_ASTRAL=slow MOCK_NPMMIRROR=blocked)
    assert_not_contains "[$SH] Python builds wait on npmmirror, not CERNET" "$out" "UV_PYTHON_INSTALL_MIRROR"
    assert_contains "[$SH] ... while uv still takes CERNET" "$out" "UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web"
    printf 'python-install-mirror = "https://corp.example/pbs"\n' > "$_WORK/home/uv.toml"
    out=$(_run "$SH" MOCK_ASTRAL=blocked MOCK_PYPI=blocked UV_CONFIG_FILE="$_WORK/home/uv.toml")
    assert_not_contains "[$SH] uv.toml python-install-mirror is kept" "$out" "UV_PYTHON_INSTALL_MIRROR"
    assert_contains "[$SH] uv.toml python-install-mirror leaves the index fallback on" "$out" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    for _src in UV_INSTALLER_GITHUB_BASE_URL UNSLOTH_UV_WHEEL_MIRROR; do out=$(_run "$SH" MOCK_ASTRAL=blocked $_src=https://corp.example/uv)
        assert_not_contains "[$SH] a user $_src keeps the uv source" "$out" "UNSLOTH_UV_WHEEL_MIRROR=$M"; done
    out=$(_run "$SH" MOCK_ASTRAL=blocked UV_PYTHON_INSTALL_MIRROR=https://corp.example/pbs UNSLOTH_UV_WHEEL_MIRROR=https://corp.example/uv)
    assert_eq "[$SH] user Python and uv sources skip the releases.astral.sh probe" "" "$(grep -F releases.astral.sh "$_WORK/curl.log" || true)"
    out=$(_run "$SH" MOCK_TORCH=blocked UNSLOTH_TORCH_INDEX_URL=https://corp.example/whl/cu128)
    assert_not_contains "[$SH] a pinned torch index is not overridden" "$out" "UNSLOTH_PYTORCH_MIRROR"
    out=$(_run "$SH" MOCK_NPM=blocked UNSLOTH_NPM_REGISTRY=https://corp.example/npm/)
    assert_contains "[$SH] a user npm registry is kept" "$out" "UNSLOTH_NPM_REGISTRY=https://corp.example/npm/"
    assert_eq "[$SH] a user npm registry skips the npm probe" "" "$(grep -F npmjs "$_WORK/curl.log" || true)"
    for _off in 0 false off; do
        out=$(_run "$SH" MOCK_PYPI=blocked UNSLOTH_MIRROR_FALLBACK=$_off)
        assert_eq "[$SH] UNSLOTH_MIRROR_FALLBACK=$_off probes nothing" "" "$(cat "$_WORK/curl.log")"
    done
    out=$(_run "$SH" MOCK_PYPI=blocked _UNSLOTH_MIRROR_PROBED=1)
    assert_eq "[$SH] a parent installer's probe is not repeated" "" "$(cat "$_WORK/curl.log")"
    out=$(_run "$SH" MOCK_TORCH=blocked PIP_INDEX_URL=https://corp.example/simple _AFTER='echo "SPARE $_UNSLOTH_MIRROR_SPARE"; _mirror_switch pypi || :; _mirror_switch pypi || echo AGAIN no')
    assert_eq "[$SH] hosts left on their default, not the switched one, are spared with their blocked-mode vars" "SPARE pypi|UV_DEFAULT_INDEX=$M/pypi/web/simple node|UNSLOTH_NODE_MIRROR=https://registry.npmmirror.com/-/binary/node npm|UNSLOTH_NPM_REGISTRY=https://registry.npmmirror.com python|UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone uvbin|UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web" "$(echo "$out" | grep '^SPARE')"
    assert_eq "[$SH] a failed step switches its host once" "STEP PyPI failed; retrying through $M/pypi/web/simple|AGAIN no|UV_DEFAULT_INDEX=$M/pypi/web/simple" "$(echo "$out" | grep -E '^(STEP PyPI failed|AGAIN|UV_DEFAULT_INDEX)' | paste -sd'|' -)"
done

# Real uv / pip / npm failure output (trimmed): only a transport failure naming a probed default is the mirror's to retry.
_fail_uv_timeout='error: Failed to fetch: `https://pypi.org/simple/six/`
  Caused by: error sending request for url (https://pypi.org/simple/six/)
  Caused by: operation timed out'
_fail_uv_503='  Caused by: HTTP status server error (503 Service Unavailable) for url (https://download.pytorch.org/whl/cu128/torch/)'
_fail_pip_timeout="WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError(\"HTTPSConnectionPool(host='pypi.org', port=443): Read timed out. (read timeout=3.0)\")': /simple/six/"
_fail_npm='npm error code ECONNRESET: network request to https://registry.npmjs.org/vite failed, reason: socket hang up'
_fail_python='error: Failed to download https://releases.astral.sh/github/python-build-standalone/releases/download/20260910/cpython-3.12.12.tar.gz: error decoding response body'
_fail_uv_stall='error: Failed to download `torch==2.9.1+cu128`: Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).'
_fail_nover='  Caused by: Because there is no version of torch==1.0.99 [...] hint: `torch` was found on https://download.pytorch.org/whl/cu128, but not at the requested version (torch==1.0.99). A compatible version may be available on a subsequent index (e.g., https://pypi.org/simple).'
_fail_uv_lag='  cause: Because only unsloth<=2026.9.9 is available and you require unsloth>=2026.9.10, we can conclude that your requirements are unsatisfiable.'
_fail_pip_lag='ERROR: No matching distribution found for unsloth>=2026.9.10'
_fail_git='error: Git operation failed: failed to fetch https://github.com/unslothai/unsloth-zoo: Connection reset by peer (os error 54)'
_failed_host() { printf '%s\n' "$1" > "$_WORK/fail.log"; sh -c ". '$_WORK/block.sh'; _mirror_failed_host '$_WORK/fail.log' '${2:-}'" || echo none; }
assert_eq "failed host: the default each transport failure names; unsynced for a version or package not found; none for another host" "pypi torch pypi npm python unsynced unsynced unsynced none" "$(for _o in "$_fail_uv_timeout" "$_fail_uv_503" "$_fail_pip_timeout" "$_fail_npm" "$_fail_python" "$_fail_nover" "$_fail_uv_lag" "$_fail_pip_lag" "$_fail_git"; do _failed_host "$_o"; done | paste -sd' ' -)"
assert_eq "failed host: a stalled download names no URL, so the host that ran it; none when another URL is named" "torch none none" "$({ _failed_host "$_fail_uv_stall" torch; _failed_host "$_fail_uv_stall"; _failed_host "$_fail_git" pypi; } | paste -sd' ' -)"

# run_install_cmd(_retry) around the real _run_install_cmd_once; the stub uv logs each run and fails as FAIL says, without a mirror index.
{ cat "$_WORK/block.sh"; for _f in run_install_cmd _mirror_retry_install _ric_tee _run_install_cmd_once run_install_cmd_retry; do sed -n "/^$_f() {/,/^}/p" "$INSTALL_SH"; done; } > "$_WORK/retry.sh"
mkdir -p "$_WORK/uvbin"
cat > "$_WORK/uvbin/uv" <<'EOF'
#!/bin/sh
echo "RUN $* ${UV_DEFAULT_INDEX:-}${UV_INDEX:+ +$UV_INDEX}" >&3
case "$*" in
    *download.pytorch.org*) printf '%s\n' "$FAIL"; exit 7 ;;
    *--default-index*) ;;
    *) [ -n "${UV_DEFAULT_INDEX:-}" ] && { [ -z "${LAGGING:-}" ] || [ -n "${UV_INDEX:-}" ]; } || { printf '%s\n' "$FAIL"; exit 7; } ;;
esac
[ -z "${MIRROR_FAILS:-}" ] || { printf '%s\n' "$FAIL"; exit 8; }
EOF
chmod +x "$_WORK/uvbin/uv"
_retry() {
    _cmd=$1; shift
    env -i PATH="$_WORK/uvbin:/usr/bin:/bin" _UNSLOTH_MIRROR_SPARE="pypi|UV_DEFAULT_INDEX=$M/pypi/web/simple torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" UNSLOTH_INSTALL_RETRY_DELAY=0 "$@" sh -c "
step() { echo \"STEP \$2\"; }; substep() { :; }; tauri_stream_log() { :; }; tauri_clear_install_error() { :; }; _redact_install_output() { cat \"\$@\" > /dev/null; }
_is_verbose() { [ \"\${VERBOSE:-}\" = 1 ]; }
_uv_download_markers() { if [ -n \"\$1\" ]; then cat >> \"\$1\"; else cat; fi; }
. '$_WORK/retry.sh'
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
\$1 3>&1; echo \"RC \$? INDEX=\${UV_DEFAULT_INDEX:-} TORCH=\$TORCH_INDEX_URL SPARE=\$_UNSLOTH_MIRROR_SPARE\"; [ -z \"\${THEN:-}\" ] || { \$THEN 3>&1; echo \"RC \$?\"; }" retry "$_cmd"
}
_pipe() { "$@" 2>/dev/null | grep -E '^(RUN|STEP|RC)' | paste -sd'|' -; }
assert_eq "retry: a PyPI transport failure reruns once on the mirror, which later steps keep" "RUN pip install foo |STEP PyPI failed; retrying through $M/pypi/web/simple|RUN pip install foo $M/pypi/web/simple|RC 0 INDEX=$M/pypi/web/simple TORCH=https://download.pytorch.org/whl/cu128 SPARE=torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "$(_pipe _retry 'run_install_cmd deps uv pip install foo' FAIL="$_fail_uv_timeout")"
assert_eq "retry: ... also when the output was streamed (verbose)" "RUN pip install foo |STEP PyPI failed; retrying through $M/pypi/web/simple|RUN pip install foo $M/pypi/web/simple|RC 0 INDEX=$M/pypi/web/simple TORCH=https://download.pytorch.org/whl/cu128 SPARE=torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "$(_pipe _retry 'run_install_cmd deps uv pip install foo' FAIL="$_fail_uv_timeout" VERBOSE=1)"
assert_eq "retry: a version not found, with nothing spared behind the mirror, keeps the default and its spares" "RUN pip install foo |RC 7 INDEX= TORCH=https://download.pytorch.org/whl/cu128 SPARE=pypi|UV_DEFAULT_INDEX=$M/pypi/web/simple torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "$(_pipe _retry 'run_install_cmd deps uv pip install foo' FAIL="$_fail_nover")"
assert_eq "retry: a failed mirror rerun restores the default for later steps" "RUN pip install foo |STEP PyPI failed; retrying through $M/pypi/web/simple|RUN pip install foo $M/pypi/web/simple|RC 8 INDEX= TORCH=https://download.pytorch.org/whl/cu128 SPARE=torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "$(_pipe _retry 'run_install_cmd deps uv pip install foo' FAIL="$_fail_uv_timeout" MIRROR_FAILS=1)"
assert_eq "retry: a torch transport failure reruns on the mirror's index, and later torch steps follow" "RUN pip install torch --default-index https://download.pytorch.org/whl/cu128 |STEP download.pytorch.org failed; retrying through $M/pytorch/whl|RUN pip install torch --default-index $M/pytorch/whl/cu128 |RC 0 INDEX= TORCH=$M/pytorch/whl/cu128 SPARE=pypi|UV_DEFAULT_INDEX=$M/pypi/web/simple|RUN pip install torchvision --default-index $M/pytorch/whl/cu128 |RC 0" "$(_pipe _retry 'run_install_cmd torch uv pip install torch --default-index https://download.pytorch.org/whl/cu128' FAIL="$_fail_uv_stall" THEN='run_install_cmd tv uv pip install torchvision --default-index https://download.pytorch.org/whl/cu128')"
assert_eq "retry: the retrying runner gives the default every attempt before the mirror (a stall naming no URL is its PyPI)" "RUN pip install foo |RUN pip install foo |STEP PyPI failed; retrying through $M/pypi/web/simple|RUN pip install foo $M/pypi/web/simple|RC 0" "$(_pipe _retry 'run_install_cmd_retry deps uv pip install foo' FAIL="$_fail_uv_stall" UNSLOTH_INSTALL_RETRIES=2 | sed 's/ INDEX=.*//')"
assert_eq "retry: a pinned command is not moved to the PyPI mirror" "RUN pip install x --index-url https://download.pytorch.org/whl/cu128 |RC 7" "$(_pipe _retry 'run_install_cmd x uv pip install x --index-url https://download.pytorch.org/whl/cu128' FAIL="$_fail_uv_timeout" | sed 's/ INDEX=.*//')"
assert_eq "retry: a torch failure without a torch URL to move is not retried" "RUN pip install torch --torch-backend=auto |RC 7" "$(_pipe _retry 'run_install_cmd tb uv pip install torch --torch-backend=auto' FAIL="$_fail_uv_503" | sed 's/ INDEX=.*//')"
assert_eq "retry: ... nor a stall under a source the command picks itself" "RUN pip install torch --torch-backend=auto |RC 7" "$(_pipe _retry 'run_install_cmd tb uv pip install torch --torch-backend=auto' FAIL="$_fail_uv_stall" | sed 's/ INDEX=.*//')"
_lag() { _lagcmd=$1; shift; _pipe _retry "run_install_cmd u uv pip install $_lagcmd" FAIL="$_fail_uv_lag" LAGGING=1 UV_DEFAULT_INDEX="$M/pypi/web/simple" _UNSLOTH_MIRROR_SPARE="torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl unsynced|UV_DEFAULT_INDEX=https://pypi.org/simple|UV_INDEX=$M/pypi/web/simple" "$@" | sed 's/ TORCH=[^ ]*//'; }
assert_eq "retry: a release the mirror lacks reruns once with pypi.org behind it, which later steps keep" "RUN pip install unsloth $M/pypi/web/simple|STEP The PyPI mirror failed; retrying through https://pypi.org/simple|RUN pip install unsloth https://pypi.org/simple +$M/pypi/web/simple|RC 0 INDEX=https://pypi.org/simple SPARE=torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "$(_lag unsloth)"
assert_eq "retry: ... also under --torch-backend, whose other packages resolve on the index" "RC 0 INDEX=https://pypi.org/simple" "$(_lag 'unsloth --torch-backend=auto' | grep -o 'RC [0-9] INDEX=[^ ]*')"
assert_eq "retry: a command pinning its index is not given pypi.org" "RUN pip install unsloth --default-index https://corp.example/simple |RC 8 INDEX=$M/pypi/web/simple SPARE=torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl unsynced|UV_DEFAULT_INDEX=https://pypi.org/simple|UV_INDEX=$M/pypi/web/simple" "$(_lag 'unsloth --default-index https://corp.example/simple' MIRROR_FAILS=1)"
assert_eq "retry: ... nor one ending in --no-index" "RC 7" "$(_lag 'unsloth --no-index' | grep -o 'RC [0-9]')"

_npm() {
    printf '%s\n' "$1" > "$_WORK/npm.log"; shift
    env -i PATH=/usr/bin:/bin _UNSLOTH_MIRROR_SPARE="npm|UNSLOTH_NPM_REGISTRY=$NPMM" "$@" bash -c "
step() { echo \"STEP \$2\"; }; run_quiet_no_exit() { shift; echo \"RUN \$*\"; [ -z \"\${MIRROR_FAILS:-}\" ]; }
. '$_WORK/block.sh'
$(sed -n '/^_npm_mirror_retry() {/,/^}/p' "$SETUP_SH")
_CAPTURE_LOG='$_WORK/npm.log'; _NPM_REGISTRY_ARGS=()
_npm_mirror_retry 'npm install'; echo \"RC \$? REG=\${UNSLOTH_NPM_REGISTRY:-} ARGS=\${_NPM_REGISTRY_ARGS[*]}\"" | paste -sd'|' -
}
NPMM=https://registry.npmmirror.com
assert_eq "npm: a registry transport failure reruns once on npmmirror, which later installs keep" "STEP registry.npmjs.org failed; retrying through $NPMM|RUN npm install --no-fund --no-audit --loglevel=error --registry $NPMM|RC 0 REG=$NPMM ARGS=--registry $NPMM" "$(_npm "$_fail_npm")"
assert_eq "npm: ... not when the rerun fails too" "STEP registry.npmjs.org failed; retrying through $NPMM|RUN npm install --no-fund --no-audit --loglevel=error --registry $NPMM|RC 1 REG= ARGS=" "$(_npm "$_fail_npm" MIRROR_FAILS=1)"
assert_eq "npm: a dependency conflict is not retried" "RC 1 REG= ARGS=" "$(_npm 'npm error code ERESOLVE: While resolving: vite@7.1.0 from https://registry.npmjs.org/vite')"

summary
