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
    https://mirrors.cernet.edu.cn/pypi/web/simple/*) _h=CERNETPYPIINDEX ;; https://mirrors.cernet.edu.cn/pytorch/whl/cpu/torch/) _h=CERNETTORCHINDEX ;;
    https://mirrors.cernet.edu.cn/pytorch/*) _h=CERNETTORCH ;; https://mirrors.cernet.edu.cn/nodejs-release/*) _h=CERNETNODE ;;
    https://mirrors.cernet.edu.cn/*) _h=CERNET ;; https://registry.npmmirror.com/*) _h=NPMMIRROR ;;
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
_mirror_fallback
for _v in $_VARS; do eval \"[ -z \\\"\\\${\$_v+x}\\\" ] || echo \\\"\$_v=\\\$\$_v\\\"\"; done"
}

M=https://mirrors.cernet.edu.cn
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
    out=$(_run "$SH" MOCK_PYPI=slow)
    assert_contains "[$SH] slow pypi: mirror is the preferred uv index" "$out" "UV_INDEX=$M/pypi/web/simple"
    assert_contains "[$SH] slow pypi: pypi.org stays as uv's default" "$out" "UV_DEFAULT_INDEX=https://pypi.org/simple"
    assert_contains "[$SH] slow pypi: unsafe-first-match" "$out" "UV_INDEX_STRATEGY=unsafe-first-match"
    assert_contains "[$SH] slow pypi: pip index is the mirror" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    assert_contains "[$SH] slow pypi: pip keeps pypi.org as extra" "$out" "PIP_EXTRA_INDEX_URL=https://pypi.org/simple"
    assert_eq "[$SH] slow pypi: raced again beside the mirror" "2" "$(grep -c files.pythonhosted "$_WORK/curl.log")"
    out=$(_run "$SH" MOCK_PYPI=slow UV_INDEX_STRATEGY=first-index)
    assert_contains "[$SH] a user index strategy is kept" "$out" "UV_INDEX_STRATEGY=first-index"
    assert_eq "[$SH] a mirror slower than the slow default changes nothing" "" "$(_run "$SH" MOCK_PYPI=slow MOCK_CERNET="206 200000")"
    out=$(_run "$SH" MOCK_PYPI=blocked MOCK_CERNET=blocked)
    assert_eq "[$SH] a dead mirror changes and prints nothing" "" "$out"
    assert_eq "[$SH] a default back above 1 MiB/s in the race is kept" "" "$(_run "$SH" MOCK_PYPI="slow|206 2000000" MOCK_CERNET="206 3000000")"
    assert_contains "[$SH] a redirect that stalls counts as blocked" "$(_run "$SH" MOCK_PYPI="302 0")" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    assert_contains "[$SH] an HTTP error in the race weighs as 0 B/s" "$(_run "$SH" MOCK_PYPI="slow|404 5000000" MOCK_CERNET="206 400000")" "STEP PyPI is blocked (0 KB/s, mirror 390 KB/s)"
    out=$(_run "$SH" MOCK_PYPI="404 0")
    assert_eq "[$SH] a default answering an HTTP error is kept, without timing the mirror" "" "$out$(grep -F "$M" "$_WORK/curl.log" || true)"
    out=$(_run "$SH" MOCK_TORCH=blocked MOCK_NODE=slow MOCK_NPM=blocked)
    for _kv in "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl" "UNSLOTH_NODE_MIRROR=$M/nodejs-release" "UNSLOTH_NPM_REGISTRY=https://registry.npmmirror.com"; do assert_contains "[$SH] mirror $_kv" "$out" "$_kv"; done
    assert_not_contains "[$SH] only slow hosts switch" "$out" "PIP_INDEX_URL"
    assert_eq "[$SH] each CERNET tree is timed once" "2" "$(grep "$M" "$_WORK/curl.log" | grep -cE '[.](gz|whl)$')"
    out=$(_run "$SH" MOCK_TORCH=blocked MOCK_NODE=blocked MOCK_CERNETNODE=blocked)
    assert_contains "[$SH] torch waits on CERNET's torch tree ..." "$out" "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"
    assert_not_contains "[$SH] ... and node on its node tree" "$out" "UNSLOTH_NODE_MIRROR"
    assert_contains "[$SH] an unreachable pypi.org means blocked mode" "$(_run "$SH" MOCK_PYPIINDEX=blocked)" "UV_DEFAULT_INDEX=$M/pypi/web/simple"
    out=$(_run "$SH" MOCK_PYPI=blocked MOCK_TORCH=blocked MOCK_ASTRAL=blocked MOCK_CERNETPYPIINDEX=blocked)
    assert_not_contains "[$SH] a mirror whose index does not answer is not used ..." "$out" "PIP_INDEX_URL"
    assert_contains "[$SH] ... while a mirror with its own live index is" "$out" "UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"
    assert_contains "[$SH] ... and the uv wheel, which skips the index, still is" "$out" "UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web"
    assert_eq "[$SH] a mirror answering a redirect or an HTTP error is not used" "" "$(_run "$SH" MOCK_TORCH=blocked MOCK_CERNETTORCH="302 900000" MOCK_NODE=blocked MOCK_CERNETNODE="403 900000")"
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
    mkdir -p "$_WORK/home/.pip"
    printf '[global]\nindex-url = https://corp.example/simple\n' > "$_WORK/home/.pip/pip.conf"
    out=$(_run "$SH" MOCK_PYPI=blocked)
    assert_not_contains "[$SH] pip.conf index: pip untouched" "$out" "PIP_INDEX_URL"
    printf '[global]\ntimeout = 60\n' > "$_WORK/home/.pip/pip.conf"
    out=$(_run "$SH" MOCK_PYPI=blocked)
    assert_contains "[$SH] pip.conf without an index still falls back" "$out" "PIP_INDEX_URL=$M/pypi/web/simple"
    rm -rf "$_WORK/home/.pip"
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
done

summary
