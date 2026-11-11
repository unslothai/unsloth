#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# run_install_cmd_retry clears wheels uv calls invalid from its cache (#17375).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_FUNC_FILE=$(mktemp)
for _f in run_install_cmd_retry _ric_tee _ric_run _run_install_cmd_once _mirror_retry_install _mirror_failed_host \
    _uv_damaged_wheels _repair_uv_damaged_wheels tauri_log tauri_stream_log tauri_clear_install_error _uv_download_markers; do
    sed -n "/^$_f()/,/^}/p" "$INSTALL_SH" >> "$_FUNC_FILE"
done
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

substep() { printf '%s\n' "$1" >> "$_SUBSTEPS"; }
step() { :; }
sleep() { :; }
_is_verbose() { return 1; }
_redact_install_output() { cat "$@"; }

WORK=$(mktemp -d)
trap 'rm -rf "${WORK:?}"' EXIT
_SUBSTEPS="$WORK/substeps"
fail=0
check() {
    if [ "$2" = 0 ]; then echo "  PASS  $1"; else echo "  FAIL  $1"; fail=1; fi
}

echo "=== _uv_damaged_wheels ==="
# Verbatim from #17375 (Windows), plus the older "Caused by:" spelling.
cat > "$WORK/issue.log" <<'EOF'
Using Python 3.13.16 environment at: .unsloth\studio\unsloth_studio
Resolved 87 packages in 227ms
Uninstalled 1 package in 8ms
error: Failed to install: tqdm-4.70.1-py3-none-any.whl (tqdm==4.70.1)
  cause: The wheel is invalid: Missing .dist-info directory
error: Failed to install: Nest_Asyncio-1.6.0-py3-none-any.whl (nest-asyncio==1.6.0)
  Caused by: The wheel is invalid: Metadata field Name not found
error: Failed to install: zoo-1.0-py3-none-any.whl (unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo)
  cause: The wheel is invalid: Missing .dist-info directory
EOF
[ "$(_uv_damaged_wheels "$WORK/issue.log" | tr '\n' ' ')" = "tqdm nest-asyncio unsloth-zoo " ]
check "names every package uv called an invalid wheel" $?
cat > "$WORK/other.log" <<'EOF'
error: Failed to install: torch-2.10.0-cp313-cp313-win_amd64.whl (torch==2.10.0)
  cause: failed to hardlink file: Access is denied. (os error 5)
error: Request failed after 3 retries: error sending request for url (https://pypi.org/simple/tqdm/)
The wheel is invalid: Missing .dist-info directory
EOF
[ -z "$(_uv_damaged_wheels "$WORK/other.log")" ]
check "ignores other install failures and an unattributed message" $?

echo "=== retry loop with a stub uv ==="
mkdir -p "$WORK/stub"
cat > "$WORK/stub/uv" <<'EOF'
#!/bin/sh
if [ "$1" = cache ]; then shift 2; for n in "$@"; do touch "$STUB_STATE/clean-$n"; done; exit 0; fi
for n in tqdm nest-asyncio; do
    if [ ! -e "$STUB_STATE/clean-$n" ]; then
        echo "error: Failed to install: $n-1.0-py3-none-any.whl ($n==1.0)"
        echo "  cause: The wheel is invalid: Missing .dist-info directory"
        exit 2
    fi
done
EOF
chmod +x "$WORK/stub/uv"
export STUB_STATE="$WORK/state"
mkdir -p "$STUB_STATE"
: > "$_SUBSTEPS"
UNSLOTH_INSTALL_RETRIES=1
UNSLOTH_INSTALL_RETRY_DELAY=0
PATH="$WORK/stub:$PATH" run_install_cmd_retry "install unsloth" uv pip install x > /dev/null 2>&1
check "two damaged packages repaired within one attempt" $?
[ "$(grep -c 'clearing damaged uv cache entries' "$_SUBSTEPS")" = 2 ]
check "one cache clean per damaged package" $?

cat > "$WORK/stub/uv" <<'EOF'
#!/bin/sh
[ "$1" = cache ] && { echo x >> "$STUB_STATE/cleans"; exit 0; }
echo "error: Failed to install: tqdm-1.0-py3-none-any.whl (tqdm==1.0)"
echo "  cause: The wheel is invalid: Missing .dist-info directory"
exit 2
EOF
PATH="$WORK/stub:$PATH" UNSLOTH_INSTALL_RETRIES=2 run_install_cmd_retry "install unsloth" uv pip install x > /dev/null 2>&1
rc=$?
[ "$rc" = 2 ] && [ "$(wc -l < "$STUB_STATE/cleans")" -eq 1 ]
check "a package still damaged after its clean is not cleaned again (exit $rc)" $?

echo "=== real uv, damaged cache archive ==="
if ! command -v uv > /dev/null 2>&1; then
    echo "  FAIL  uv is not on PATH (CI installs it)"
    exit 1
fi
python3 - "$WORK/wheels" alpha_pkg beta_pkg <<'EOF'
import os, sys, zipfile
os.makedirs(sys.argv[1])
for name in sys.argv[2:]:
    di = f"{name}-1.0.dist-info"
    files = {f"{name}/__init__.py": "", f"{di}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
             f"{di}/WHEEL": "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    files[f"{di}/RECORD"] = "".join(f"{p},,\n" for p in files) + f"{di}/RECORD,,\n"
    with zipfile.ZipFile(os.path.join(sys.argv[1], f"{name}-1.0-py3-none-any.whl"), "w") as z:
        for p, c in files.items():
            z.writestr(p, c)
EOF
export UV_CACHE_DIR="$WORK/cache"
uv venv -q "$WORK/v1"
uv venv -q "$WORK/v2"
uv pip install -q --python "$WORK/v1/bin/python" --no-index --find-links "$WORK/wheels" alpha_pkg beta_pkg
for d in "$UV_CACHE_DIR"/archive-v0/*/*.dist-info; do rm -rf "${d:?}"; done
uv pip install -q --python "$WORK/v2/bin/python" --no-index --find-links "$WORK/wheels" alpha_pkg beta_pkg > "$WORK/base.log" 2>&1
rc=$?
[ "$rc" != 0 ] && grep -q 'The wheel is invalid' "$WORK/base.log"
check "control: the damaged cache fails a plain install" $?
: > "$_SUBSTEPS"
UNSLOTH_INSTALL_RETRIES=1 run_install_cmd_retry "install unsloth" uv pip install --python "$WORK/v2/bin/python" --no-index --find-links "$WORK/wheels" alpha_pkg beta_pkg > /dev/null 2>&1
check "retry loop repairs the cache and installs" $?
"$WORK/v2/bin/python" -c "import alpha_pkg, beta_pkg"
check "both packages import" $?

exit "$fail"
