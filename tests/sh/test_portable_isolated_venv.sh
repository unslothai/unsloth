#!/usr/bin/env bash
# The path resolver must work in a clean interpreter with only structlog, and
# seeding must create no studio.db (get_app_setting CREATES one). Needs uv.
set -u
REPO="${1:-$(CDPATH= cd -P -- "$(dirname "$0")/../.." && pwd -P)}"
if ! command -v uv >/dev/null 2>&1; then
    echo "SKIP: uv not on PATH, cannot build an isolated venv"
    exit 0
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fails=0
check() { if [ "$2" = 0 ]; then echo "  PASS  $1"; else echo "  FAIL  $1 ${3:-}"; fails=$((fails+1)); fi; }

echo "[1] build an isolated venv (uv, no site packages from the workspace)"
uv venv --python 3.13 "$WORK/venv" >/dev/null 2>&1 || { echo "FAIL: uv venv"; exit 1; }
PY="$WORK/venv/bin/python"
# structlog is the only thing storage_roots pulls in via loggers/path_utils.
uv pip install --python "$PY" structlog >/dev/null 2>&1 || { echo "FAIL: pip install"; exit 1; }
echo "  venv: $("$PY" -V), packages: $(uv pip list --python "$PY" 2>/dev/null | wc -l)"

echo
echo "[2] resolver imports and runs with only structlog installed"
HOME_A="$WORK/home_a"; mkdir -p "$HOME_A"
out="$(env -i HOME="$HOME_A" PATH="/usr/bin:/bin" UNSLOTH_HOME="$WORK/root_a" \
  "$PY" -c "
import sys, os, json
sys.path.insert(0, '$REPO/studio/backend')
from utils.paths import storage_roots as sr
sr.setup_cache_env()
print(json.dumps({
  'studio_root': str(sr.studio_root()),
  'portable': sr.portable_mode(),
  'compile': os.environ.get('UNSLOTH_COMPILE_LOCATION'),
  'triton_home': os.environ.get('TRITON_HOME'),
}))
" 2>&1)"
case "$out" in
  *studio_root*) check "resolver runs on a bare interpreter" 0; echo "        $out" ;;
  *) check "resolver runs on a bare interpreter" 1 "$out" ;;
esac

echo
echo "[3] no torch, transformers or other heavy import is required"
heavy="$(env -i HOME="$HOME_A" PATH="/usr/bin:/bin" "$PY" -c "
import sys
sys.path.insert(0, '$REPO/studio/backend')
from utils.paths import storage_roots as sr
sr.setup_cache_env()
bad = [m for m in ('torch','transformers','numpy','huggingface_hub','fastapi') if m in sys.modules]
print(','.join(bad) if bad else 'NONE')
" 2>&1 | tail -1)"
[ "$heavy" = "NONE" ] && check "seeding pulls in no heavy modules" 0 \
  || check "seeding pulls in no heavy modules" 1 "imported: $heavy"

echo
echo "[4] CLI seeding side effects: must not create a studio DB or leak outside the root"
HOME_B="$WORK/home_b"; mkdir -p "$HOME_B"
ROOT_B="$WORK/root_b"
env -i HOME="$HOME_B" PATH="/usr/bin:/bin" UNSLOTH_HOME="$ROOT_B" "$PY" -c "
import sys
sys.path.insert(0, '$REPO/studio/backend')
from utils.paths.storage_roots import setup_cache_env
setup_cache_env()
" >/dev/null 2>&1
dbs="$(find "$ROOT_B" "$HOME_B" -name '*.db' 2>/dev/null | wc -l)"
[ "$dbs" = 0 ] && check "seeding creates no database" 0 || check "seeding creates no database" 1 "found $dbs"
outside="$(find "$HOME_B" -mindepth 1 2>/dev/null | wc -l)"
[ "$outside" = 0 ] && check "seeding writes nothing to HOME" 0 \
  || check "seeding writes nothing to HOME" 1 "$(find "$HOME_B" -mindepth 1 | head -5 | tr '\n' ' ')"

echo
echo "[5] seeding is idempotent and cheap on repeat"
t0=$(date +%s%N)
env -i HOME="$HOME_B" PATH="/usr/bin:/bin" UNSLOTH_HOME="$ROOT_B" "$PY" -c "
import sys
sys.path.insert(0, '$REPO/studio/backend')
from utils.paths.storage_roots import setup_cache_env
for _ in range(50):
    setup_cache_env()
" >/dev/null 2>&1
t1=$(date +%s%N)
ms=$(( (t1 - t0) / 1000000 ))
[ "$ms" -lt 20000 ] && check "50 repeat seedings in ${ms}ms" 0 || check "50 repeat seedings too slow (${ms}ms)" 1

echo
echo "[6] resolver survives a missing HOME entirely"
out6="$(env -i PATH="/usr/bin:/bin" UNSLOTH_HOME="$WORK/root_c" "$PY" -c "
import sys
sys.path.insert(0, '$REPO/studio/backend')
from utils.paths import storage_roots as sr
print(sr.studio_root())
" 2>&1 | tail -1)"
case "$out6" in
  "$WORK/root_c/studio") check "no HOME set still resolves the portable root" 0 ;;
  *) check "no HOME set still resolves the portable root" 1 "$out6" ;;
esac

echo
if [ "$fails" -ne 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "All isolated-venv checks passed"
