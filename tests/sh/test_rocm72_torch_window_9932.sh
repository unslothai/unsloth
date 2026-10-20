#!/usr/bin/env bash
# The generic rocm7.2 leaf admits torch 2.11-2.14 (#9932); the AMD per-arch leaves stay on 2.11,
# and an existing install keeps its release. Lifts the real case block out of install.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$HERE/../../install.sh"
PASS=0
FAIL=0
assert_eq() {
    if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 (expected '$2', got '$3')"; fi
}

_BLOCK=$(awk '/^# rocm7.2 and per-gfx indexes ship torch 2.11.0/ { on = 1 } on { print } on && /^esac$/ { exit }' "$INSTALL_SH")
assert_eq "case block found" "yes" "$([ -n "$_BLOCK" ] && echo yes || echo no)"

constraints_for_leaf() {
    bash -c '
        TORCH_CONSTRAINT="torch>=2.4,<2.12.0"
        TORCHVISION_CONSTRAINT="torchvision>=0.19,<0.27.0"
        TORCHAUDIO_CONSTRAINT="torchaudio>=2.4,<2.12.0"
        _torch_index_leaf="$1"
        eval "$2"
        echo "$TORCH_CONSTRAINT|$TORCHVISION_CONSTRAINT|$TORCHAUDIO_CONSTRAINT"
    ' _ "$1" "$_BLOCK"
}

echo "=== generic rocm7.2: 2.11 floor, 2.14 ceiling ==="
assert_eq "rocm7.2" "torch>=2.11.0,<2.15.0|torchvision>=0.26.0,<0.30.0|torchaudio>=2.11.0,<2.12.0" \
    "$(constraints_for_leaf rocm7.2)"

echo "=== AMD per-arch leaves unchanged ==="
for _leaf in gfx120x-all gfx1151 gfx1150 gfx1152 gfx103x-all gfx110x-all; do
    assert_eq "$_leaf" "torch>=2.11.0,<2.12.0|torchvision>=0.26.0,<0.27.0|torchaudio>=2.11.0,<2.12.0" \
        "$(constraints_for_leaf "$_leaf")"
done

echo "=== other leaves keep the default window ==="
for _leaf in rocm7.1 rocm6.4 cu128 cpu; do
    assert_eq "$_leaf" "torch>=2.4,<2.12.0|torchvision>=0.19,<0.27.0|torchaudio>=2.4,<2.12.0" \
        "$(constraints_for_leaf "$_leaf")"
done

echo "=== an existing rocm7.2 install keeps its release ==="
_PIN_FN=$(awk '/^_torch_release_in_window\(\)/,/^}$/' "$INSTALL_SH"; awk '/^_previous_torch_pin\(\)/,/^}$/' "$INSTALL_SH")
prev_pin() {
    bash -c 'unset UNSLOTH_TORCH_UPGRADE; eval "$1"; _previous_torch_pin "$2" "$3"' _ "$_PIN_FN" "$1" "$2"
}
assert_eq "2.11.0+rocm7.2 kept" "torch==2.11.0" "$(prev_pin 2.11.0+rocm7.2 'torch>=2.11.0,<2.15.0')"
assert_eq "2.14.1+rocm7.2 kept" "torch==2.14.1" "$(prev_pin 2.14.1+rocm7.2 'torch>=2.11.0,<2.15.0')"
assert_eq "Repair (UNSLOTH_TORCH_UPGRADE=1) moves" "" \
    "$(bash -c 'eval "$1"; UNSLOTH_TORCH_UPGRADE=1 _previous_torch_pin 2.11.0+rocm7.2 "torch>=2.11.0,<2.15.0"' _ "$_PIN_FN")"

echo ""
echo "PASS: $PASS"
echo "FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
