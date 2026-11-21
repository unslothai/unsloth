#!/bin/bash
# _resolve_cuda_archs() from studio/setup.sh: compute_cap text to a deduped ';' arch list;
# empty means a CPU build, and an explicit override wins.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_resolve_cuda_archs()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"

# $1 = raw compute_cap text, $2 = override
run_resolve() {
    bash -c ". '$_FUNC_FILE'; _resolve_cuda_archs \"\$1\" \"\$2\"" _ "$1" "$2"
}

echo "=== test_resolve_cuda_archs ==="

assert_eq "single 8.6" "86" "$(run_resolve "8.6" "")"

assert_eq "distinct 8.6 + 9.0" "86;90" "$(run_resolve "$(printf '8.6\n9.0\n')" "")"

assert_eq "dedup 12.0 x2" "120" "$(run_resolve "$(printf '12.0\n12.0\n')" "")"

assert_eq "empty input" "" "$(run_resolve "" "")"

assert_eq "garbage N/A" "" "$(run_resolve "$(printf 'N/A\n[Not Supported]\n')" "")"

assert_eq "mixed valid+junk" "86;90" "$(run_resolve "$(printf '8.6\nfoo\n9.0\n')" "")"

assert_eq "whitespace stripped" "86" "$(run_resolve "$(printf '  8.6 \r\n')" "")"

assert_eq "override wins" "120" "$(run_resolve "8.6" "120")"

assert_eq "override no detection" "86;90" "$(run_resolve "" "86;90")"

assert_eq "future 10.0" "100" "$(run_resolve "10.0" "")"

rm -f "$_FUNC_FILE"

summary
