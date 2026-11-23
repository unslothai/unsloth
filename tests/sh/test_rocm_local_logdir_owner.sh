#!/usr/bin/env bash
# docker/test_locally-rocm.sh must refuse a LOG_DIR another user owns (a pre-created shared /tmp
# path), since the container later runs a script from it as root with the HF cache and HF_TOKEN.
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
STUB="$(mktemp -d)"; trap 'rm -rf "${STUB:?}"' EXIT
printf '#!/bin/sh\necho REACHED_DOCKER; exit 1\n' > "$STUB/docker"; chmod +x "$STUB/docker"
FOREIGN=/tmp   # owned by root, not by the test user
[[ -O "$FOREIGN" ]] && { echo "SKIP: running as the owner of $FOREIGN"; exit 0; }
out="$(PATH="$STUB:$PATH" bash "$HERE/docker/test_locally-rocm.sh" --skip-build --skip-notebook --log-dir "$FOREIGN" 2>&1)"
if grep -q REACHED_DOCKER <<<"$out"; then echo "FAIL: ran with a foreign-owned LOG_DIR"; exit 1; fi
grep -q "not owned by" <<<"$out" || { echo "FAIL: unexpected output: $out"; exit 1; }
# Our own fresh dir still proceeds and ends up go-w.
OWN="$STUB/logs"
out="$(PATH="$STUB:$PATH" bash "$HERE/docker/test_locally-rocm.sh" --skip-build --skip-notebook --log-dir "$OWN" 2>&1)"
grep -q REACHED_DOCKER <<<"$out" || { echo "FAIL: own LOG_DIR refused: $out"; exit 1; }
(( (8#$(stat -c %a "$OWN") & 8#022) == 0 )) || { echo "FAIL: own dir writable by others"; exit 1; }
echo PASS
