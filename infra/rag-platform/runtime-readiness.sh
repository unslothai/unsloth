#!/usr/bin/env bash

set -euo pipefail

curl --fail --silent --show-error --max-time 5 \
  http://127.0.0.1:9380/api/v1/system/ping >/dev/null
curl --fail --silent --show-error --max-time 5 \
  http://127.0.0.1:9381/api/v1/admin/ping >/dev/null
curl --fail --silent --show-error --max-time 5 \
  http://127.0.0.1:9383/api/v1/admin/ping >/dev/null
curl --fail --silent --show-error --max-time 5 \
  http://127.0.0.1:9384/health >/dev/null
curl --fail --silent --show-error --max-time 5 \
  http://127.0.0.1/api/v1/system/ping >/dev/null
