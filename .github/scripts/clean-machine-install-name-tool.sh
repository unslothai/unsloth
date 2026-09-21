#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Shared install_name_tool trace wrapper and CLT-absent sentinel contract.
set -euo pipefail

usage() {
  echo "usage: $0 write {sentinel|passthrough} TARGET [REAL_TOOL]" >&2
  echo "       $0 verify-sentinel TRACE MARKER" >&2

  echo "       $0 decode TRACE_ARG" >&2
  exit 2
}


decode_trace_arg() {
  encoded=$1
  case "$encoded" in h*) hex=${encoded#h} ;; *) return 1 ;; esac
  case "$hex" in *[!0123456789abcdef]*) return 1 ;; esac
  [ $(( ${#hex} % 2 )) -eq 0 ] || return 1
  decoded=""
  while [ -n "$hex" ]; do
    rest=${hex#??}
    pair=${hex%"$rest"}
    hex=$rest
    printf -v byte '%b' "\\x$pair"
    decoded+=$byte
  done
  printf '%s' "$decoded"
}

case "${1:-}" in
  write)
    kind="${2:-}"
    target="${3:-}"
    [ -n "$target" ] || usage
    case "$kind" in sentinel|passthrough) ;; *) usage ;; esac

    cat > "$target" <<'WRAPPER'
#!/bin/sh
: "${UNSLOTH_TOOL_TRACE:?UNSLOTH_TOOL_TRACE is required}"
encode_trace_arg() {
  printf '%s' "$1" | od -An -v -tx1 | tr -d ' \n'
}
printf 'install_name_tool\t%s' "$#" >> "$UNSLOTH_TOOL_TRACE"
for arg in "$@"; do printf '\th%s' "$(encode_trace_arg "$arg")" >> "$UNSLOTH_TOOL_TRACE"; done
printf '\n' >> "$UNSLOTH_TOOL_TRACE"
WRAPPER
    if [ "$kind" = "sentinel" ]; then
      printf '%s\n' 'exit 97' >> "$target"
    else
      real_tool="${4:-}"
      [ -n "$real_tool" ] || usage
      case "$real_tool" in *'"'*|*$'\n'*) echo "unsupported tool path: $real_tool" >&2; exit 2 ;; esac
      printf 'exec "%s" "$@"\n' "$real_tool" >> "$target"
    fi
    chmod +x "$target"
    ;;

  decode)
    [ "$#" -eq 2 ] || usage
    decode_trace_arg "$2"
    ;;


  verify-sentinel)
    trace="${2:-}"
    marker="${3:-}"
    [ -n "$trace" ] && [ -n "$marker" ] || usage
    [ -n "${UNSLOTH_TOOL_TRACE:-}" ] && [ "$UNSLOTH_TOOL_TRACE" = "$trace" ] || {
      echo "::error::install_name_tool sentinel trace environment is not active" >&2
      exit 1
    }

    set +e
    install_name_tool "--${marker}-sentinel-self-test" >/dev/null 2>&1
    sentinel_rc=$?
    set -e
    [ "$sentinel_rc" -eq 97 ] || {
      echo "::error::install_name_tool sentinel returned $sentinel_rc, expected 97" >&2
      exit 1
    }
    marker_hex=$(printf '%s' "--${marker}-sentinel-self-test" | od -An -v -tx1 | tr -d ' \n')
    expected=$(printf 'install_name_tool\t1\th%s' "$marker_hex")
    grep -Fqx "$expected" "$trace" || {
      echo "::error::install_name_tool sentinel did not preserve/record its self-test argv" >&2
      cat "$trace" >&2 || true
      exit 1
    }
    : > "$trace"
    ;;

  *) usage ;;
esac
