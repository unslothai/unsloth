#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE=$(printf '\342\224\200%.0s' {1..52})

# ── Parse flags ──
# --local: install from the local repo checkout (overlays unsloth as editable
# and unsloth-zoo from git main). Mirrors install.sh --local for the Colab
# path that runs setup.sh directly without going through install.sh.
if [ "$#" -gt 0 ]; then
    for _arg in "$@"; do
        case "$_arg" in
            --local)
                export STUDIO_LOCAL_INSTALL=1
                export STUDIO_LOCAL_REPO="$REPO_ROOT"
                ;;
        esac
    done
fi

# ── Maintainer-editable defaults ──────────────────────────────────────────
# Change these in the GitHub-hosted script so all users get updated defaults.
# User environment variables always override these baked-in values.
#
#   _DEFAULT_LLAMA_PR_FORCE : PR number to build by default ("" = normal path)
#   _DEFAULT_LLAMA_SOURCE   : git clone URL for source builds
#   _DEFAULT_LLAMA_TAG      : llama.cpp ref to build ("latest" = newest release,
#                             "master" = bleeding-edge, "bNNNN" = specific tag)
#                             Prefer "latest" over "master" -- "master" bypasses
#                             the prebuilt resolver (no matching GitHub release),
#                             forces a source build, and causes HTTP 422 errors.
#                             Only use "master" temporarily when the latest release
#                             is missing support for a new model architecture.
#
#   UNSLOTH_LLAMA_CPP_BACKEND : "auto" (default), "cpu", "cuda", "vulkan",
#                           "hip", or "rocm". Concrete values select and persist a
#                           backend across updates; "auto" restores detection.
#                           Overrides Unsloth's Settings > System selection.
# ──────────────────────────────────────────────────────────────────────────
_DEFAULT_LLAMA_PR_FORCE=""
_DEFAULT_LLAMA_SOURCE="https://github.com/ggml-org/llama.cpp"
_DEFAULT_LLAMA_TAG="latest"
_DEFAULT_LLAMA_FORCE_COMPILE_REF="master"

# ── Colors (same palette as startup_banner / install_python_stack) ──
if [ -n "${NO_COLOR:-}" ]; then
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
elif [ -t 1 ] || [ -n "${FORCE_COLOR:-}" ]; then
    C_TITLE=$'\033[38;5;150m'
    C_DIM=$'\033[38;5;245m'
    C_OK=$'\033[38;5;108m'
    C_WARN=$'\033[38;5;136m'
    C_ERR=$'\033[91m'
    C_RST=$'\033[0m'
else
    C_TITLE= C_DIM= C_OK= C_WARN= C_ERR= C_RST=
fi

# ── Output helpers ──
# Consistent column layout: 2-space indent, 15-char label (fits llama-quantize), then value.
# Usage: step <label> <message> [color]   (color defaults to C_OK)
# Usage: substep <message> [color]         (color defaults to C_DIM)
step()    { printf "  ${C_DIM}%-15.15s${C_RST}${3:-$C_OK}%s${C_RST}\n" "$1" "$2"; }
substep() { printf "  %-15s${2:-$C_DIM}%s${C_RST}\n" "" "$1"; }

setup_fail() {
    local exit_code=$1
    shift
    [ "$exit_code" -ne 0 ] || exit_code=1
    local message
    message=$(printf '%s' "$*" | tr '\r\n' '  ')
    # Match setup.ps1: update.rs sets UNSLOTH_TAURI_UPDATE everywhere and promotes
    # this line over the generic "Update exited with code N". Test each variable
    # separately: one joined subject lets a comma in either value alias the other arm.
    local tauri_marker=0
    case "${UNSLOTH_TAURI_MODE:-0}" in 1|true) tauri_marker=1 ;; esac
    case "${UNSLOTH_TAURI_UPDATE:-0}" in 1|true) tauri_marker=1 ;; esac
    if [ "$tauri_marker" -eq 1 ]; then printf '[TAURI:ERROR] %s\n' "$message"; fi
    exit "$exit_code"
}

# ── Helper: can the controlling terminal actually be opened for reading? ──
# `test -r` only checks permission bits, which look fine in containers and
# systemd units where open() then fails with ENXIO. Probe with a real open.
# Mirrors install.sh's _can_read_tty; defined here too because setup.sh runs
# as its own process (install.sh invokes it, it does not source it).
_can_read_tty() {
    ( : </dev/tty ) >/dev/null 2>&1
}

_is_verbose() {
    [ "${UNSLOTH_VERBOSE:-0}" = "1" ]
}

verbose_substep() {
    if _is_verbose; then
        substep "$1"
    fi
    return 0
}

_remove_agent_instruction_files() {
    local _root
    for _root in "$@"; do
        [ -d "$_root" ] || continue
        [ -L "$_root" ] && continue
        find "$_root" \( -type f -o -type l \) \( -name 'AGENTS.md' -o -name 'CLAUDE.md' \) \
            -exec rm -f {} + 2>/dev/null || true
    done
}

# ── Corporate-mirror / proxy escape hatch for the frontend npm/bun install (#6491) ──
# studio/frontend/.npmrc pins registry=https://registry.npmjs.org/ as a supply-chain
# lock. A project-level pin overrides a corporate user's ~/.npmrc proxy, so the install
# hits npmjs.org directly and a firewall returns 403. UNSLOTH_NPM_REGISTRY is a
# deliberate opt-in: when set we thread it as `--registry <url>` into every npm/bun
# install. `--registry` is the highest-precedence override for BOTH tools and leaves
# min-release-age / save-exact in force. Empty array (the default) expands to nothing
# under `set -u`, so normal installs are unchanged.
_NPM_REGISTRY_ARGS=()
if [ -n "${UNSLOTH_NPM_REGISTRY:-}" ]; then
    _NPM_REGISTRY_ARGS=(--registry "$UNSLOTH_NPM_REGISTRY")
fi
# Failure-path capture log consumed by _suggest_npm_registry. Set to a temp file
# around the npm/bun installs; "" elsewhere so unrelated run_quiet calls don't capture.
_CAPTURE_LOG=""

# Print actionable guidance when a frontend/OXC npm/bun install fails and the registry
# lock is the likely cause (corporate firewall/proxy). No-op once the user has opted in
# via UNSLOTH_NPM_REGISTRY. We never switch registries automatically -- we only guide.
# $1 = path to a captured install log (may be empty/missing).
_suggest_npm_registry() {
    [ -n "${UNSLOTH_NPM_REGISTRY:-}" ] && return 0
    local _log="${1:-}"
    # If we captured output and it does NOT look like a registry/network problem, stay
    # quiet -- the raw error already shown is more useful than a misleading hint.
    if [ -n "$_log" ] && [ -s "$_log" ] \
        && ! grep -Eqi '40[13]|ENOTFOUND|ECONNREFUSED|ECONNRESET|ETIMEDOUT|EAI_AGAIN|ConnectionRefused|failed to resolve|registry\.npmjs\.org|getaddrinfo|tunneling socket|network|proxy|self.?signed|unable to (get|verify)' "$_log"; then
        return 0
    fi
    # Best-effort: surface a mirror the user already configured (env or ~/.npmrc).
    # Read npm config from / (a dir with no project .npmrc) so the frontend's pinned
    # registry= does not mask the user's ~/.npmrc / global mirror -- the caller is
    # still inside studio/frontend when this runs.
    local _mirror="${NPM_CONFIG_REGISTRY:-${npm_config_registry:-}}"
    if [ -z "$_mirror" ] && command -v npm >/dev/null 2>&1; then
        _mirror="$( (cd / 2>/dev/null && npm config get registry) 2>/dev/null || true )"
    fi
    case "$_mirror" in
        ""|undefined|null|https://registry.npmjs.org|https://registry.npmjs.org/) _mirror="" ;;
    esac
    printf '\n' >&2
    step "frontend" "registry.npmjs.org looks blocked (corporate firewall/proxy?)" "$C_WARN" >&2
    if [ -n "$_mirror" ]; then
        substep "Unsloth pins the public npm registry; your mirror is being ignored." >&2
        substep "Detected a registry in your npm config:" >&2
        substep "  $_mirror" >&2
        substep "Re-run pointing Unsloth at it:" >&2
        substep "  UNSLOTH_NPM_REGISTRY=$_mirror ./install.sh --local" >&2
    else
        substep "If you use a private mirror/proxy, point Unsloth at it and re-run:" >&2
        substep "  UNSLOTH_NPM_REGISTRY=https://your-mirror.example/api/npm/ ./install.sh --local" >&2
    fi
    substep "(min-release-age and save-exact stay enforced.)" >&2
    return 0
}

run_maybe_quiet() {
    if _is_verbose; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

# ── Helper: run command quietly, show output only on failure ──
_run_quiet() {
    local on_fail=$1
    local label=$2
    shift 2

    if _is_verbose; then
        local exit_code
        "$@" && return 0
        exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR" >&2
        if [ "$on_fail" = "exit" ]; then
            setup_fail "$exit_code" "$label failed (exit code $exit_code)"
        else
            return "$exit_code"
        fi
    fi

    local tmplog
    tmplog=$(mktemp) || {
        step "error" "Failed to create temporary file" "$C_ERR" >&2
        if [ "$on_fail" = "exit" ]; then
            setup_fail 1 "Failed to create temporary file for $label"
        fi
        return 1
    }

    if "$@" >"$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        step "error" "$label failed (exit code $exit_code)" "$C_ERR" >&2
        cat "$tmplog" >&2
        if [ -n "${_CAPTURE_LOG:-}" ]; then cat "$tmplog" >> "$_CAPTURE_LOG" 2>/dev/null || true; fi
        rm -f "$tmplog"

        if [ "$on_fail" = "exit" ]; then
            setup_fail "$exit_code" "$label failed (exit code $exit_code)"
        else
            return "$exit_code"
        fi
    fi
}

run_quiet() {
    _run_quiet exit "$@"
}

run_quiet_no_exit() {
    _run_quiet return "$@"
}

_nvcc_meets_llama_minimum() {
    # Echo "ok|too_old|unknown" then the parsed "X.Y" version, one per line.
    # llama.cpp needs CUDA toolkit >= 12.4 (#4437; setup.ps1 aborts via #4517).
    _nvcc_bin=$1
    [ -n "$_nvcc_bin" ] || { echo "unknown"; echo ""; return 0; }
    _raw=$("$_nvcc_bin" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
        | head -1)
    if [ -z "$_raw" ]; then
        echo "unknown"; echo ""; return 0
    fi
    _maj=${_raw%%.*}
    _min_raw=${_raw#*.}
    _min=${_min_raw%%.*}
    if [ "$_maj" -lt 12 ] 2>/dev/null; then
        echo "too_old"
    elif [ "$_maj" -eq 12 ] && [ "$_min" -lt 4 ] 2>/dev/null; then
        echo "too_old"
    else
        echo "ok"
    fi
    echo "$_raw"
}

# Echo a ';'-separated CUDA arch list (e.g. "86;120"). Override ($2,
# UNSLOTH_LLAMA_CUDA_ARCHS) wins verbatim; else parse+dedupe compute_cap text
# ($1). Empty means "no arch detected", so the caller builds CPU instead of a
# PTX-only binary that fails on an old driver (#5854).
_resolve_cuda_archs() {
    local _raw_caps=$1
    local _arch_override=$2
    if [ -n "$_arch_override" ]; then
        printf '%s' "$_arch_override"
        return 0
    fi
    local _archs="" _cap _arch
    while IFS= read -r _cap; do
        _cap=$(printf '%s' "$_cap" | tr -d '[:space:]')
        if [[ "$_cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            _arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            case ";$_archs;" in
                *";$_arch;"*) ;;
                *) _archs="${_archs:+$_archs;}$_arch" ;;
            esac
        fi
    done <<< "$_raw_caps"
    printf '%s' "$_archs"
}

# Reserved for the OS, and budgeted per compile job, both in MiB. Measured on
# llama.cpp with CUDA 13.1, the heaviest translation units (flash-attention
# template instances) peak at ~400 MiB RSS each, and -j20 peaked at 8.2 GiB
# across ~30 processes, since nvcc forks cicc and ptxas. 2048 is deliberately
# above that: it must also cover MSVC and hipcc, older and far heavier CUDA
# toolkits (ggml-org/llama.cpp#17844 climbs past 16 GiB), and the link step.
# Erring high costs build time; erring low costs the machine.
_LLAMA_BUILD_RESERVE_MB=2048
_LLAMA_BUILD_MB_PER_JOB=2048

# Echo the cmake -j count. Args: core count, total RAM MiB ("" = unreadable,
# which keeps the old core-count behaviour). UNSLOTH_LLAMA_BUILD_JOBS wins.
# Pure, so the tests can drive it without faking hardware.
_llama_jobs_for() {
    local _cores=$1 _mem_mb=$2 _jobs
    if [[ "${UNSLOTH_LLAMA_BUILD_JOBS:-}" =~ ^[0-9]+$ ]] && [ "$UNSLOTH_LLAMA_BUILD_JOBS" -ge 1 ]; then
        printf '%s' "$UNSLOTH_LLAMA_BUILD_JOBS"
        return 0
    fi
    if ! [[ "$_cores" =~ ^[0-9]+$ ]] || [ "$_cores" -lt 1 ]; then _cores=4; fi
    if ! [[ "$_mem_mb" =~ ^[0-9]+$ ]]; then
        printf '%s' "$_cores"
        return 0
    fi
    _jobs=$(( (_mem_mb - _LLAMA_BUILD_RESERVE_MB) / _LLAMA_BUILD_MB_PER_JOB ))
    [ "$_jobs" -lt 1 ] && _jobs=1
    [ "$_jobs" -gt "$_cores" ] && _jobs=$_cores
    printf '%s' "$_jobs"
}

# Echo the first line of $1 with whitespace stripped, or nothing. A builtin read
# plus an expansion, not `head | tr`: this is best-effort on the install's
# critical path under `set -euo pipefail`, and a failing pipeline took the whole
# install down. -r alone does not rule that out (a directory passes it, and a
# cgroup can be torn down between the test and the open); -f does.
_cg_read() {
    local _v=""
    [ -f "$1" ] && [ -r "$1" ] || return 0
    IFS= read -r _v < "$1" 2>/dev/null || true
    printf '%s' "${_v//[[:space:]]/}"
    return 0
}

# Echo $1 when it is a real limit. v2 writes "max"; v1 a near-2^63 sentinel.
# Zero is a limit, not the absence of one: memory.high throttles and never
# invokes the OOM killer (cgroup-v2.rst), so a process runs happily under
# MemoryHigh=0 and must not then be handed its full core count.
_cg_limit() {
    [[ "$1" =~ ^[0-9]+$ ]] || return 0
    [ "$1" -lt 4611686018427387904 ] && printf '%s' "$1"
    return 0
}

# Echo "$1/$2" and every ancestor up to $1, innermost first, NUL-delimited: a
# mount point may contain a newline, so a line-delimited list would split one
# path into two. dirname is inlined for the same reason ($() eats a trailing
# newline).
_cg_dirs() {
    local _root=$1 _cur
    if [ -n "${2:-}" ] && [ "$2" != "/" ]; then
        _cur="$_root/${2#/}"
        while [ "$_cur" != "$_root" ] && [ "$_cur" != "/" ]; do
            printf '%s\0' "$_cur"
            case "$_cur" in
                */*) _cur=${_cur%/*}; [ -n "$_cur" ] || _cur="/" ;;
                *) break ;;
            esac
        done
    fi
    printf '%s\0' "$_root"
}

# The mountinfo path decoder as an awk function, in one place so its two callers
# cannot drift. mountinfo escapes space, tab, newline and backslash as \040 and
# friends; strtonum is a gawk extension, so the octal maths is done by hand to
# also run under the mawk and BSD awk that Debian and macOS ship.
_cg_unesc_prog() {
    cat <<'_CG_AWK_UNESC'
        function unesc(s,   out, i, c, o, v) {
            out = ""; i = 1
            while (i <= length(s)) {
                c = substr(s, i, 1)
                if (c == "\\" && substr(s, i + 1, 3) ~ /^[0-7][0-7][0-7]$/) {
                    o = substr(s, i + 1, 3)
                    v = (substr(o, 1, 1) + 0) * 64 + (substr(o, 2, 1) + 0) * 8 + (substr(o, 3, 1) + 0)
                    out = out sprintf("%c", v); i += 4
                } else { out = out c; i++ }
            }
            return out
        }
_CG_AWK_UNESC
}

# Echo $1 with its mountinfo escapes decoded. Decoding is deliberately late:
# \011 and \012 decode to the very tab and newline that delimit the records
# below, so decoding at the read would split one record into two. An escaped
# path holds neither, so it travels intact and is decoded here.
_cg_unesc() {
    [ -n "$1" ] || return 0
    printf '%s\n' "$1" | awk "$(_cg_unesc_prog)"'{ printf "%s", unesc($0); exit }' || true
    return 0
}

# Echo every matching cgroup hierarchy as "<mount root><tab><mount point>", both
# still escaped. $1 = a mountinfo file, $2 = "cgroup2" or a v1 controller name.
# mountinfo is "... <root> <mountpoint> <opts> [tags] - <fstype> <source>
# <superopts>", and a v1 hierarchy lists its controllers in the super options,
# so a co-mounted or relocated one is found by name rather than assumed to sit
# at <root>/<name>.
_cg_mounts() {
    [ -r "$1" ] || return 0
    awk -v want="$2" '
        {
            for (i = 1; i <= NF; i++) if ($i == "-") break
            if (i + 3 > NF) next
            if (want == "cgroup2") {
                if ($(i + 1) == "cgroup2") print $4 "\t" $5
                next
            }
            if ($(i + 1) != "cgroup") next
            n = split($(i + 3), opts, ",")
            for (j = 1; j <= n; j++) if (opts[j] == want) { print $4 "\t" $5; next }
        }' "$1" 2>/dev/null || true
    return 0
}

# Echo the process's path within a mount, or nothing when the mount does not
# expose it. Args: mount root, process cgroup path. A bind-mounted subtree shows
# a mount root like /slice while /proc/self/cgroup reports /slice/job and the
# files sit at <mountpoint>/job, so joining the two unmapped walks a path that
# does not exist and settles on an outer limit instead of the binding one.
_cg_rel() {
    local _root=$1 _rel=$2
    [ -n "$_rel" ] || return 0
    [ "$_root" = "/" ] && { printf '%s' "$_rel"; return 0; }
    case "$_rel" in
        "$_root") printf '%s' "/" ;;
        "$_root"/*) printf '%s' "${_rel#"$_root"}" ;;
        # Not under this mount's root: nothing here describes this process.
        *) : ;;
    esac
    return 0
}

# Echo EVERY "<root><tab><point>" on stdin whose root contains the process path
# ($1), still escaped, one per line; or the first seen when none does. A
# hierarchy can be mounted twice with different subtree roots (rootless podman
# inside rootless podman): taking the first steps past the binding mount, and
# taking only the most specific hides a limit above the narrower mount's root.
# So every containing mount is inspected and the smallest allowance wins, which
# makes the answer order-independent.
_cg_pick_mounts() {
    local _rel=$1 _root _point _droot _any="" _firstroot="" _firstpoint=""
    while IFS=$'\t' read -r _root _point; do
        [ -n "$_point" ] || continue
        if [ -z "$_firstpoint" ]; then _firstroot=$_root; _firstpoint=$_point; fi
        # /proc/self/cgroup is not escaped, so the root is compared decoded.
        _droot=$(_cg_unesc "$_root")
        [ -n "$(_cg_rel "$_droot" "$_rel")" ] || continue
        printf '%s\t%s\n' "$_root" "$_point"
        _any=1
    done
    [ -n "$_any" ] || [ -z "$_firstpoint" ] || printf '%s\t%s\n' "$_firstroot" "$_firstpoint"
    return 0
}

# Echo the memory free under the binding cgroup limit in MiB, or nothing. Args:
# fallback cgroup root, /proc/self/cgroup path, /proc/self/mountinfo path; all
# arguments so the tests can drive a real tree. Mirrors dataset_num_proc.py:
# reading the hierarchy root alone only works with a private cgroup namespace,
# and under Slurm, systemd or --cgroupns=host the binding limit is the process's
# own path or an ancestor's. Each limit pairs with the usage of the directory
# that set it, since an ancestor's usage counts siblings this process cannot
# see, and the smallest remaining allowance wins.
_cgroup_free_mb() {
    local _root=$1 _proc=$2 _mnt=${3:-} _rel _dir _used _limit _free _name _min=""
    local _v2rel _v1rel _v2mnts _v1mnts _mroot _mpoint
    _cg_consider() {
        [ -n "$1" ] || return 0
        if [[ "$2" =~ ^[0-9]+$ ]]; then _free=$(( $1 - $2 )); else _free=$1; fi
        [ "$_free" -lt 0 ] && _free=0
        if [ -z "$_min" ] || [ "$_free" -lt "$_min" ]; then _min=$_free; fi
        return 0
    }
    # The process path is read first so the right mount can be chosen among
    # several. Only the first two colons are delimiters: a systemd unit name may
    # contain one, and -F: with $3 would truncate the path there.
    _v2rel=$(awk '/^0::/ { print substr($0, 4); exit }' "$_proc" 2>/dev/null || true)
    _v1rel=$(awk '
        {
            a = index($0, ":"); if (a == 0) next
            rest = substr($0, a + 1)
            b = index(rest, ":"); if (b == 0) next
            if (substr(rest, 1, b - 1) ~ /(^|,)memory(,|$)/) { print substr(rest, b + 1); exit }
        }' "$_proc" 2>/dev/null || true)
    _v2mnts=$(_cg_mounts "$_mnt" cgroup2 | _cg_pick_mounts "$_v2rel")
    _v1mnts=$(_cg_mounts "$_mnt" memory | _cg_pick_mounts "$_v1rel")
    # Fall back to the conventional layout when mountinfo is unreadable.
    [ -n "$_v2mnts" ] || _v2mnts=$(printf '/\t%s' "$_root")
    [ -n "$_v1mnts" ] || _v1mnts=$(printf '/\t%s' "$_root/memory")

    # The v2 line is "0::<path>"; systemd hybrid mode adds v1 lines alongside.
    while IFS=$'\t' read -r _mroot _mpoint; do
        [ -n "$_mpoint" ] || continue
        # Decode now that a single path is in hand, after it survived the tab-
        # and newline-delimited transport. The sentinel keeps $() from eating a
        # trailing newline.
        _mroot=$(_cg_unesc "$_mroot"; printf X); _mroot=${_mroot%X}
        _mpoint=$(_cg_unesc "$_mpoint"; printf X); _mpoint=${_mpoint%X}
        [ -d "$_mpoint" ] || continue
        _rel=$(_cg_rel "$_mroot" "$_v2rel")
        while IFS= read -r -d '' _dir; do
            _used=$(_cg_read "$_dir/memory.current")
            for _name in memory.max memory.high; do
                _limit=$(_cg_limit "$(_cg_read "$_dir/$_name")")
                _cg_consider "$_limit" "$_used"
            done
        done < <(_cg_dirs "$_mpoint" "$_rel")
    done <<< "$_v2mnts"

    # v1 is "<id>:<controllers>:<path>", and mounts are often combined.
    while IFS=$'\t' read -r _mroot _mpoint; do
        [ -n "$_mpoint" ] || continue
        _mroot=$(_cg_unesc "$_mroot"; printf X); _mroot=${_mroot%X}
        _mpoint=$(_cg_unesc "$_mpoint"; printf X); _mpoint=${_mpoint%X}
        [ -d "$_mpoint" ] || continue
        _rel=$(_cg_rel "$_mroot" "$_v1rel")
        while IFS= read -r -d '' _dir; do
            _used=$(_cg_read "$_dir/memory.usage_in_bytes")
            _limit=$(_cg_limit "$(_cg_read "$_dir/memory.limit_in_bytes")")
            _cg_consider "$_limit" "$_used"
        done < <(_cg_dirs "$_mpoint" "$_rel")
    done <<< "$_v1mnts"

    [ -n "$_min" ] && printf '%d' "$(( _min / 1048576 ))"
    return 0
}

# macOS available memory in MiB, read from vm_stat on stdin; empty when the
# output does not parse. free + inactive is the reclaim-aware equivalent of
# MemAvailable, and the page size is read from the header rather than assumed to
# be 4096, which is wrong on Apple Silicon. Only those two: xnu's
# osfmk/mach/vm_statistics.h says speculative pages "are already accounted for
# in free_count", and purgeable is an attribute of a page rather than a queue,
# so a volatile page is already counted on active or inactive. Adding either
# double-counts. Under-counting is the safe direction for a cap: it costs build
# time on a busy Mac rather than the machine.
_vm_stat_avail_mb() {
    awk '
        /page size of/ {
            for (i = 1; i < NF; i++) if ($i == "of") { ps = $(i + 1) + 0; break }
        }
        /^Pages (free|inactive)/ {
            gsub(/\./, "", $NF); pages += $NF
        }
        # A zero page count is a reading; only a missing page size is a failure.
        END { if (ps > 0) printf "%d", pages * ps / 1048576 }' || true
    return 0
}

# Usable RAM in MiB; empty when it cannot be read. MemAvailable, not MemTotal: a
# workstation with 8 GiB already resident cannot host a 14 GiB compile just
# because 16 GiB is fitted. /proc is not namespaced either, so a lower cgroup
# allowance wins. $1 is the meminfo file, an argument like every other reader's
# path here so the tests can pin a number instead of racing the live one.
_usable_ram_mb() {
    local _meminfo=${1:-/proc/meminfo} _bytes _mb="" _free _avail
    if [ -r "$_meminfo" ]; then
        # MemAvailable counts reclaimable page cache; MemFree does not. Absent
        # before Linux 3.14, where MemTotal is the only thing to go on. `|| true`
        # because bash applies errexit to a failing assignment in POSIX mode, and
        # an unreadable meminfo must cost the cap, not the install.
        _mb=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1024; exit }' "$_meminfo") || true
        [ -n "$_mb" ] || _mb=$(awk '/^MemTotal:/ { printf "%d", $2 / 1024; exit }' "$_meminfo") || true
    elif _bytes=$(sysctl -n hw.memsize 2>/dev/null); then
        [[ "$_bytes" =~ ^[0-9]+$ ]] && _mb=$(( _bytes / 1048576 ))
        # macOS has no MemAvailable, so hw.memsize is installed RAM and would
        # hand a busy Mac a budget it cannot honour; vm_stat is the equivalent,
        # with installed RAM as the fallback when it does not parse. Zero is
        # kept: a Mac with nothing reclaimable should build at 1 job, not take
        # its full core count.
        _avail=$(vm_stat 2>/dev/null | _vm_stat_avail_mb || true)
        if [[ "$_avail" =~ ^[0-9]+$ ]]; then _mb=$_avail; fi
    fi
    _free=$(_cgroup_free_mb /sys/fs/cgroup /proc/self/cgroup /proc/self/mountinfo)
    if [[ "$_free" =~ ^[0-9]+$ ]]; then
        if [ -z "$_mb" ] || [ "$_free" -lt "$_mb" ]; then _mb=$_free; fi
    fi
    printf '%s' "$_mb"
    return 0
}

_llama_build_jobs() {
    _llama_jobs_for \
        "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
        "$(_usable_ram_mb)"
}

# Opt-in staged GPU smoke test after a source build (#5854 gap 2). Default off:
# llama-server's first GPU forward pass JIT-compiles CUDA kernels and stalls
# installs for minutes on Blackwell. Same env as install_llama_prebuilt.py.
_staged_validation_enabled() {
    local _raw="${UNSLOTH_LLAMA_STAGED_VALIDATION:-}"
    # Match install_llama_prebuilt.py staged_validation_enabled(): strip + lowercase.
    _raw="$(printf '%s' "$_raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')"
    case "$_raw" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Map the source-build GPU backend to install_llama_prebuilt --install-kind so
# validate_server enables --n-gpu-layers for the right backends.
_source_smoke_install_kind() {
    if [ "${_TRY_METAL_CPU_FALLBACK:-false}" = true ]; then
        printf '%s' "macos-arm64"
        return 0
    fi
    case "${GPU_BACKEND:-}" in
        cuda)
            case "$(uname -m 2>/dev/null || true)" in
                aarch64|arm64) printf '%s' "linux-arm64-cuda" ;;
                *) printf '%s' "linux-cuda" ;;
            esac
            ;;
        rocm) printf '%s' "linux-rocm" ;;
        *) printf '%s' "" ;;
    esac
}

# Run a GPU probe under a 10s timeout when `timeout` is available so a wedged
# NVIDIA driver cannot hang setup; fall back to a bare call where it is not.
_setup_run_smi() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 10 "$@"
    else
        "$@"
    fi
}

# Returns 0 when CUDA_VISIBLE_DEVICES is set to "" or "-1", i.e. every NVIDIA
# device is deliberately hidden (mixed AMD+NVIDIA hosts steering work to the
# AMD card). Unset means all devices visible. nvidia-smi ignores this env var,
# so the probes below cannot see the distinction on their own.
_setup_cvd_hides_nvidia() {
    [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] || return 1
    _setup_cvd_trim=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]')
    [ -z "$_setup_cvd_trim" ] || [ "$_setup_cvd_trim" = "-1" ]
}

# Returns 0 when an NVIDIA GPU is present and usable. Primary probe is
# `nvidia-smi -L` (timeout-bounded). Fallback is /proc/driver/nvidia/gpus,
# which the driver populates per GPU regardless of nvidia-smi state -- handles
# PATH gaps and driver init races. Mirrors install.sh _has_usable_nvidia_gpu
# (PR 6174) so setup routes the same way as the torch installer. A GPU hidden
# via CUDA_VISIBLE_DEVICES=""/-1 counts as NOT usable (matches
# install_llama_prebuilt.py has_usable_nvidia), so the AMD probes still run
# and a mixed host steered to its AMD card keeps the ROCm route.
_setup_has_usable_nvidia_gpu() {
    if _setup_cvd_hides_nvidia; then
        return 1
    fi
    _setup_nvsmi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _setup_nvsmi="nvidia-smi"
    elif [ -x "/usr/bin/nvidia-smi" ]; then
        _setup_nvsmi="/usr/bin/nvidia-smi"
    fi
    if [ -n "$_setup_nvsmi" ]; then
        if _setup_run_smi "$_setup_nvsmi" -L 2>/dev/null \
           | awk '/^GPU[[:space:]]+[0-9]+:/{found=1} END{exit !found}'; then
            return 0
        fi
    fi
    if [ -d /proc/driver/nvidia/gpus ] && \
       [ -n "$(ls -A /proc/driver/nvidia/gpus 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

_cuda_driver_max_version() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    _setup_run_smi nvidia-smi 2>/dev/null \
        | sed -nE 's/.*CUDA( UMD)? Version:[[:space:]]*([0-9]+)\.([0-9]+).*/\2.\3/p' \
        | head -1 || true
}

_cuda_version_gt() {
    local _left=${1:-}
    local _right=${2:-}
    if ! [[ "$_left" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _left_major=$((10#${BASH_REMATCH[1]}))
    local _left_minor=$((10#${BASH_REMATCH[2]}))
    if ! [[ "$_right" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _right_major=$((10#${BASH_REMATCH[1]}))
    local _right_minor=$((10#${BASH_REMATCH[2]}))

    if [ "$_left_major" -gt "$_right_major" ]; then
        return 0
    fi
    if [ "$_left_major" -eq "$_right_major" ] && [ "$_left_minor" -gt "$_right_minor" ]; then
        return 0
    fi
    return 1
}

_cuda_toolkit_major_gt_driver() {
    local _toolkit_version=${1:-}
    local _driver_version=${2:-}
    if ! [[ "$_toolkit_version" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _toolkit_major=$((10#${BASH_REMATCH[1]}))
    if ! [[ "$_driver_version" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    local _driver_major=$((10#${BASH_REMATCH[1]}))
    [ "$_toolkit_major" -gt "$_driver_major" ]
}

_cuda_nvcc_candidate_paths() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        printf '%s\n' "/usr/local/cuda/bin/nvcc"
    fi
    ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V -r 2>/dev/null || true
}

_cuda_find_compatible_nvcc_for_driver() {
    local _driver_version=$1
    local _exclude_path=${2:-}
    local _candidate _seen _check _status _version
    local _best_path="" _best_version=""
    _seen="
"
    while IFS= read -r _candidate; do
        [ -n "$_candidate" ] || continue
        [ "$_candidate" != "$_exclude_path" ] || continue
        [ -x "$_candidate" ] || continue
        case "$_seen" in
            *"
$_candidate
"*) continue ;;
        esac
        _seen="${_seen}${_candidate}
"
        _check="$(_nvcc_meets_llama_minimum "$_candidate")"
        _status="$(printf '%s\n' "$_check" | sed -n '1p')"
        _version="$(printf '%s\n' "$_check" | sed -n '2p')"
        [ "$_status" = "ok" ] || continue
        [ -n "$_version" ] || continue
        if _cuda_toolkit_major_gt_driver "$_version" "$_driver_version"; then
            continue
        fi
        if [ -z "$_best_version" ] || _cuda_version_gt "$_version" "$_best_version"; then
            _best_path="$_candidate"
            _best_version="$_version"
        fi
    done <<EOF
$(_cuda_nvcc_candidate_paths)
EOF
    [ -n "$_best_path" ] || return 1
    printf '%s\n%s\n' "$_best_path" "$_best_version"
}

_print_cuda_driver_toolkit_mismatch() {
    local _toolkit_version=$1
    local _driver_version=$2
    local _toolkit_major=${_toolkit_version%%.*}
    local _driver_major=${_driver_version%%.*}
    substep "CUDA Toolkit $_toolkit_version is a major-version mismatch: toolkit major $_toolkit_major exceeds driver CUDA major $_driver_major ($_driver_version)." "$C_WARN"
    substep "Update the NVIDIA GPU driver to run CUDA Toolkit $_toolkit_version, or install a CUDA $_driver_major.x toolkit." "$C_WARN"
    substep "Or let Unsloth use the prebuilt CUDA bundle; it does not need the local toolkit." "$C_WARN"
}

print_llama_error_log() {
    local log_file=$1
    [ -s "$log_file" ] || return 0
    substep "llama.cpp diagnostics (last 120 lines):"
    tail -n 120 "$log_file" | sed 's/^/   | /' >&2
}

installed_llama_prebuilt_release() {
    local install_dir=${1:-}
    local metadata_path="$install_dir/UNSLOTH_PREBUILT_INFO.json"
    [ -f "$metadata_path" ] || return 0
    python - "$metadata_path" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

if not isinstance(payload, dict):
    raise SystemExit(0)

repo = str(payload.get("published_repo") or "").strip()
release_tag = str(payload.get("release_tag") or "").strip()
llama_tag = str(payload.get("tag") or "").strip()
source = str(payload.get("source") or "").strip()
binary_repo = str(payload.get("binary_repo") or "").strip()
binary_tag = str(payload.get("binary_release_tag") or "").strip()
if not repo or not release_tag:
    raise SystemExit(0)

# For non-fork sources (e.g. ggml-org upstream prebuilts) the published_repo/
# release_tag refer to the unsloth source tree while the actual binaries came
# from a different repo. Show both so the log is unambiguous.
if source and source != "upstream" and binary_repo and binary_tag and binary_repo != repo:
    message = f"installed release: {repo}@{release_tag} + {source}@{binary_tag}"
else:
    message = f"installed release: {repo}@{release_tag}"
    if llama_tag and llama_tag != release_tag:
        message += f" (tag {llama_tag})"
print(message)
PY
}

print_installed_llama_prebuilt_release() {
    local install_dir=${1:-}
    local installed_release
    installed_release="$(installed_llama_prebuilt_release "$install_dir")"
    if [ -n "$installed_release" ]; then
        substep "$installed_release"
    fi
}

# ── Banner ──
echo ""
printf "  ${C_TITLE}%s${C_RST}\n" "🦥 Unsloth Studio Setup"
printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
verbose_substep "verbose diagnostics enabled"
_LLAMA_ONLY="${UNSLOTH_STUDIO_LLAMA_ONLY:-0}"
if [ "$_LLAMA_ONLY" = "1" ]; then
    substep "llama.cpp only mode"
fi
if [ "${STUDIO_LOCAL_INSTALL:-0}" = "1" ]; then
    substep "local mode: overlaying $REPO_ROOT (editable) + unsloth-zoo from git main"
fi
# ── Clean up stale caches ──
rm -rf "$REPO_ROOT/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/backend/unsloth_compiled_cache"
rm -rf "$SCRIPT_DIR/tmp/unsloth_compiled_cache"

# WebView caches keyed by the bundle id can keep serving the previous frontend
# after an update. Cache-only: storage, cookies, settings, models and the studio
# database are untouched.
_clear_webview_caches() {
    # No HOME: bail rather than let `set -u` abort or "" expand to /Library and /.cache.
    [ -n "${HOME:-}" ] || return 0
    _wvc_bid="ai.unsloth.studio"
    _wvc_paths=()
    # The app's version stamp dir (main.rs webview_profile_root), which on macOS
    # is not the cache dir, so it is tracked separately.
    _wvc_root=""
    case "$(uname -s 2>/dev/null)" in
        Darwin)
            # WKWebView keeps every cache-typed store under Library/Caches/<bid>;
            # Library/WebKit/<bid> is user storage and is left alone.
            _wvc_paths=("$HOME/Library/Caches/$_wvc_bid")
            _wvc_root="$HOME/Library/Application Support/$_wvc_bid"
            ;;
        Linux)
            # wry points WebKitGTK's base-cache dir at the app data dir, so the
            # caches sit beside localstorage/, databases/ and cookies, which stay.
            # A relative XDG_DATA_HOME is invalid per the XDG spec and dropped by
            # dirs, so match Tauri and use the default rather than rm -rf under
            # whatever directory the installer runs from.
            _wvc_data="${XDG_DATA_HOME:-$HOME/.local/share}"
            case "$_wvc_data" in /*) ;; *) _wvc_data="$HOME/.local/share" ;; esac
            _wvc_data="$_wvc_data/$_wvc_bid"
            _wvc_paths=(
                "$_wvc_data/WebKitCache"
                "$_wvc_data/CacheStorage"
                "$_wvc_data/serviceworkers"
            )
            _wvc_root="$_wvc_data"
            ;;
        *) return 0 ;;
    esac
    # Drop the version stamp first. An update runs while the old WebView still holds
    # these files, so the rm below can fail; the app's own clear is the retry, and it
    # is skipped while the stamp matches the running version. Unconditional, since a
    # repair or a local rebuild leaves the version unchanged and a redundant clear on
    # the next launch is the cheap side. An `if`, not `[ ... ] && rm`: under `set -e`
    # that compound returns 1 when the guard is false and would abort the install.
    if [ -n "$_wvc_root" ]; then
        rm -f "$_wvc_root/.webview-cache-cleared" 2>/dev/null || true
    fi
    _wvc_cleared=false
    for _wvc_p in "${_wvc_paths[@]}"; do
        # -L too: a dangling symlink still occupies the path.
        [ -e "$_wvc_p" ] || [ -L "$_wvc_p" ] || continue
        rm -rf "$_wvc_p" 2>/dev/null && _wvc_cleared=true || true
    done
    if [ "$_wvc_cleared" = true ]; then
        substep "cleared stale WebView caches ($_wvc_bid); settings and data kept"
    fi
    return 0
}
# Not called here: under `set -e` with no trap, clearing before the UNSLOTH_STUDIO_HOME
# / STUDIO_HOME override is validated lets a typo'd override delete the cache, then abort.

# ── Detect Colab ──
IS_COLAB=false
keynames=$'\n'$(printenv | cut -d= -f1)
if [[ "$keynames" == *$'\nCOLAB_'* ]]; then
    IS_COLAB=true
fi

# Resolve studio home + ownership marker before the llama-only split: the
# llama.cpp section needs STUDIO_HOME / _STUDIO_HOME_IS_CUSTOM, but
# UNSLOTH_STUDIO_LLAMA_ONLY=1 ('unsloth studio update') skips the base install.
# UNSLOTH_STUDIO_HOME (or STUDIO_HOME alias) overrides the install root
# (mirrors install.sh). UNSLOTH_STUDIO_HOME wins when both are set.
_studio_override_var=""
_studio_override="${UNSLOTH_STUDIO_HOME:-}"
if [ -n "$_studio_override" ]; then
    _studio_override_var="UNSLOTH_STUDIO_HOME"
else
    _studio_override="${STUDIO_HOME:-}"
    [ -n "$_studio_override" ] && _studio_override_var="STUDIO_HOME"
fi
# Strip whitespace so " " is treated as unset (matches Python .strip()).
_studio_override=$(printf '%s' "$_studio_override" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
case "$_studio_override" in
    "~") _studio_override="$HOME" ;;
    "~/"*) _studio_override="$HOME/${_studio_override#'~/'}" ;;
esac
if [ -n "$_studio_override" ]; then
    # setup.sh runs against an existing install (via 'unsloth studio update');
    # a typo in the override must fail fast instead of materializing an
    # empty workspace dir. Mirrors setup.ps1 behavior.
    if [ ! -d "$_studio_override" ]; then
        echo "ERROR: $_studio_override_var=$_studio_override does not exist." >&2
        echo "       Run install.sh to create the install root before 'unsloth studio update'." >&2
        setup_fail 1 "$_studio_override_var=$_studio_override does not exist"
    fi
    if [ ! -w "$_studio_override" ]; then
        echo "ERROR: $_studio_override_var=$_studio_override is not writable." >&2
        setup_fail 1 "$_studio_override_var=$_studio_override is not writable"
    fi
    STUDIO_HOME="$(CDPATH= cd -P -- "$_studio_override" && pwd -P)" ||
        setup_fail 1 "Could not resolve $_studio_override_var=$_studio_override"
else
    STUDIO_HOME="$HOME/.unsloth/studio"
fi

STAGE_ROOT="${UNSLOTH_STUDIO_STAGE_ROOT:-}"
RUNTIME_ROOT="${STAGE_ROOT:-$STUDIO_HOME}"
VENV_DIR="$RUNTIME_ROOT/unsloth_studio"
VENV_T5_530_DIR="$RUNTIME_ROOT/.venv_t5_530"
VENV_T5_550_DIR="$RUNTIME_ROOT/.venv_t5_550"
VENV_T5_510_DIR="$RUNTIME_ROOT/.venv_t5_510"

# The override is validated, so a typo can no longer cost the cache. Venv-gated because
# a writable-but-empty override still aborts at the venv check below, and clearing first
# would cost the cache for a run that then does nothing; a fresh install has neither venv
# nor cache. Still before any install work, while the old frontend is the one on disk.
if [ -z "$STAGE_ROOT" ] && [ -x "$VENV_DIR/bin/python" ]; then
    _clear_webview_caches
fi

_STUDIO_OWNED_MARKER=".unsloth-studio-owned"
_LEGACY_STUDIO_HOME="$HOME/.unsloth/studio"
_studio_home_canon="$STUDIO_HOME"
if [ -d "$_studio_home_canon" ]; then
    _studio_home_canon=$(CDPATH= cd -P -- "$_studio_home_canon" 2>/dev/null && pwd -P) \
        || _studio_home_canon="$STUDIO_HOME"
fi
if [ -d "$_LEGACY_STUDIO_HOME" ]; then
    _LEGACY_STUDIO_HOME=$(CDPATH= cd -P -- "$_LEGACY_STUDIO_HOME" 2>/dev/null && pwd -P) \
        || _LEGACY_STUDIO_HOME="$HOME/.unsloth/studio"
fi
_STUDIO_HOME_IS_CUSTOM=false
if [ "$_studio_home_canon" != "$_LEGACY_STUDIO_HOME" ]; then
    _STUDIO_HOME_IS_CUSTOM=true
fi
# Directory-local evidence Unsloth created "$1": only prebuilt-installer metadata
# counts (UNSLOTH_PREBUILT_INFO.json for llama.cpp, UNSLOTH_NODE_PREBUILT_INFO.json
# for Node, UNSLOTH_WHISPER_PREBUILT_INFO.json for whisper.cpp), all written only
# by our installers. Mirrors the setup.ps1 Node guard. A markerless source build
# stays strict since this runs right before an rm -rf.
_studio_owned_adoptable() {
    [ -f "$1/UNSLOTH_PREBUILT_INFO.json" ] && return 0
    [ -f "$1/UNSLOTH_NODE_PREBUILT_INFO.json" ] && return 0
    [ -f "$1/UNSLOTH_WHISPER_PREBUILT_INFO.json" ] && return 0
    return 1
}
# Marker probes need search (+x), not read (+r): in an unsearchable dir every probe reports absent, so our install looks foreign.
_studio_dir_unsearchable() {
    [ -d "$1" ] || return 1
    ( cd -- "$1" ) 2>/dev/null && return 1
    return 0
}

# Also needs +r for callers that list or replace the tree: mode 111 is searchable but still fails install_llama_prebuilt.py.
_studio_dir_unreadable() {
    [ -d "$1" ] || return 1
    _studio_dir_unsearchable "$1" && return 0
    ls -A -- "$1" >/dev/null 2>&1 && return 1
    return 0
}

# Mirrors Exit-PathAccessDenied in setup.ps1. owner-unverified means the marker
# was unreadable, so do not claim the tree is ours or advise deleting it.
_path_access_denied() {
    _pad_dir="$1"
    _pad_label="$2"
    _pad_mode="${3:-}"
    step "permissions" "$_pad_label at $_pad_dir cannot be read: permission denied" "$C_ERR"
    if [ "$_pad_mode" = "owner-unverified" ]; then
        substep "Unsloth cannot confirm this folder is its own install while it is unreadable, so it will not tell you to remove it" "$C_WARN"
        substep "Restore access, or move the folder aside, then re-run setup:" "$C_WARN"
    else
        substep "This folder lives outside the app, so reinstalling Unsloth Studio reuses it and fails the same way" "$C_WARN"
        substep "Simplest fix: delete or rename $_pad_dir, then re-run setup (it is a managed cache and gets reinstalled)" "$C_WARN"
        substep "If deleting is denied too, it belongs to another user; restore access with:" "$C_WARN"
    fi
    substep "ls -ld \"$_pad_dir\"" "$C_WARN"
    substep "chmod -R u+rwX \"$_pad_dir\"" "$C_WARN"
    if [ "$_pad_mode" = "owner-unverified" ]; then
        setup_fail 1 "Permission denied reading $_pad_label at $_pad_dir. Unsloth cannot confirm that folder is its own install while it is unreadable: restore access, or move it aside, then re-run setup."
    fi
    setup_fail 1 "Permission denied reading the existing $_pad_label at $_pad_dir. Delete or rename that folder (Unsloth reinstalls it) or restore access, then re-run setup. Reinstalling the app does not reset it."
}

# POSIX follows a final symlink when the path ends in /, so "link/" is never -L. Strip it, but never past the root.
_studio_rstrip_slash() {
    _srs_path="$1"
    while [ "$_srs_path" != "/" ] && [ "${_srs_path%/}" != "$_srs_path" ]; do
        _srs_path="${_srs_path%/}"
    done
    printf '%s' "$_srs_path"
}

# An unsearchable ancestor makes a real path read as missing. Walk up to the deepest
# ancestor we can stat and report it as the blocker; stay quiet if the path is just absent.
_report_denied_ancestor() {
    _rda_probe="$(_studio_rstrip_slash "$1")"
    _rda_hops=0
    while [ ! -e "$_rda_probe" ] && [ "$_rda_probe" != "/" ] && [ "$_rda_probe" != "." ]; do
        # An unfollowable symlink is the deepest name we have, so walk its target:
        # the denied ancestor lives there. The hop cap breaks symlink cycles.
        if [ -L "$_rda_probe" ] && [ "$_rda_hops" -lt 40 ]; then
            _rda_hops=$((_rda_hops + 1))
            _rda_target="$(readlink -- "$_rda_probe")" || break
            case "$_rda_target" in
                /*) _rda_probe="$_rda_target" ;;
                *) _rda_probe="$(dirname -- "$_rda_probe")/$_rda_target" ;;
            esac
            _rda_probe="$(_studio_rstrip_slash "$_rda_probe")"
            continue
        fi
        # -- keeps a leading-dash path an operand, not a dirname option.
        _rda_probe="$(dirname -- "$_rda_probe")"
    done
    if _studio_dir_unsearchable "$_rda_probe"; then
        _path_access_denied "$_rda_probe" "$2" owner-unverified
    fi
}

_assert_studio_owned_or_absent() {
    _aso_dir="$1"
    _aso_label="$2"
    [ -d "$_aso_dir" ] || return 0
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ ! -f "$_aso_dir/$_STUDIO_OWNED_MARKER" ]; then
        if _studio_owned_adoptable "$_aso_dir"; then
            : > "$_aso_dir/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
            return 0
        fi
        # An unsearchable tree hides its own marker and so looks like someone else's:
        # report permissions, not ownership.
        if _studio_dir_unsearchable "$_aso_dir"; then
            _path_access_denied "$_aso_dir" "$_aso_label" owner-unverified
        fi
        echo "ERROR: $_aso_dir already exists and is not marked as an Unsloth-owned $_aso_label." >&2
        echo "       Move it aside or choose an empty UNSLOTH_STUDIO_HOME before re-running." >&2
        setup_fail 1 "$_aso_label path is not an Unsloth-owned install: $_aso_dir"
    fi
}


_packaged_frontend_available() {
    # install.sh and `unsloth studio update` explicitly set 0 for PyPI installs.
    # Wheel extraction mtimes do not preserve build ordering, so source files can
    # appear newer than the release-built dist even though both came from the
    # same wheel. Trust the packaged artifact when its entry point is present;
    # local/source installs set 1 (or leave the mode unset) and still rebuild.
    #
    # The mode alone is not enough. It records where the Python package came
    # from, not which tree this script is running out of, and an editable
    # overlay separates the two: UNSLOTH_CI_SOURCE_OVERLAY (and a venv left
    # editable by an earlier --local run) leaves the mode at 0 while
    # $SCRIPT_DIR is a checkout, whose dist is a stale build artifact rather
    # than a release one. A wheel ships no top-level files, so a pyproject.toml
    # next to studio/ means source tree -- keep the mtime rebuild there.
    [ "${STUDIO_LOCAL_INSTALL:-}" = "0" ] &&
        [ ! -f "$REPO_ROOT/pyproject.toml" ] &&
        [ -f "$SCRIPT_DIR/frontend/dist/index.html" ]
}

if [ "$_LLAMA_ONLY" != "1" ]; then
# ── Detect whether frontend needs building ──
# Tauri owns its frontend bundle. Standard PyPI installs use the release-built
# dist shipped in the wheel. Only local/source installs use mtime-based rebuilds.
if [ "${SKIP_STUDIO_FRONTEND:-0}" = "1" ]; then
    _NEED_FRONTEND_BUILD=false
    step "frontend" "bundled (Tauri)"
elif _packaged_frontend_available; then
    _NEED_FRONTEND_BUILD=false
    step "frontend" "bundled (pip install)"
else
_NEED_FRONTEND_BUILD=true
if [ -d "$SCRIPT_DIR/frontend/dist" ]; then
    _changed=$(find "$SCRIPT_DIR/frontend" -maxdepth 1 -type f \
        ! -name 'bun.lock' \
        -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null)
    if [ -z "$_changed" ]; then
        _changed=$(find "$SCRIPT_DIR/frontend/src" "$SCRIPT_DIR/frontend/public" \
            -type f -newer "$SCRIPT_DIR/frontend/dist" -print -quit 2>/dev/null) || true
    fi
    [ -z "$_changed" ] && _NEED_FRONTEND_BUILD=false
fi
fi  # end packaged/Tauri guard

# OXC validator runtime (below) needs node/npm whenever its dir exists, regardless
# of dist staleness; provision Node when the frontend builds OR the OXC dir exists.
_OXC_DIR="$SCRIPT_DIR/backend/core/data_recipe/oxc-validator"
if [ "$_NEED_FRONTEND_BUILD" = false ] && [ ! -d "$_OXC_DIR" ]; then
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# ── Node (isolated; never touches the system Node/npm) ──
# Unsloth's frontend (Vite 8) needs Node ^20.19 || >=22.12 || >=23 and npm >= 11.
# Three sources:
#   system  -- system Node + npm already satisfy both; used read-only.
#   bundled -- install a pinned isolated Node under $UNSLOTH_HOME/node, build-only.
#   skip    -- UNSLOTH_SKIP_NODE_INSTALL=1 and system unsuitable; print manual fix.
# decide_node_source(node_v, npm_v, skip_flag) -> system | bundled | skip
# (pure; unit-tested in tests/sh/test_node_decision.sh).
decide_node_source() {
    _dns_node="${1#v}"
    _dns_npm="$2"
    _dns_skip="$3"
    # Treat empty or non-numeric versions as "missing".
    case "$_dns_node" in ''|*[!0-9.]*) _dns_node='' ;; esac
    case "$_dns_npm"  in ''|*[!0-9.]*) _dns_npm=''  ;; esac
    if [ -n "$_dns_node" ] && [ -n "$_dns_npm" ]; then
        _dns_nmaj="${_dns_node%%.*}"
        case "$_dns_node" in
            *.*) _dns_rest="${_dns_node#*.}"; _dns_nmin="${_dns_rest%%.*}" ;;
            *)   _dns_nmin=0 ;;
        esac
        case "$_dns_nmin" in ''|*[!0-9]*) _dns_nmin=0 ;; esac
        _dns_pmaj="${_dns_npm%%.*}"
        _dns_ok=false
        if [ "$_dns_nmaj" -eq 20 ] && [ "$_dns_nmin" -ge 19 ]; then _dns_ok=true; fi
        if [ "$_dns_nmaj" -eq 22 ] && [ "$_dns_nmin" -ge 12 ]; then _dns_ok=true; fi
        if [ "$_dns_nmaj" -ge 23 ]; then _dns_ok=true; fi
        if [ "$_dns_ok" = true ] && [ "$_dns_pmaj" -ge 11 ]; then
            echo system
            return 0
        fi
    fi
    if [ "$_dns_skip" = "1" ]; then
        echo skip
        return 0
    fi
    echo bundled
}

# Mirror the llama.cpp UNSLOTH_HOME derivation; the frontend build runs first.
if [ -n "$STAGE_ROOT" ]; then
    _NODE_PARENT="$RUNTIME_ROOT"
elif [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    _NODE_PARENT="$STUDIO_HOME"
else
    _NODE_PARENT="$HOME/.unsloth"
fi
NODE_DIR="$_NODE_PARENT/node"

_SYS_NODE_VER="$(node -v 2>/dev/null || true)"
_SYS_NPM_VER="$(npm -v 2>/dev/null || true)"
NODE_SOURCE="$(decide_node_source "$_SYS_NODE_VER" "$_SYS_NPM_VER" "${UNSLOTH_SKIP_NODE_INSTALL:-0}")"
_FRONTEND_SKIP=false

if [ "$NODE_SOURCE" = system ]; then
    step "node" "$(node -v) | npm $(npm -v) (system)"
elif [ "$NODE_SOURCE" = bundled ]; then
    mkdir -p "$_NODE_PARENT"
    # install_node_prebuilt.py uses os.replace(); guard a custom-home dir so we
    # never displace a user-owned $UNSLOTH_STUDIO_HOME/node.
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$NODE_DIR" "Node install"
    fi
    substep "installing isolated Node (system Node/npm left untouched)..."
    # Runs before the venv is activated, so bare `python` may be absent; resolve
    # venv python, then python3, then python.
    if [ -x "$VENV_DIR/bin/python" ]; then
        _NODE_PY="$VENV_DIR/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        _NODE_PY="python3"
    else
        _NODE_PY="python"
    fi
    _NODE_LOG="$(mktemp)"
    set +e
    if _is_verbose; then
        "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" 2>&1 | tee "$_NODE_LOG"
        _NODE_STATUS=${PIPESTATUS[0]}
    else
        "$_NODE_PY" "$SCRIPT_DIR/install_node_prebuilt.py" --install-dir "$NODE_DIR" >"$_NODE_LOG" 2>&1
        _NODE_STATUS=$?
    fi
    set -e
    if [ "$_NODE_STATUS" -eq 3 ]; then
        step "node" "install blocked by another active Unsloth install" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "close other Unsloth installs and retry"
        setup_fail 3 "Node install is blocked by another active Unsloth install"
    elif [ "$_NODE_STATUS" -ne 0 ]; then
        step "node" "isolated Node install failed" "$C_ERR"
        sed 's/^/   | /' "$_NODE_LOG" >&2; rm -f "$_NODE_LOG"
        substep "install Node >= 20.19 (with npm >= 11) yourself and re-run, or check your network"
        setup_fail 1 "Could not install an isolated Node runtime"
    fi
    grep -Fq "already matches" "$_NODE_LOG" && verbose_substep "isolated Node already up to date"
    rm -f "$_NODE_LOG"
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$NODE_DIR" ]; then
        : > "$NODE_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    # Prepend the isolated bin (this process only) so node/npm/bun resolve here.
    export PATH="$NODE_DIR/bin:$PATH"
    # Keep npm and module resolution inside the isolated Node.
    export NPM_CONFIG_PREFIX="$NODE_DIR"
    export npm_config_prefix="$NODE_DIR"
    unset NODE_PATH
    hash -r 2>/dev/null || true
    step "node" "$(node -v) | npm $(npm -v) (isolated)"
else
    _FRONTEND_SKIP=true
    step "frontend" "skipped (no suitable Node; system left untouched)" "$C_WARN"
    substep "found Node='${_SYS_NODE_VER:-none}' npm='${_SYS_NPM_VER:-none}'; Unsloth needs Node >=20.19/22.12/23 and npm >= 11"
    substep "install a suitable Node + npm, or unset UNSLOTH_SKIP_NODE_INSTALL to let Unsloth manage an isolated Node"
fi
verbose_substep "node source: $NODE_SOURCE (sys node=${_SYS_NODE_VER:-none} npm=${_SYS_NPM_VER:-none}) dir=$NODE_DIR"

if [ "$_FRONTEND_SKIP" = true ]; then
    : # no suitable Node (skip source): message already shown above; nothing to build
elif [ "$_NEED_FRONTEND_BUILD" = false ]; then
    # Node was provisioned only for the OXC runtime; the dist is already current.
    step "frontend" "up to date"
    verbose_substep "frontend dist is newer than source inputs"
else

# ── Install bun (optional, faster package installs) ──
# Install bun via npm only when we manage the isolated Node (npm -g lands in the
# isolated prefix); on a system Node we install nothing global. Build falls back to npm.
if command -v bun &>/dev/null; then
    substep "bun already installed ($(bun --version))"
elif [ "$NODE_SOURCE" = bundled ]; then
    substep "installing bun..."
    # --allow-scripts=bun: npm >=11.16 gates install scripts and bun's
    # postinstall fetches its binary; without it the install is a broken stub.
    if run_maybe_quiet npm install -g bun --allow-scripts=bun "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" && command -v bun &>/dev/null; then
        substep "bun installed ($(bun --version))"
    else
        substep "bun install skipped (npm will be used instead)"
    fi
else
    verbose_substep "skipping global bun install on system Node (npm will be used)"
fi

# ── Build frontend ──
substep "building frontend..."
cd "$SCRIPT_DIR/frontend"
_HIDDEN_GITIGNORES=()
_dir="$(pwd)"
while [ "$_dir" != "/" ]; do
    _dir="$(dirname "$_dir")"
    if [ -f "$_dir/.gitignore" ] && grep -qx '\*' "$_dir/.gitignore" 2>/dev/null; then
        mv "$_dir/.gitignore" "$_dir/.gitignore._twbuild"
        _HIDDEN_GITIGNORES+=("$_dir/.gitignore")
    fi
done

_restore_gitignores() {
    for _gi in "${_HIDDEN_GITIGNORES[@]+"${_HIDDEN_GITIGNORES[@]}"}"; do
        mv "${_gi}._twbuild" "$_gi" 2>/dev/null || true
    done
}
trap _restore_gitignores EXIT

# Use bun for install if available (faster), fall back to npm.
# Build always uses npm (Node runtime -- avoids bun runtime issues on some platforms).
# NOTE: We intentionally avoid run_quiet for the bun install attempt because
# run_quiet calls exit on failure, which would kill the script before the npm
# fallback can run. Instead we capture output manually and only show it on failure.
#
# IMPORTANT: bun's package cache can become corrupt -- packages get stored
# with only metadata (package.json, README) but no actual content (bin/,
# lib/). When this happens bun install exits 0 but leaves binaries missing.
# We verify critical binaries after install. If missing, we clear the cache
# and retry once before falling back to npm.
_try_bun_install() {
    local _log _exit_code=0
    _log=$(mktemp)
    bun install "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" >"$_log" 2>&1 || _exit_code=$?

    # bun may create .exe shims on Windows (Git Bash / MSYS2) instead of plain scripts
    if [ "$_exit_code" -eq 0 ] \
        && { [ -x node_modules/.bin/tsc ] || [ -f node_modules/.bin/tsc.exe ] || [ -f node_modules/.bin/tsc.bunx ]; } \
        && { [ -x node_modules/.bin/vite ] || [ -f node_modules/.bin/vite.exe ] || [ -f node_modules/.bin/vite.bunx ]; }; then
        rm -f "$_log"
        return 0
    fi

    # Either bun install failed or it exited 0 but left packages missing
    if [ "$_exit_code" -ne 0 ]; then
        echo "   bun install failed (exit code $_exit_code):"
    else
        echo "   bun install exited 0 but critical binaries are missing:"
    fi
    sed 's/^/   | /' "$_log" >&2
    if [ -n "${_CAPTURE_LOG:-}" ]; then cat "$_log" >> "$_CAPTURE_LOG" 2>/dev/null || true; fi
    rm -f "$_log"
    rm -rf node_modules
    return 1
}

# Capture install output (bun + npm fallback) so we can detect a registry block.
_FRONTEND_INSTALL_LOG=$(mktemp)
_CAPTURE_LOG="$_FRONTEND_INSTALL_LOG"
_bun_install_ok=false
if command -v bun &>/dev/null; then
    substep "using bun for package install (faster)"
    if _try_bun_install; then
        _bun_install_ok=true
    else
        # First attempt failed, likely due to corrupt cache entries.
        # Clear the cache and retry once.
        echo "   Clearing bun cache and retrying..."
        run_maybe_quiet bun pm cache rm || true
        if _try_bun_install; then
            _bun_install_ok=true
        fi
    fi
fi
if [ "$_bun_install_ok" = false ]; then
    # `|| _npm_install_rc=$?` keeps this off `set -e`'s exit path (run_quiet_no_exit
    # returns non-zero on failure) so the hint branch is reachable; it also captures
    # the exact exit code. Mirrors the `|| BUILD_OK=false` idiom used below.
    _npm_install_rc=0
    run_quiet_no_exit "npm install" npm install --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _npm_install_rc=$?
    if [ "$_npm_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_FRONTEND_INSTALL_LOG"
        rm -f "$_FRONTEND_INSTALL_LOG"
        setup_fail "$_npm_install_rc" "Frontend dependency installation failed (exit code $_npm_install_rc)"
    fi
fi
_CAPTURE_LOG=""
rm -f "$_FRONTEND_INSTALL_LOG"
run_quiet "npm run build" npm run build

_restore_gitignores
trap - EXIT

_MAX_CSS=$(find "$SCRIPT_DIR/frontend/dist/assets" -name '*.css' -exec wc -c {} + 2>/dev/null | sort -n | tail -1 | awk '{print $1}')
if [ -z "$_MAX_CSS" ]; then
    step "frontend" "built (warning: no CSS emitted)" "$C_WARN"
elif [ "$_MAX_CSS" -lt 100000 ]; then
    step "frontend" "built (warning: CSS may be truncated)" "$C_WARN"
else
    step "frontend" "built"
fi

cd "$SCRIPT_DIR"

fi  # end _FRONTEND_SKIP guard (Node available: system or isolated)

fi  # end frontend build check

# ── oxc-validator runtime ──
# Skip when the user opted out of Node (NODE_SOURCE=skip): there is no suitable
# Node, so do not run npm install against an unsuitable/absent system Node.
if [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ] && command -v npm &>/dev/null; then
    cd "$_OXC_DIR"
    _OXC_INSTALL_LOG=$(mktemp)
    _CAPTURE_LOG="$_OXC_INSTALL_LOG"
    # `|| _oxc_install_rc=$?` keeps this off `set -e`'s exit path so the hint branch
    # below is reachable; it also captures the exact exit code.
    _oxc_install_rc=0
    run_quiet_no_exit "npm install (oxc validator runtime)" npm install --no-fund --no-audit --loglevel=error "${_NPM_REGISTRY_ARGS[@]+"${_NPM_REGISTRY_ARGS[@]}"}" || _oxc_install_rc=$?
    _CAPTURE_LOG=""
    if [ "$_oxc_install_rc" -ne 0 ]; then
        _suggest_npm_registry "$_OXC_INSTALL_LOG"
        rm -f "$_OXC_INSTALL_LOG"
        setup_fail "$_oxc_install_rc" "OXC validator dependency installation failed (exit code $_oxc_install_rc)"
    fi
    rm -f "$_OXC_INSTALL_LOG"
    cd "$SCRIPT_DIR"
elif [ -d "$_OXC_DIR" ] && [ "${NODE_SOURCE:-}" != skip ]; then
    # No npm on PATH: skip rather than abort; the backend Node resolver degrades
    # the validator gracefully. Mirrors setup.ps1's elseif on this block.
    substep "OXC validator runtime skipped (no npm found); code validation degrades until Node is available" "$C_WARN"
fi

_remove_agent_instruction_files \
    "$SCRIPT_DIR/frontend/node_modules" \
    "$_OXC_DIR/node_modules"

# ── Python venv + deps ──

[ -d "$REPO_ROOT/.venv" ] && rm -rf "$REPO_ROOT/.venv"
[ -d "$REPO_ROOT/.venv_overlay" ] && rm -rf "$REPO_ROOT/.venv_overlay"
[ -d "$REPO_ROOT/.venv_t5" ] && rm -rf "$REPO_ROOT/.venv_t5"
[ -d "$REPO_ROOT/.venv_t5_530" ] && rm -rf "$REPO_ROOT/.venv_t5_530"
[ -d "$REPO_ROOT/.venv_t5_550" ] && rm -rf "$REPO_ROOT/.venv_t5_550"
# Note: do NOT delete $STUDIO_HOME/.venv here — install.sh handles migration

_COLAB_NO_VENV=false
if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ "$IS_COLAB" = true ]; then
        # On Colab there is no Unsloth venv -- install backend deps into system Python.
        # Strip all version constraints so pip keeps Colab's pre-installed
        # packages (huggingface-hub, datasets, transformers) and only pulls
        # in genuinely missing ones (structlog, fastapi, etc.).
        substep "Colab detected, installing Unsloth backend dependencies..."
        _COLAB_REQS_TMP="$(mktemp)"
        sed 's/[><=!~;].*//' "$SCRIPT_DIR/backend/requirements/studio.txt" \
            | grep -v '^#' | grep -v '^$' > "$_COLAB_REQS_TMP"
        if [ -s "$_COLAB_REQS_TMP" ]; then
            if ! run_quiet_no_exit "install Colab backend deps" pip install -q -r "$_COLAB_REQS_TMP"; then
                rm -f "$_COLAB_REQS_TMP"
                step "python" "Colab backend dependency install failed" "$C_ERR"
                setup_fail 1 "Colab backend dependency installation failed"
            fi
        else
            step "python" "no Colab backend dependencies resolved from requirements file" "$C_WARN"
        fi
        rm -f "$_COLAB_REQS_TMP"
        _COLAB_NO_VENV=true
    else
        step "python" "venv not found at $VENV_DIR" "$C_ERR"
        substep "Run install.sh first to create the environment:"
        substep "curl -fsSL https://unsloth.ai/install.sh | sh"
        setup_fail 1 "Virtual environment not found at $VENV_DIR"
    fi
elif [ -n "$STAGE_ROOT" ]; then
    VIRTUAL_ENV="$VENV_DIR"
    PATH="$VENV_DIR/bin:$PATH"
    export VIRTUAL_ENV PATH
    unset PYTHONHOME
    hash -r 2>/dev/null || true
else
    source "$VENV_DIR/bin/activate"
fi

install_python_stack() {
    python "$SCRIPT_DIR/install_python_stack.py"
}

# ── HTTP GET to stdout (supports curl and wget) ──
# install.sh takes either transport everywhere, so a wget-only box installs fine
# and then stalled here, where curl was the only way to fetch anything.
_setup_http_get() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        return 1
    fi
}

# Same, with a deadline, for the checks that must not hang the install.
# wget has nothing like curl's total-transfer --max-time: --timeout is per
# operation and it retries 20 times, so a stalled server took minutes and a slow
# drip never ended. --tries=1 plus an outer `timeout` restores the 5s ceiling;
# without timeout (base macOS, which ships curl anyway) the per-operation bound
# stands rather than the check being dropped.
_setup_http_get_timed() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --max-time 5 "$1"
    elif command -v wget >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 5 wget -qO- --timeout=5 --tries=1 "$1"
        else
            wget -qO- --timeout=5 --tries=1 "$1"
        fi
    else
        return 1
    fi
}

# ── uv from a pinned release ──
# Same archive and destination as astral's installer, but it fetches a data file with a
# pinned SHA-256 instead of piping remote script text into a shell. Mirrors install.sh.
# Bumping the version means bumping every hash:
#   curl -sL https://github.com/astral-sh/uv/releases/download/<ver>/<asset>.sha256
#
# Only the four mainstream targets are pinned; the rest fall through to the existing path
# rather than risk a binary for the wrong triple.
_SETUP_UV_PINNED_VERSION="0.12.1"

# Mirrors _uv_glibc_minor in install.sh: "not musl" is not the same as "a glibc new enough to
# run the GNU build", and astral drops to its musl-static archive below its floor.
_setup_uv_glibc_minor() {
    _sugm_line=$( (ldd --version 2>/dev/null || true) | head -1 )
    case "$_sugm_line" in *[Mm]usl*) return 1 ;; esac
    _sugm_ver=$(printf '%s\n' "$_sugm_line" | awk '{print $NF}')
    case "$_sugm_ver" in
        2.[0-9]*) : ;;
        *) _sugm_ver=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $NF}') ;;
    esac
    case "$_sugm_ver" in 2.[0-9]*) : ;; *) return 1 ;; esac
    _sugm_minor=${_sugm_ver#2.}
    _sugm_minor=${_sugm_minor%%.*}
    case "$_sugm_minor" in "" | *[!0-9]*) return 1 ;; esac
    echo "$_sugm_minor"
    return 0
}

_setup_uv_pinned_asset() {
    _supa_os=$(uname -s 2>/dev/null || echo unknown)
    _supa_arch=$(uname -m 2>/dev/null || echo unknown)
    case "$_supa_os" in
        Linux)
            # A 32-bit userland on a 64-bit kernel still reports x86_64 from uname.
            [ "$(getconf LONG_BIT 2>/dev/null || echo 0)" = "64" ] || return 1
            _supa_glibc=$(_setup_uv_glibc_minor) || return 1
            case "$_supa_arch" in
                x86_64|amd64)
                    [ "$_supa_glibc" -ge 17 ] 2>/dev/null || return 1
                    echo "uv-x86_64-unknown-linux-gnu.tar.gz 90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb" ;;
                aarch64|arm64)
                    [ "$_supa_glibc" -ge 28 ] 2>/dev/null || return 1
                    echo "uv-aarch64-unknown-linux-gnu.tar.gz 769d373e146692c639b5fbaae33b331c297a32e03d30448772051902df52bbf4" ;;
                *) return 1 ;;
            esac
            ;;
        Darwin)
            # Rosetta 2 reports x86_64 from a translated shell; astral reads the same sysctl.
            if [ "$_supa_arch" = "x86_64" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
                _supa_arch=arm64
            fi
            case "$_supa_arch" in
                x86_64)
                    echo "uv-x86_64-apple-darwin.tar.gz 69d9f9a00337f25a50dcb13882052da08b8469bac11091c98c5694c3c6721467" ;;
                arm64|aarch64)
                    echo "uv-aarch64-apple-darwin.tar.gz 77d2906988e8074fd43f2f329ec452ebbf9b0c257ba1c66451c71de70a6baf42" ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
    return 0
}

_setup_uv_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Bounded liveness probe: no stdin, so a build that prompts reads EOF, and a ceiling where
# `timeout` exists (stock macOS has none).
_setup_uv_probe_exec() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 20 "$1" --version >/dev/null 2>&1 </dev/null
    else
        "$1" --version >/dev/null 2>&1 </dev/null
    fi
}

# The function's own cleanup only runs when it returns, so an interrupt left the unpacked
# archive behind plus a staging file inside a directory on PATH. No trap is active here (the
# gitignore EXIT trap is cleared well above), so the pinned install owns these four for its
# duration and hands them back on the way out.
_setup_uv_cleanup_temporaries() {
    [ -n "${_SIUP_WORK:-}" ] && rm -rf "$_SIUP_WORK" 2>/dev/null || true
    [ -n "${_SIUP_STAGE:-}" ] && rm -f "$_SIUP_STAGE" 2>/dev/null || true
    [ -n "${_SIUP_STAGE2:-}" ] && rm -f "$_SIUP_STAGE2" 2>/dev/null || true
}

_setup_uv_on_signal() {
    trap - EXIT HUP INT TERM
    _setup_uv_cleanup_temporaries
    exit "$1"
}

# The PATH a new shell inherits, captured before the uv destination can be prepended to it.
_SETUP_LOGIN_PATH="$PATH"
_SIUP_WORK=""
_SIUP_STAGE=""
_SIUP_STAGE2=""

_setup_install_uv_pinned() {
    _siup_spec=$(_setup_uv_pinned_asset) || return 1
    [ -n "$_siup_spec" ] || return 1
    _siup_asset=${_siup_spec%% *}
    _siup_want=${_siup_spec##* }
    command -v tar >/dev/null 2>&1 || return 1
    [ -n "$(_setup_uv_sha256 /dev/null)" ] || return 1
    [ -n "${HOME:-}" ] || return 1

    # astral's full destination priority, including the XDG_DATA_HOME tier that sits between
    # XDG_BIN_HOME and the home default. Dropping it would leave uv under ~/.local/bin on a
    # host that configured an XDG location, where no later shell looks for it.
    _siup_dest="${UV_INSTALL_DIR:-${UV_UNMANAGED_INSTALL:-${XDG_BIN_HOME:-}}}"
    if [ -z "$_siup_dest" ] && [ -n "${XDG_DATA_HOME:-}" ]; then _siup_dest="$XDG_DATA_HOME/../bin"; fi
    [ -n "$_siup_dest" ] || _siup_dest="$HOME/.local/bin"
    # 2>/dev/null, as install.sh does: a speculative attempt whose failure falls back.
    _siup_work=$(mktemp -d 2>/dev/null) || return 1
    _SIUP_WORK="$_siup_work"
    trap _setup_uv_cleanup_temporaries EXIT
    trap '_setup_uv_on_signal 129' HUP
    trap '_setup_uv_on_signal 130' INT
    trap '_setup_uv_on_signal 143' TERM
    _siup_rc=1
    # A configured mirror is EXCLUSIVE, matching astral's installer and the PowerShell side: a
    # restricted network sets one because the public hosts are unreachable, so trying those first
    # would stall instead of falling through.
    if [ -n "${UV_DOWNLOAD_URL:-}" ]; then
        _siup_bases="${UV_DOWNLOAD_URL%/}"
    elif [ -n "${INSTALLER_DOWNLOAD_URL:-}" ]; then
        _siup_bases="${INSTALLER_DOWNLOAD_URL%/}"
    elif [ -n "${UV_INSTALLER_GHE_BASE_URL:-}" ]; then
        _siup_bases="${UV_INSTALLER_GHE_BASE_URL%/}/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    elif [ -n "${UV_INSTALLER_GITHUB_BASE_URL:-}" ]; then
        _siup_bases="${UV_INSTALLER_GITHUB_BASE_URL%/}/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    else
        _siup_bases="https://releases.astral.sh/github/uv/releases/download/$_SETUP_UV_PINNED_VERSION
https://github.com/astral-sh/uv/releases/download/$_SETUP_UV_PINNED_VERSION"
    fi
    for _siup_base in $_siup_bases; do
        _setup_http_get "$_siup_base/$_siup_asset" > "$_siup_work/$_siup_asset" 2>/dev/null || continue
        [ -s "$_siup_work/$_siup_asset" ] || continue
        [ "$(_setup_uv_sha256 "$_siup_work/$_siup_asset")" = "$_siup_want" ] || continue
        tar -xzf "$_siup_work/$_siup_asset" -C "$_siup_work" 2>/dev/null || continue
        mkdir -p "$_siup_dest" 2>/dev/null || break
        # Stage both, then publish both, as install.sh does: the renames sit next to each
        # other so the pair is replaced as one.
        _siup_ready=1
        for _siup_exe in uv uvx; do
            # `mv f d` moves f INTO d and reports success, and a searchable directory passes -x
            # too, so a directory called uv at the destination would look like a published
            # binary and skip the fallback. The installer already refuses one for its own shim.
            if [ -d "$_siup_dest/$_siup_exe" ]; then _siup_ready=0; break; fi
            _siup_src=$(find "$_siup_work" -type f -name "$_siup_exe" 2>/dev/null | head -1)
            if [ -z "$_siup_src" ]; then _siup_ready=0; break; fi
            # cp writes through a symlinked destination, and a per-process staging name keeps
            # two racing installers from publishing each other's half-written file.
            _siup_stage=$(mktemp "$_siup_dest/.$_siup_exe.XXXXXX" 2>/dev/null) || { _siup_ready=0; break; }
            if [ "$_siup_exe" = "uv" ]; then _SIUP_STAGE="$_siup_stage"; else _SIUP_STAGE2="$_siup_stage"; fi
            if ! cp -f "$_siup_src" "$_siup_stage" 2>/dev/null; then _siup_ready=0; break; fi
            # 0755, not +x: the staging file carries the umask default and +x only adds execute
            # where read was allowed, so umask 077 would leave uv unusable for every other
            # account. astral ships these 0755.
            chmod 0755 "$_siup_stage" 2>/dev/null || true
            # Validate before publishing: the rename destroys the incumbent, so a binary that
            # cannot run here must never replace one that could.
            if [ "$_siup_exe" = "uv" ] && ! _setup_uv_probe_exec "$_siup_stage"; then _siup_ready=0; break; fi
        done
        if [ "$_siup_ready" = "1" ] &&
           mv -f "$_SIUP_STAGE" "$_siup_dest/uv" 2>/dev/null &&
           mv -f "$_SIUP_STAGE2" "$_siup_dest/uvx" 2>/dev/null; then
            _siup_rc=0
        fi
        rm -f "$_SIUP_STAGE" "$_SIUP_STAGE2" 2>/dev/null || true
        _SIUP_STAGE=""
        _SIUP_STAGE2=""
        break
    done
    rm -rf "$_siup_work"
    _SIUP_WORK=""
    _SIUP_STAGE=""
    _SIUP_STAGE2=""
    trap - EXIT HUP INT TERM
    # The staged binary already answered --version above, before it replaced anything.
    [ -x "$_siup_dest/uv" ] || _siup_rc=1
    if [ "$_siup_rc" = "0" ]; then
        export PATH="$_siup_dest:$PATH"
        _setup_persist_uv_path "$_siup_dest"
    fi
    return "$_siup_rc"
}

# astral's installer wrote a profile line for whichever destination it chose. This replaces that
# installer, so setup.sh run directly (local or Colab) has to do the same or the export above
# dies with this shell and every later run reinstalls uv. Both of astral's opt-outs apply, and
# fish is handled on its own terms since it reads none of the POSIX rc files.
# Is $2 one of the colon-separated entries of $1? Field splitting also globs, so pathname
# expansion is off for the walk and restored afterwards.
_setup_path_has_dir() {
    _sphd_glob=on
    case $- in *f*) _sphd_glob=off ;; esac
    set -f
    _sphd_found=1
    _sphd_old_ifs="$IFS"
    IFS=:
    for _sphd_entry in $1; do
        if [ "$_sphd_entry" = "$2" ]; then _sphd_found=0; break; fi
    done
    IFS="$_sphd_old_ifs"
    [ "$_sphd_glob" = on ] && set +f
    return "$_sphd_found"
}

_setup_persist_uv_path() {
    _supp_dir="$1"
    [ -n "$_supp_dir" ] || return 0
    [ -n "${HOME:-}" ] || return 0
    [ -z "${UV_NO_MODIFY_PATH:-}" ] || return 0
    [ -z "${UV_UNMANAGED_INSTALL:-}" ] || return 0
    # The PATH a new shell inherits, not the one this process has already prepended to, and
    # compared entry by entry: a directory holding *, ? or [ is a glob inside a case pattern.
    _setup_path_has_dir "${_SETUP_LOGIN_PATH:-$PATH}" "$_supp_dir" && return 0
    # ~/.config, not XDG_CONFIG_HOME, because that is where astral's installer put its own fish
    # file, and it is written regardless of the current shell for the same reason.
    _supp_fish_dir="$HOME/.config/fish/conf.d"
    if mkdir -p "$_supp_fish_dir" 2>/dev/null; then
        _supp_fish="$_supp_fish_dir/unsloth.fish"
        # Single-quoted: an unquoted path with a space is two arguments to fish_add_path.
        _supp_quoted=$(printf '%s' "$_supp_dir" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")
        # The exact line, not any occurrence: /opt/uv-old must not pass for /opt/uv.
        if ! grep -v '^[[:space:]]*#' "$_supp_fish" 2>/dev/null | grep -qxF "fish_add_path '$_supp_quoted'"; then
            echo "# Added by Unsloth setup" >> "$_supp_fish"
            echo "fish_add_path '$_supp_quoted'" >> "$_supp_fish"
        fi
    fi
    # An entry has to be active, whole and on a line that SETS PATH: a commented-out export,
    # /opt/uv-old when we want /opt/uv, and PYTHONPATH=/opt/uv are none of them, and taking any
    # for an entry leaves the next shell unable to resolve uv.
    _supp_path_line='(^|[^[:alnum:]_])(PATH[[:space:]]*=|fish_add_path|pathmunge|path_helper)'
    _supp_grep=$(printf '%s' "$_supp_dir" | sed 's/[].[\\()*+?{}|^$\/]/\\&/g')
    # Escaped: the line is double-quoted, so a path holding $, ` or " would be expanded or
    # terminated by the shell that reads it.
    _supp_literal=$(printf '%s' "$_supp_dir" | sed 's/[\\"$`]/\\&/g')
    # Every startup file astral's installer wired, because it is the installer this replaced:
    # ~/.profile always, each bash file that exists, and zsh under ZDOTDIR. Writing only the
    # file for the shell that happens to be running would leave a bash user whose .bash_profile
    # does not source .bashrc, or anyone who later switches shells, without uv on PATH.
    for _supp_profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" \
                         "$HOME/.bash_login" "${ZDOTDIR:-$HOME}/.zshrc" "${ZDOTDIR:-$HOME}/.zshenv"; do
        if [ "$_supp_profile" != "$HOME/.profile" ] && [ ! -f "$_supp_profile" ]; then continue; fi
        # Only lines that actually set PATH count: `UV_CACHE=/opt/uv` and `PYTHONPATH=/opt/uv`
        # are not PATH entries, and taking one for an entry leaves the next shell without uv.
        if grep -v '^[[:space:]]*#' "$_supp_profile" 2>/dev/null \
            | grep -E "$_supp_path_line" \
            | grep -qE "(^|[^[:alnum:]_.~/-])$_supp_grep([^[:alnum:]_.~/-]|\$)"; then continue; fi
        echo '' >> "$_supp_profile"
        echo '# Added by Unsloth setup' >> "$_supp_profile"
        echo "export PATH=\"$_supp_literal:\$PATH\"" >> "$_supp_profile"
    done
}

USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
elif [ -n "$STAGE_ROOT" ]; then
    step "uv" "using pip inside the staged environment"
elif {
    _SETUP_UV_PINNED_OK=false
    if _setup_install_uv_pinned; then
        _SETUP_UV_PINNED_OK=true
    elif _is_verbose; then
        _setup_http_get https://astral.sh/uv/install.sh | sh
    else
        _setup_http_get https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    fi
}; then
    # Only for astral's installer, which writes to ~/.local/bin. The pinned path already put its
    # own destination first, and prepending here would let a stale ~/.local/bin/uv shadow the
    # 0.12.1 we just verified, so the rest of setup would run the wrong one.
    [ "$_SETUP_UV_PINNED_OK" = true ] || export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null && USE_UV=true
fi

fast_install() {
    if [ "$USE_UV" = true ]; then
        uv pip install --python "$(command -v python)" "$@" && return 0
    fi
    python -m pip install "$@"
}

fast_install_sidecar() (
    unset UV_OVERRIDE
    fast_install "$@"
)

cd "$SCRIPT_DIR"

# On Colab without a venv, skip venv-dependent Python deps sections but
# continue to llama.cpp install so GGUF inference is available.
if [ "$_COLAB_NO_VENV" = true ]; then
    step "python" "backend deps installed into system Python"
    substep "continuing to llama.cpp install for GGUF inference support"
fi

# ── Check if Python deps need updating ──
# Compare installed package version against PyPI latest.
# Skip all Python dependency work if versions match (fast update path).
# On Colab (no venv), skip this version check (it needs $VENV_DIR/bin/python)
# but still run install_python_stack below (it uses sys.executable).
_SKIP_PYTHON_DEPS=false
_SKIP_VERSION_CHECK=false
if [ "$_COLAB_NO_VENV" = true ]; then
    _SKIP_VERSION_CHECK=true
fi
_PKG_NAME="${STUDIO_PACKAGE_NAME:-unsloth}"
if [ "$_SKIP_VERSION_CHECK" != true ] && [ "${SKIP_STUDIO_BASE:-0}" != "1" ] && [ "${STUDIO_LOCAL_INSTALL:-0}" != "1" ]; then
    # Only check when NOT called from install.sh (which just installed the package)
    _INSTALLED_VERSION_PROBE_EXIT=0
    if INSTALLED_VER=$("$VENV_DIR/bin/python" -c "
import sys
sys.path.insert(0, sys.argv[2])
import install_manifest
version, conflict = install_manifest.installed_version_probe(sys.argv[1], ('unsloth-zoo',))
print(version)
sys.exit(2 if conflict else (0 if version else 1))
" "$_PKG_NAME" "$SCRIPT_DIR" 2>/dev/null); then
        :
    else
        _INSTALLED_VERSION_PROBE_EXIT=$?
        INSTALLED_VER=""
    fi

    LATEST_VER=$(_setup_http_get_timed "https://pypi.org/pypi/$_PKG_NAME/json" 2>/dev/null \
        | "$VENV_DIR/bin/python" -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null \
        || echo "")

    if [ "$_INSTALLED_VERSION_PROBE_EXIT" -eq 2 ]; then
        substep "duplicate metadata found for a core package -- forcing package repair..."
    elif [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then
        step "python" "$_PKG_NAME $INSTALLED_VER is up to date"
        _SKIP_PYTHON_DEPS=true
        # A pre-#6483-fix install can be stuck on anyio>=4.14 even though
        # $_PKG_NAME itself is current; the fast path above would otherwise
        # never reach install_python_stack's anyio repair (#6797).
        if "$VENV_DIR/bin/python" -c "
import re, sys
from importlib.metadata import version, PackageNotFoundError
try:
    parts = version('anyio').split('.')
    major = int(parts[0])
    minor = int(re.sub(r'[^0-9].*', '', parts[1])) if len(parts) > 1 else 0
except (PackageNotFoundError, ValueError, IndexError):
    sys.exit(1)
sys.exit(0 if (major, minor) >= (4, 14) else 1)
" 2>/dev/null; then
            substep "anyio >=4.14 found (#6483) -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        fi
        # An interrupted install leaves $_PKG_NAME current while studio.txt
        # never finished, so the compare above says "up to date" and update --
        # plus the desktop Repair button -- no-ops on a venv that cannot boot.
        if ! "$VENV_DIR/bin/python" -c "
import sys
sys.path.insert(0, sys.argv[1])
try:
    import install_manifest
except Exception:
    sys.exit(0)  # older tree without the manifest helper: leave the fast path alone
sys.exit(0 if install_manifest.verify_install()['ok'] else 1)
" "$SCRIPT_DIR" 2>/dev/null; then
            substep "studio install incomplete -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        fi
        # If the desktop app specifies a minimum required backend version and the installed
        # package is older than that requirement, force the dependency pass to upgrade it.
        if [ -n "${UNSLOTH_DESKTOP_BACKEND_VERSION:-}" ]; then
            if ! "$VENV_DIR/bin/python" -c "
import re, sys
try:
    from packaging.version import parse as parse_v
except ImportError:
    def parse_v(v):
        match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', (v or '').strip())
        return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None
installed = parse_v(sys.argv[1])
required = parse_v(sys.argv[2])
sys.exit(0 if installed is not None and required is not None and installed >= required else 1)
" "$INSTALLED_VER" "$UNSLOTH_DESKTOP_BACKEND_VERSION" 2>/dev/null; then
                substep "$_PKG_NAME $INSTALLED_VER < $UNSLOTH_DESKTOP_BACKEND_VERSION (required by desktop app) -- forcing dependency pass to update..."
                _SKIP_PYTHON_DEPS=false
            fi
        fi
        # An XPU pin the venv does not satisfy. Only the dependency pass acts on it
        # (install_python_stack's _ensure_xpu_torch), so without this escape a CPU install
        # switched to UNSLOTH_TORCH_INDEX_FAMILY=xpu keeps its CPU wheel forever: the package
        # version is current, so the fast path calls it up to date. Mirrors setup.ps1.
        _setup_pin="${UNSLOTH_TORCH_INDEX_URL:-${UNSLOTH_TORCH_INDEX_FAMILY:-}}"
        # Strip query/fragment first: an authenticated mirror (…/whl/xpu?token=...) is a
        # supported pin shape, and missing it reads as "no XPU pin" and skips the repair.
        _setup_pin="${_setup_pin%%\#*}"
        _setup_pin="${_setup_pin%%\?*}"
        # ALL trailing slashes, like the shared leaf parsers: a single %/ leaves "…/xpu/" behind.
        while [ "${_setup_pin%/}" != "$_setup_pin" ]; do _setup_pin="${_setup_pin%/}"; done
        # Exact, lowercased leaf, like every other leaf parser: a *xpu suffix match (…/private-xpu)
        # would force a pass every update that _ensure_xpu_torch then declines to act on, and an
        # uncased match would miss UNSLOTH_TORCH_INDEX_FAMILY=XPU that those parsers do accept.
        _setup_pin_leaf=$(printf '%s' "${_setup_pin##*/}" | tr '[:upper:]' '[:lower:]')
        # Disk first, no interpreter: version.py carries the local label, so a wedged Intel
        # driver cannot hang `studio update` inside `import torch`. Read unconditionally, not
        # only under a pin: the pin is one-shot, so the installed wheel is the only durable
        # signal -- the same one _ensure_xpu_triton keys on.
        _setup_pin_ok=false
        _setup_pin_is_xpu=false
        for _setup_pin_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
            [ -f "$_setup_pin_tv" ] || continue
            _setup_pin_ver=$(sed -n "s/^__version__ = '\([^']*\)'.*/\1/p" "$_setup_pin_tv" | head -1)
            case "$_setup_pin_ver" in
                *+xpu)
                    _setup_pin_is_xpu=true
                    _setup_pin_maj=${_setup_pin_ver%%.*}
                    _setup_pin_rest=${_setup_pin_ver#*.}
                    _setup_pin_min=${_setup_pin_rest%%.*}
                    case "$_setup_pin_maj$_setup_pin_min" in
                        *[!0-9]*) ;;
                        *) [ "$_setup_pin_maj" -eq 2 ] && [ "$_setup_pin_min" -ge 6 ] && \
                           [ "$_setup_pin_min" -lt 11 ] && _setup_pin_ok=true ;;
                    esac
                    ;;
            esac
            break
        done
        # Correct torch is not enough: the Triton swap also lives in install_python_stack, so a
        # migrated +xpu venv with a leftover generic triton keeps the CUDA build shadowing the
        # XPU one. The dist-info glob below matches only generic "triton-<ver>" -- the XPU builds
        # are pytorch_triton_xpu-* / triton_xpu-*.
        # Leaves the shared classifiers recognise as a non-XPU family. EXACT families, mirroring
        # install.sh _is_pip_rocm_family_leaf and install_python_stack _is_cuda_family_leaf: a
        # merely prefixed leaf (cu128-private) is a custom verbatim pin they never repair, so
        # escaping on one would force a pass every update that changes nothing.
        _setup_known_nonxpu_leaf() {
            case "$1" in
                cpu|gfx[0-9]*) return 0 ;;
                cu[0-9]*) case "${1#cu}" in *[!0-9]*) return 1 ;; esac ;;
                rocm[0-9]*)
                    # Both parts non-empty all-digits: rocm7., rocm7.2.1 are custom pins.
                    _setup_rocm_rest="${1#rocm}"
                    case "$_setup_rocm_rest" in
                        *.*.*) return 1 ;;
                        *.*)
                            case "${_setup_rocm_rest%%.*}" in *[!0-9]*) return 1 ;; esac
                            case "${_setup_rocm_rest#*.}" in "" | *[!0-9]*) return 1 ;; esac
                            ;;
                        *[!0-9]*) return 1 ;;
                    esac
                    ;;
                *) return 1 ;;
            esac
            return 0
        }
        _setup_pin_known_nonxpu=false
        _setup_known_nonxpu_leaf "$_setup_pin_leaf" && _setup_pin_known_nonxpu=true
        _setup_generic_triton=false
        if [ "$_setup_pin_is_xpu" = true ] || [ "$_setup_pin_leaf" = "xpu" ]; then
            for _setup_tri in "$VENV_DIR"/lib/python*/site-packages/triton-*.dist-info; do
                [ -d "$_setup_tri" ] && _setup_generic_triton=true && break
            done
        fi
        if [ "$_setup_pin_leaf" = "xpu" ] && [ "$_setup_pin_ok" = false ]; then
            substep "XPU index pinned but torch does not match -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        elif [ "$_setup_pin_is_xpu" = true ] && [ "$_setup_generic_triton" = true ]; then
            substep "generic triton shadows the XPU build -- forcing dependency pass to repair..."
            _SKIP_PYTHON_DEPS=false
        elif [ "$_setup_pin_is_xpu" = true ] && [ "$_setup_pin_known_nonxpu" = true ]; then
            # Migrating AWAY from XPU: the pin is authoritative, but only install_python_stack
            # acts on it, so an up-to-date install kept its +xpu wheel over the requested family.
            substep "$_setup_pin_leaf pinned over an XPU wheel -- forcing dependency pass to migrate..."
            _SKIP_PYTHON_DEPS=false
        fi
    elif [ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ]; then
        substep "$_PKG_NAME $INSTALLED_VER -> $LATEST_VER available, updating..."
    elif [ -z "$LATEST_VER" ]; then
        substep "could not reach PyPI, updating to be safe..."
    fi
fi

# A current package can still have CPU/CUDA torch because the fast path skips ROCm repair.
# Exit 0 forces the dependency pass; failures and timeouts keep the fast path.
if [ "$_SKIP_PYTHON_DEPS" = true ] && [ -x "$VENV_DIR/bin/python" ]; then
    _setup_amd_torch_stale=false
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 5 180 "$VENV_DIR/bin/python" \
            "$SCRIPT_DIR/install_python_stack.py" --amd-torch-needs-dependency-pass \
            >/dev/null 2>&1 && _setup_amd_torch_stale=true
    elif "$VENV_DIR/bin/python" "$SCRIPT_DIR/install_python_stack.py" \
            --amd-torch-needs-dependency-pass >/dev/null 2>&1; then
        _setup_amd_torch_stale=true
    fi
    if [ "$_setup_amd_torch_stale" = true ]; then
        substep "installed PyTorch is not a ROCm build on this AMD host -- forcing dependency pass to repair..."
        substep "   (set UNSLOTH_TORCH_BACKEND=cpu to keep a deliberate CPU install)"
        _SKIP_PYTHON_DEPS=false
    fi
fi

if [ "$_SKIP_PYTHON_DEPS" = false ]; then
    install_python_stack
else
    step "python" "dependencies up to date"
    verbose_substep "python deps check: installed=$_PKG_NAME@${INSTALLED_VER:-unknown} latest=${LATEST_VER:-unknown}"
fi

# ── 6b. Pre-install transformers 5.x into .venv_t5_530/, .venv_t5_550/, and .venv_t5_510/ ──
# Models like GLM-4.7-Flash, Qwen3 MoE need transformers>=5.3.0.
# Gemma 4 models need transformers>=5.5.0; Gemma 4 Unified needs 5.10.x.
# Pre-install into separate directories to avoid runtime pip overhead.
# The training subprocess prepends the appropriate dir to sys.path.
_target_has_pkg_version() {
    _thpv_dir="$1"
    _thpv_pkg="$2"
    _thpv_version="$3"
    [ -d "$_thpv_dir" ] || return 1
    _thpv_pkg_norm=$(printf '%s' "$_thpv_pkg" | tr '-' '_')
    for _thpv_metadata in \
        "$_thpv_dir"/"$_thpv_pkg_norm"-*.dist-info/METADATA \
        "$_thpv_dir"/"$_thpv_pkg"-*.dist-info/METADATA
    do
        [ -f "$_thpv_metadata" ] || continue
        grep -qx "Version: $_thpv_version" "$_thpv_metadata" && return 0
    done
    return 1
}
_NEED_T5_INSTALL=false
if [ -d "$STUDIO_HOME/.venv_t5" ]; then
    # Legacy layout — migrate. The tiered venvs a staged run builds land under the
    # stage root and may never be activated, so removing the live legacy one here
    # would strip the running install of its only sidecar. The live update does it.
    if [ -z "$STAGE_ROOT" ]; then
        _assert_studio_owned_or_absent "$STUDIO_HOME/.venv_t5" "legacy transformers sidecar venv"
        rm -rf "$STUDIO_HOME/.venv_t5"
    fi
    _NEED_T5_INSTALL=true
fi
[ ! -d "$VENV_T5_530_DIR" ] && _NEED_T5_INSTALL=true
[ ! -d "$VENV_T5_550_DIR" ] && _NEED_T5_INSTALL=true
[ ! -d "$VENV_T5_510_DIR" ] && _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_530_DIR" "transformers" "5.3.0" || _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_550_DIR" "transformers" "5.5.0" || _NEED_T5_INSTALL=true
_target_has_pkg_version "$VENV_T5_510_DIR" "transformers" "5.10.2" || _NEED_T5_INSTALL=true
# Also reinstall when python deps were updated (packages may need rebuild)
[ "$_SKIP_PYTHON_DEPS" = false ] && _NEED_T5_INSTALL=true

if [ "$_NEED_T5_INSTALL" = true ]; then
    _assert_studio_owned_or_absent "$VENV_T5_530_DIR" "transformers 5.3 sidecar venv"
    [ -d "$VENV_T5_530_DIR" ] && rm -rf "$VENV_T5_530_DIR"
    mkdir -p "$VENV_T5_530_DIR"
    : > "$VENV_T5_530_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.3.0" fast_install_sidecar --target "$VENV_T5_530_DIR" --no-deps "transformers==5.3.0"
    run_quiet "install huggingface_hub for t5_530" fast_install_sidecar --target "$VENV_T5_530_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_530" fast_install_sidecar --target "$VENV_T5_530_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_530" fast_install_sidecar --target "$VENV_T5_530_DIR" --no-deps "tiktoken"
    step "transformers" "5.3.0 pre-installed"

    _assert_studio_owned_or_absent "$VENV_T5_550_DIR" "transformers 5.5 sidecar venv"
    [ -d "$VENV_T5_550_DIR" ] && rm -rf "$VENV_T5_550_DIR"
    mkdir -p "$VENV_T5_550_DIR"
    : > "$VENV_T5_550_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.5.0" fast_install_sidecar --target "$VENV_T5_550_DIR" --no-deps "transformers==5.5.0"
    run_quiet "install huggingface_hub for t5_550" fast_install_sidecar --target "$VENV_T5_550_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_550" fast_install_sidecar --target "$VENV_T5_550_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_550" fast_install_sidecar --target "$VENV_T5_550_DIR" --no-deps "tiktoken"
    step "transformers" "5.5.0 pre-installed"

    _assert_studio_owned_or_absent "$VENV_T5_510_DIR" "transformers 5.10 sidecar venv"
    [ -d "$VENV_T5_510_DIR" ] && rm -rf "$VENV_T5_510_DIR"
    mkdir -p "$VENV_T5_510_DIR"
    : > "$VENV_T5_510_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    run_quiet "install transformers 5.10.2" fast_install_sidecar --target "$VENV_T5_510_DIR" --no-deps "transformers==5.10.2"
    run_quiet "install huggingface_hub for t5_510" fast_install_sidecar --target "$VENV_T5_510_DIR" --no-deps "huggingface_hub==1.8.0"
    run_quiet "install hf_xet for t5_510" fast_install_sidecar --target "$VENV_T5_510_DIR" --no-deps "hf_xet==1.4.2"
    run_quiet "install tiktoken for t5_510" fast_install_sidecar --target "$VENV_T5_510_DIR" --no-deps "tiktoken"
    step "transformers" "5.10.2 pre-installed"
fi
fi

# ── GPU detection summary (mirrors setup.ps1 step "gpu" block) ──
# WSL2 ROCDXG: the system rocminfo enumerates the GPU over /dev/dxg only when
# HSA_ENABLE_DXG_DETECTION=1 (a no-op on bare metal), and /opt/rocm/bin can be
# off PATH outside login shells (the profile.d drop-in). Seed both before the
# probes or a ROCDXG WSL host is misdetected as CPU-only.
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
if ! command -v rocminfo >/dev/null 2>&1 && [ -x /opt/rocm/bin/rocminfo ]; then
    PATH="$PATH:/opt/rocm/bin"
fi
_setup_amd_detected=false
_setup_nvidia_usable=false
_setup_gfx_all=""
_setup_gfx=""
_setup_hip_map_missing=0
_setup_mkt=""
_setup_amd_records=""

# Pair each rocminfo GPU gfx id with its marketing name instead of using the CPU-first
# global name (#7307). Blank names keep device ordinals; no GPU keeps the old fallback.
# Keep in sync with install.sh.
_setup_rocminfo_gpu_records() {
    awk '
        # Split at the first colon so embedded colons survive.
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        /^[[:space:]]*Name:/ {
            # Keep a slot for a nameless GPU.
            if (gfx != "" && !named) { print gfx "|"; gpus++ }
            gfx = ""; named = 0
            name = value($0)
            # Accept target suffixes such as gfx90a:sramecc+, but reject ISA names.
            if (match(name, /^gfx[1-9][0-9a-z][0-9a-z][0-9a-z]?/)) {
                rest = substr(name, RLENGTH + 1)
                if (rest == "" || rest ~ /^[^0-9a-z]/) gfx = substr(name, 1, RLENGTH)
            }
            next
        }
        /^[[:space:]]*Marketing Name:/ {
            mkt = value($0)
            if (gfx != "" && !named) { print gfx "|" mkt; gpus++; named = 1 }
            else if (first == "") first = mkt
            next
        }
        END {
            if (gfx != "" && !named) { print gfx "|"; gpus++ }
            if (gpus == 0 && first != "") print "|" first
        }
    '
}

# amd-smi enumerates in discovery order over its KFD view; HIP_VISIBLE_DEVICES and
# ROCR_VISIBLE_DEVICES index HIP/ROCr order, which the library derives from the KFD node
# id instead. The two disagree on real hardware (MI350X SPX/NPS1), and _gfx here becomes
# --rocm-gfx, so an untranslated ordinal can fetch a prebuilt for another card's arch.
# `amd-smi list -e` is the map AMD publishes for this (HIP_ID, ROCm 6.4.0+); the Python
# side reads the same field in utils/hardware/amd.py get_hip_id_by_gpu_index.
# Keep in sync with install.sh.
_setup_amd_smi_hip_order() {
    # POSIX awk forbids a physical newline in a -v value (gawk --posix makes it fatal),
    # so the records arrive on stdin ahead of the map, separated by a sentinel. The first
    # output line reports which index space the records came back in; the caller needs to
    # know, because a mask cannot be applied to an untranslated list of unlike adapters.
    { printf '%s\n' "$1"; echo "@@hip-map@@"; cat; } | awk '
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        function keep(   i) { print "discovery"; for (i = 1; i <= r; i++) print rec[i] }
        !split_seen && $0 == "@@hip-map@@" { split_seen = 1; next }
        !split_seen { if ($0 != "") rec[++r] = $0; next }
        /^[[:space:]]*GPU:[[:space:]]*[0-9]/ { n++; hip[n] = -1; next }
        n && tolower($0) ~ /hip.?id/ {
            if (hip[n] < 0) { v = value($0); if (v ~ /^[0-9]+$/) hip[n] = v + 0 }
            next
        }
        END {
            # All or nothing, like get_hip_id_by_gpu_index: an older CLI rejects -e, and
            # hip_id reads N/A when the library cannot reach a KFD node. A partial or
            # colliding map is not a 1:1 device mapping, so keep discovery order.
            if (r == 0 || n != r) { keep(); exit }
            for (i = 1; i <= n; i++) {
                if (hip[i] < 0 || hip[i] >= r || (hip[i] in used)) { keep(); exit }
                used[hip[i]] = 1
                out[hip[i]] = rec[i]
            }
            print "hip"
            for (i = 0; i < r; i++) print out[i]
        }
    '
}

# One `gfx|marketing name` per adapter, in `GPU: N` order, so the mask picks both halves
# of one device. Was: arch indexed, name always adapter 0's -- and on amd-smi 6.1.1, which
# has no TARGET_GRAPHICS_VERSION, that name is what --rocm-gfx is inferred from.
# Keep in sync with install.sh.
_setup_amd_smi_gpu_records() {
    awk '
        function value(line,   v) {
            v = line
            sub(/^[^:]*:[[:space:]]*/, "", v)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            return v
        }
        function flush() {
            if (started) print gfx "|" mkt
            gfx = ""; mkt = ""
        }
        # amd-smi upper-cases every key (amdsmi_logger.py _capitalize_keys): MARKET_NAME,
        # TARGET_GRAPHICS_VERSION. Matched case-folded so older spellings work too.
        /^[[:space:]]*GPU:[[:space:]]*[0-9]/ { flush(); started = 1; next }
        !started { next }
        tolower($0) ~ /market.?name/ { if (mkt == "") mkt = value($0); next }
        tolower($0) ~ /target.?graphics.?version/ {
            v = value($0)
            if (gfx == "" && v ~ /^gfx[1-9][0-9a-z][0-9a-z][0-9a-z]?$/) gfx = v
            next
        }
        END { flush() }
    '
}

# Intel XPU. There is no vendor probe here like nvidia-smi / rocminfo -- Linux Intel support is
# an explicit index pin, not autodetection -- so the installed runtime IS the signal. The local
# label is read off disk first so a CPU-only host never pays for an `import torch`.
_setup_torch_is_xpu=false
_setup_xpu_ready=false
for _setup_tv in "$VENV_DIR"/lib/python*/site-packages/torch/version.py; do
    [ -f "$_setup_tv" ] || continue
    grep -q "^__version__ = '[^']*+xpu" "$_setup_tv" 2>/dev/null || continue
    _setup_torch_is_xpu=true
    # A +xpu wheel installs fine on a host whose driver never initialises, so only the runtime
    # answer reaches the summary below. Bounded, because a stalled Intel driver wedges inside
    # `import torch`: 60s rather than the smi probes' 10s, since a cold import is seconds on its
    # own. The SIGALRM deadline lives inside the probe too, for hosts without `timeout` (base
    # macOS, minimal Linux images). Either deadline expiring means "no XPU", as a failure does.
    _setup_xpu_probe='import signal; signal.alarm(60); import torch,sys; sys.exit(0 if torch.xpu.is_available() else 1)'
    if command -v timeout >/dev/null 2>&1; then
        timeout 60 "$VENV_DIR/bin/python" -c "$_setup_xpu_probe" >/dev/null 2>&1 && _setup_xpu_ready=true
    elif "$VENV_DIR/bin/python" -c "$_setup_xpu_probe" >/dev/null 2>&1; then
        _setup_xpu_ready=true
    fi
    break
done

# bitsandbytes carries XPU kernels (libbitsandbytes_xpu2025.so, _xpu2026.so) only from 0.50.0,
# and unsloth's own floor is 0.45.5, so a pre-XPU wheel satisfies it forever. install.sh raises
# the floor, but `unsloth studio update` runs THIS file, so an existing XPU user would keep a
# kernel-less bitsandbytes and lose 4-bit QLoRA. Keyed on the wheel, not the runtime: a stalled
# driver is no reason to skip the upgrade. --no-deps (torch and numpy are in), best effort.
if [ "$_setup_torch_is_xpu" = true ]; then
    # run_quiet_no_exit, NOT run_quiet: the latter exits via setup_fail, aborting an otherwise
    # fine `studio update` over a best-effort step and leaving the warning below unreachable.
    run_quiet_no_exit "install bitsandbytes (xpu)" fast_install --no-deps "bitsandbytes>=0.50.0" || \
        substep "[WARN] could not install an XPU-capable bitsandbytes; 4-bit QLoRA may be unavailable."
fi
# The supported name table, as a matcher so the report below can ask it about a PEER adapter
# without disturbing $_setup_gfx. Kept in sync with install.sh (and the PS nameArchTable).
# gfx1102 before gfx1100 so the spaceless "RX 7700S" lands on gfx1102 (case has no lookahead).
_setup_supported_gfx_from_name() {
    _sup_gfx_in="$1"
    _sup_gfx_out=""
    case "$_sup_gfx_in" in
        *9070*|*9080*|*"R9700"*)                                                                       _sup_gfx_out="gfx1201" ;;  # RDNA 4 (Navi 48: RX 9070 / 9080, Radeon AI PRO R9700)
        *9060*)                                                                                        _sup_gfx_out="gfx1200" ;;  # RDNA 4 (Navi 44)
        *"8065S"*|*"8060S"*|*"8050S"*|*"8040S"*|*"Strix Halo"*|*"Ryzen AI Max"*|*"AI Max"*) _sup_gfx_out="gfx1151" ;;  # RDNA 3.5 (Strix Halo + Gorgon Halo: Radeon 8065S/8060S/8050S/8040S iGPU, Ryzen AI Max / Max+)
        *"890M"*|*"880M"*|*"Strix Point"*|*"HX 37"*|*"AI 9 HX"*|*"AI 9 36"*) _sup_gfx_out="gfx1150" ;;  # RDNA 3.5 (Strix Point: Radeon 890M/880M, Ryzen AI 9 HX 370/375)
        *"860M"*|*"840M"*|*"Krackan"*|*"AI 7 35"*|*"AI 5 34"*|*"AI 7 PRO 35"*|*"AI 5 33"*) _sup_gfx_out="gfx1152" ;;  # RDNA 3.5 (Krackan Point: Radeon 860M/840M, Ryzen AI 7 350 / AI 5 340)
        *"RX 7600"*|*"RX 7700S"*|*"RX 7650"*|*"PRO W7600"*|*"PRO W7500"*)                              _sup_gfx_out="gfx1102" ;;  # RDNA 3 (Navi 33)
        *"RX 7800"*|*"RX 7700"*|*"PRO W7700"*|*"PRO V710"*)                                            _sup_gfx_out="gfx1101" ;;  # RDNA 3 (Navi 32)
        *"RX 7900"*|*"PRO W7900"*|*"PRO W7800"*)                                                       _sup_gfx_out="gfx1100" ;;  # RDNA 3 desktop / workstation (Navi 31)
        *"780M"*|*"760M"*|*"740M"*|*"Phoenix"*|*"Hawk Point"*|*"Z1 Extreme"*|*"Z2 Extreme"*)            _sup_gfx_out="gfx1103" ;;  # RDNA 3 iGPU (Phoenix / Hawk Point)
        *"RX 6900"*|*"RX 6800"*|*"RX 6750"*|*"RX 6700"*|*"PRO W6800"*|*"PRO W6900"*)                    _sup_gfx_out="gfx1030" ;;  # RDNA 2 (Navi 21)
        *"RX 6650"*|*"RX 6600"*|*"PRO W6600"*|*"PRO W6650"*)                                            _sup_gfx_out="gfx1032" ;;  # RDNA 2 (Navi 23)
        *"RX 6500"*|*"RX 6400"*|*"RX 6300"*|*"PRO W6400"*|*"PRO W6500"*)                                _sup_gfx_out="gfx1034" ;;  # RDNA 2 (Navi 24)
    esac
    [ -n "$_sup_gfx_out" ] || return 1
    printf '%s\n' "$_sup_gfx_out"
}

# NVIDIA priority: classify NVIDIA first and skip the AMD probes entirely on
# a usable-NVIDIA host (mirrors _has_rocm_gpu in install_python_stack.py).
# This also keeps a wedged rocminfo/amd-smi from hanging setup before the
# host is classified; the AMD probes themselves run under _setup_run_smi.
if _setup_has_usable_nvidia_gpu; then
    _setup_nvidia_usable=true
fi
if [ "$_setup_nvidia_usable" != true ]; then
    if command -v rocminfo >/dev/null 2>&1; then
        _setup_amd_records=$(_setup_run_smi rocminfo 2>/dev/null | _setup_rocminfo_gpu_records || true)
        _setup_gfx_all=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' '$1 != "" { print $1 }')
    fi
    if [ -n "$_setup_gfx_all" ]; then
        _setup_amd_detected=true
    elif command -v amd-smi >/dev/null 2>&1 && \
         _setup_run_smi amd-smi list 2>/dev/null | awk '/^GPU[[:space:]]*[:\[][[:space:]]*[0-9]/{ found=1 } END{ exit !found }'; then
        _setup_amd_detected=true
        # amd-smi owns the device list here, so its indexed records replace rocminfo's.
        _setup_amd_records=$(_setup_run_smi amd-smi static --asic 2>/dev/null | _setup_amd_smi_gpu_records || true)
        if [ -n "$_setup_amd_records" ]; then
            _setup_amd_smi_out=$(_setup_run_smi amd-smi list -e 2>/dev/null \
                | _setup_amd_smi_hip_order "$_setup_amd_records" || true)
            _setup_amd_space=$(printf '%s\n' "$_setup_amd_smi_out" | head -n 1)
            _setup_amd_records=$(printf '%s\n' "$_setup_amd_smi_out" | tail -n +2)
            # No map, and the adapters are not interchangeable: the mask indexes HIP order
            # while these records are in discovery order, so any ordinal is a guess. Decline
            # rather than forward a guessed --rocm-gfx to the llama.cpp and whisper
            # prebuilts. amd-smi 6.1.1 reports no TARGET_GRAPHICS_VERSION at all and the
            # arch is then inferred from the name, so an archless record is compared on its
            # name instead. Interchangeable adapters are unaffected: every ordinal gives the
            # same answer, and UNSLOTH_ROCM_GFX_ARCH still overrides below.
            if [ "$_setup_amd_space" != hip ] && \
               [ "$(printf '%s\n' "$_setup_amd_records" | awk -F'|' \
                    'NF { k = ($1 != "" ? $1 : "name:" $2); if (!(k in seen)) { seen[k]; n++ } }
                     END { print n + 0 }')" -gt 1 ]; then
                _setup_amd_records=""
                _setup_gfx_all=""
                _setup_hip_map_missing=1
            fi
        fi
        _setup_gfx_all=$(_setup_run_smi amd-smi list 2>/dev/null | grep -oE 'gfx[1-9][0-9a-z]{2,3}' || true)
        [ -z "$_setup_gfx_all" ] && \
            _setup_gfx_all=$(printf '%s\n' "$_setup_amd_records" | awk -F'|' '$1 != "" { print $1 }')
    elif [ -e /dev/kfd ] && \
         awk '/vendor_id/ && $2 == 4098 { found = 1 } END { exit !found }' \
             /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null; then
        # KFD sysfs fallback, AMD vendor_id 4098 only (mirrors install.sh
        # _has_amd_rocm_gpu): covers AMD hosts where rocminfo/amd-smi are
        # missing but the kernel exposes the GPU, so the source-build gate
        # below does not drop them to a CPU llama.cpp build. Neither a gfx arch
        # nor a marketing name is available from this path, so the report below
        # reads lspci rather than _setup_mkt when it needs to name the card.
        _setup_amd_detected=true
        _setup_amd_records=""
    fi
fi

if [ "$_setup_nvidia_usable" = true ]; then
    step "gpu" "NVIDIA GPU detected"
elif [ "$_setup_amd_detected" = true ]; then
    _setup_vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}"
    _setup_vis_idx=0
    if [ -n "$_setup_vis" ] && [ "$_setup_vis" != "-1" ]; then
        _setup_first="${_setup_vis%%,*}"
        case "$_setup_first" in ''|*[!0-9]*) ;; *) _setup_vis_idx=$_setup_first ;; esac
    fi
    if [ -n "$_setup_amd_records" ]; then
        # Records already preserve device ordinals, including duplicate arches.
        _setup_amd_record=$(printf '%s\n' "$_setup_amd_records" | awk -v idx="$_setup_vis_idx" \
            'NF { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
        _setup_gfx=${_setup_amd_record%%|*}
        _setup_mkt=${_setup_amd_record#*|}
    fi
    # Only pre-TARGET_GRAPHICS_VERSION amd-smi lands here: names but no arch in the record.
    if [ -z "$_setup_gfx" ]; then
        _setup_gfx=$(printf '%s\n' "$_setup_gfx_all" | awk -v idx="$_setup_vis_idx" \
            'NF && !seen[$0]++ { a[n++]=$0 } END { if(idx>=n) idx=0; if(n>0) print a[idx] }')
    fi
    # UNSLOTH_ROCM_GFX_ARCH env override (mirrors setup.ps1)
    if [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
        _setup_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
        substep "gfx arch from UNSLOTH_ROCM_GFX_ARCH env override: $_setup_gfx"
    # Name-based arch inference when tools don't report gfx (mirrors setup.ps1 nameArchTable)
    elif [ -z "$_setup_gfx" ] && [ -n "$_setup_mkt" ]; then
        _setup_gfx=$(_setup_supported_gfx_from_name "$_setup_mkt") || _setup_gfx=""
        if [ -n "$_setup_gfx" ]; then
            substep "gfx arch inferred from GPU name: $_setup_gfx"
            substep "Tip: set UNSLOTH_ROCM_GFX_ARCH=$_setup_gfx to skip inference next time"
        fi
    fi
    # Say why the arch is missing, since the user can supply it and amd-smi cannot.
    if [ -z "$_setup_gfx" ] && [ "$_setup_hip_map_missing" = 1 ]; then
        substep "Unlike AMD adapters and no HIP id map (amd-smi list -e needs ROCm 6.4+):"
        substep "cannot tell which one this session selects. Set UNSLOTH_ROCM_GFX_ARCH to pick."
    fi
    # ROCm version via hipconfig, then amd-smi
    _setup_rocm_ver=""
    if command -v hipconfig >/dev/null 2>&1; then
        _setup_rocm_ver=$(hipconfig --version 2>/dev/null | awk 'NR==1 && /^[0-9]/{print; exit}' || true)
    fi
    if [ -z "$_setup_rocm_ver" ] && command -v amd-smi >/dev/null 2>&1; then
        _setup_rocm_ver=$(amd-smi version 2>/dev/null | awk -F'ROCm version: ' \
            'NF>1{gsub(/[[:space:]]/,"", $2); print $2; exit}' || true)
    fi
    # GPU name -> gfx arch for AMD generations Unsloth's ROCm wheels do NOT cover: RDNA 1
    # and Polaris 10/20/30 (unslothai#8529). Kept apart from the inference table above on
    # purpose: it only words the report below, never selects a wheel index or prebuilt.
    # AMD's TheRock ships RDNA 1 wheels, but not on the repo.amd.com indexes routed here,
    # and never gfx803. Order is load-bearing: `case` has no negative lookahead, so the
    # RDNA 1 arms must precede Polaris or *"RX 570"* would swallow an "RX 5700 XT".
    # Names from LLVM's AMDGPU tables plus libdrm amdgpu.ids/pci.ids for the Navi 10/14
    # professional parts LLVM omits; nothing is guessed, so Polaris 11/12 is left out.
    # Case-sensitive, unlike the regex copies: every source here (WMI, amd-smi, lspci)
    # spells these names as pci.ids does.
    _setup_unsupported_gfx_from_name() {
        case "$1" in
            *"Radeon Pro V520"*|*"Radeon Pro 5600M"*) echo gfx1011 ;;  # RDNA 1
            *"RX 5700"*|*"RX 5600"*|*"Radeon Pro 5600 XT"*|*"Radeon Pro 5700"*|*"Radeon Pro W5700"*) echo gfx1010 ;;  # RDNA 1 (Navi 10)
            *"RX 5500"*|*"RX 5300"*|*"Radeon Pro W5500"*|*"Radeon Pro W5300"*) echo gfx1012 ;;  # RDNA 1 (Navi 14)
            *"RX 470"|*"RX 470"[!0]*|*"RX 480"|*"RX 480"[!0]*|*"RX 570"|*"RX 570"[!0]*|*"RX 580"|*"RX 580"[!0]*|*"RX 590"|*"RX 590"[!0]*|*"Radeon Pro WX 7100"*|*"Radeon Pro WX 5100"*) echo gfx803 ;;  # Polaris 10/20/30
            *) return 1 ;;
        esac
    }
    # The KFD sysfs fallback above detects the GPU with neither rocminfo nor amd-smi, so it
    # leaves _setup_mkt empty -- and a runtime-less host is precisely the one this report
    # exists for. lspci still names the card there, the source install.sh already reads.
    # Deliberately NOT written back into _setup_mkt: the supported table keys on it and
    # would start feeding --rocm-gfx to the prebuilt and whisper commands on the KFD path.
    _setup_unsupported_gfx_any() {
        # Peer guard FIRST, so it covers the named hit too: amd-smi reports one market name,
        # the first device's, so where an RX 5700 precedes an RX 7900 the name IS the 5700.
        # Only when lspci can answer: with no adapter list there is no peer to find, and
        # suppressing there would silence the single-card host this report exists for.
        _setup_unsup_pci=""
        if command -v lspci >/dev/null 2>&1; then
            _setup_unsup_pci=$(lspci -nn 2>/dev/null | grep -E 'VGA compatible controller|3D controller|Display controller' | grep -E 'AMD|ATI' || true)
            while IFS= read -r _setup_unsup_ln; do
                [ -n "$_setup_unsup_ln" ] || continue
                if _setup_supported_gfx_from_name "$_setup_unsup_ln" >/dev/null 2>&1; then
                    return 1
                fi
            done <<EOF
$_setup_unsup_pci
EOF
        fi
        # Only on a HIT: a nonempty but unmapped name (a generic "AMD Radeon Graphics"
        # from rocminfo) used to end the lookup here, so the lspci scan below never ran
        # and the report fell through to the plain "AMD ROCm" line this change replaces.
        if [ -n "$1" ] && _setup_unsup_named=$(_setup_unsupported_gfx_from_name "$1"); then
            echo "$_setup_unsup_named"
            return 0
        fi
        [ -n "$_setup_unsup_pci" ] || return 1
        while IFS= read -r _setup_unsup_ln; do
            [ -n "$_setup_unsup_ln" ] || continue
            if _setup_unsup_hit=$(_setup_unsupported_gfx_from_name "$_setup_unsup_ln"); then
                echo "$_setup_unsup_hit"
                return 0
            fi
        done <<EOF
$_setup_unsup_pci
EOF
        return 1
    }
    if [ -n "$_setup_gfx" ]; then
        step "gpu" "AMD ROCm ($_setup_gfx)"
    elif _setup_unsup_gfx=$(_setup_unsupported_gfx_any "$_setup_mkt"); then
        step "gpu" "AMD GPU detected ($_setup_unsup_gfx) -- no ROCm PyTorch wheels Unsloth installs"
        # Not "training runs on CPU": with no CUDA/XPU visible, unsloth raises
        # NotImplementedError at import (unsloth/device_type.py).
        # Both lines are false under an explicit index pin, which install_python_stack.py
        # honours for any arch, so a pinned run says what it is doing instead.
        # Whitespace-trimmed, as get_torch_index_url trims them: a blank value is unset
        # there, so treating it as a pin would drop the CPU warning for nothing.
        # Distinct name: _setup_pin is the XPU block's, and these are globals in POSIX sh.
        _setup_unsup_pin="${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}"
        _setup_unsup_pin=$(printf '%s' "$_setup_unsup_pin" | tr -d '[:space:]')
        if [ -n "$_setup_unsup_pin" ]; then
            substep "The torch index you pinned is used as given, so torch is whatever it publishes."
        else
            substep "torch stays CPU-only: Unsloth training and GPU inference are unavailable."
            substep "No HIP SDK install and no UNSLOTH_ROCM_GFX_ARCH value gives this GPU one."
        fi
        substep "GGUF chat can still use this GPU through Vulkan: export UNSLOTH_LLAMA_CPP_BACKEND=vulkan,"
        substep "then re-run the installer. It picks the llama.cpp bundle at install time, so setting"
        substep "it afterwards has no effect until you install or update again."
    else
        step "gpu" "AMD ROCm"
    fi
    _setup_rocm_root="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
    substep "ROCm: $_setup_rocm_root"
    [ -n "$_setup_rocm_ver" ] && substep "hipconfig: $_setup_rocm_ver"
    [ -n "$_setup_mkt" ] && [ -n "$_setup_gfx" ] && substep "GPU: $_setup_mkt"
elif [ "$_setup_xpu_ready" = true ]; then
    # Ranks below NVIDIA and AMD, as in setup.ps1: those hosts get their own wheels.
    step "gpu" "Intel GPU detected (XPU runtime)"
    substep "PyTorch XPU (SYCL) provides training and GPU inference on this GPU."
elif [ "$_setup_torch_is_xpu" = true ]; then
    # +xpu wheel installed but torch.xpu.is_available() said no: the Intel compute driver is
    # missing or too old. Falling through would call the hardware unsupported instead.
    step "gpu" "Intel GPU (XPU runtime unavailable)" "$C_WARN"
    substep "PyTorch has the XPU build but cannot initialise it -- update the Intel GPU compute driver."
    # Not "runs on CPU": with neither CUDA nor XPU, unsloth/device_type.py raises at import.
    # llama.cpp is unaffected, which is what chat and GGUF run on.
    substep "Until then training and GPU inference are unavailable; chat and GGUF still work."
elif [ "$(uname -s 2>/dev/null)" = "Darwin" ] && [ "$(uname -m 2>/dev/null)" = "arm64" ]; then
    # Apple Silicon: llama.cpp builds with Metal over unified memory, so not a CPU-only host.
    step "gpu" "Apple Silicon (Metal, unified memory)"
else
    step "gpu" "none (chat-only / GGUF)" "$C_WARN"
    substep "Training and GPU inference require an NVIDIA or AMD ROCm GPU."
fi

# ── 7. Prefer prebuilt llama.cpp bundles before any source build path ──
# Nest llama.cpp under $STUDIO_HOME only for real env-overrides; legacy
# default keeps ~/.unsloth/llama.cpp so pre-PR builds are still discovered.
if [ -n "$STAGE_ROOT" ]; then
    UNSLOTH_HOME="$RUNTIME_ROOT"
elif [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
    UNSLOTH_HOME="$STUDIO_HOME"
else
    UNSLOTH_HOME="$HOME/.unsloth"
fi
mkdir -p "$UNSLOTH_HOME"
LLAMA_CPP_DIR="$UNSLOTH_HOME/llama.cpp"
LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_DEGRADED=false
_LLAMA_CPP_NO_SPACE=false
_LLAMA_FORCE_COMPILE="${UNSLOTH_LLAMA_FORCE_COMPILE:-0}"
_REQUESTED_LLAMA_TAG="${UNSLOTH_LLAMA_TAG:-${_DEFAULT_LLAMA_TAG}}"
_HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
_HOST_MACHINE="$(uname -m 2>/dev/null || true)"
_source_backend_choice="$(printf '%s' "${UNSLOTH_LLAMA_CPP_BACKEND:-}" | awk '{$1=$1; print tolower($0)}')"
_source_legacy_force_vulkan="$(printf '%s' "${UNSLOTH_FORCE_VULKAN:-}" | awk '{$1=$1; print tolower($0)}')"
_explicit_llama_source_backend=""
if [ "$_HOST_SYSTEM" != "Darwin" ]; then
    case "$_source_backend_choice" in
        hip) _explicit_llama_source_backend="rocm" ;;
        cpu|cuda|rocm|vulkan) _explicit_llama_source_backend="$_source_backend_choice" ;;
        auto) ;;
        *)
            case "$_source_legacy_force_vulkan" in
                1|true|yes|on) _explicit_llama_source_backend="vulkan" ;;
            esac
            ;;
    esac
fi

# Pick the release repo install_llama_prebuilt.py plans against. Every host this
# installer supports now pulls its llama.cpp prebuilt from the unslothai fork: it
# ships the CUDA (Linux x64/arm64, Windows), ROCm (Linux/Windows) and macOS
# bundles, plus the CPU bundles for Linux/Windows on both x86_64 and arm64.
# ggml-org artifacts are no longer used by default.
_HELPER_RELEASE_REPO="unslothai/llama.cpp"
# UNSLOTH_ROCM_GFX_ARCH may be set on a host where no probe fired, so the override
# nested in the AMD-detected branch above never ran and _setup_gfx is still empty.
# Honour it here so the --rocm-gfx forwarding below still sees it
# (install_llama_prebuilt.py reads the same env var as the --rocm-gfx default).
if [ "${_setup_nvidia_usable:-}" != true ] && [ -z "${_setup_gfx:-}" ] && [ -n "${UNSLOTH_ROCM_GFX_ARCH:-}" ]; then
    _setup_gfx="${UNSLOTH_ROCM_GFX_ARCH}"
fi
_LLAMA_PR="${UNSLOTH_LLAMA_PR:-}"
_SKIP_PREBUILT_INSTALL=false
_LLAMA_PR_FORCE="${UNSLOTH_LLAMA_PR_FORCE:-${_DEFAULT_LLAMA_PR_FORCE}}"
_LLAMA_SOURCE="${_DEFAULT_LLAMA_SOURCE}"
_LLAMA_SOURCE="${_LLAMA_SOURCE%.git}"  # normalize: strip trailing .git
_RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
_RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
_RESOLVED_SOURCE_REF_KIND="tag"
_RESOLVED_LLAMA_TAG="$_REQUESTED_LLAMA_TAG"

if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
    _NEED_LLAMA_SOURCE_BUILD=true
    _SKIP_PREBUILT_INSTALL=true
fi

# Baked-in PR_FORCE promotes to _LLAMA_PR when user hasn't set one.
if [ -z "$_LLAMA_PR" ] && [ -n "$_LLAMA_PR_FORCE" ] && \
   [[ "$_LLAMA_PR_FORCE" =~ ^[0-9]+$ ]] && [ "$_LLAMA_PR_FORCE" -gt 0 ]; then
    _LLAMA_PR="$_LLAMA_PR_FORCE"
    step "llama.cpp" "baked-in PR_FORCE=$_LLAMA_PR_FORCE" "$C_WARN"
fi

if [ -n "$_LLAMA_PR" ]; then
    if ! [[ "$_LLAMA_PR" =~ ^[0-9]+$ ]] || [ "$_LLAMA_PR" -le 0 ]; then
        step "llama.cpp" "UNSLOTH_LLAMA_PR=$_LLAMA_PR is not a valid PR number" "$C_ERR"
        setup_fail 1 "UNSLOTH_LLAMA_PR=$_LLAMA_PR is not a valid PR number"
    fi
    step "llama.cpp" "UNSLOTH_LLAMA_PR=$_LLAMA_PR -- will build from PR head" "$C_WARN"
    _RESOLVED_LLAMA_TAG="pr-$_LLAMA_PR"
    _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
    _RESOLVED_SOURCE_REF="pr-$_LLAMA_PR"
    _RESOLVED_SOURCE_REF_KIND="pull"
    _NEED_LLAMA_SOURCE_BUILD=true
    _SKIP_PREBUILT_INSTALL=true
fi

verbose_substep "requested llama.cpp tag: $_REQUESTED_LLAMA_TAG (repo: $_HELPER_RELEASE_REPO)"

# GGUF export's check_llama_cpp() looks for a llama-quantize shim at the root of
# the install dir, but a source build keeps the binary under build/bin/. Mirror
# the source-build-reuse step and create the shim when the reused tree has one
# but no root shim yet. Best-effort: the tree may be read-only (shared/CI cache),
# and under `set -e` a failed ln would otherwise abort an good reuse.
_link_local_llama_quantize_shim() {
    if [ -x "$1/build/bin/llama-quantize" ] && [ ! -e "$1/llama-quantize" ]; then
        ln -sf build/bin/llama-quantize "$1/llama-quantize" 2>/dev/null || \
            substep "could not create llama-quantize shim in linked dir (read-only?); GGUF export may be unavailable"
    fi
}

# Accept any layout LlamaCppBackend._layout_candidates() resolves so the flag
# never rejects a tree Unsloth could actually run: a root-level llama-server (a
# `make` build or a flat-extracted release) or the CMake build/bin/llama-server.
_has_local_llama_server() {
    [ -x "$1/llama-server" ] || [ -x "$1/build/bin/llama-server" ]
}

_LOCAL_LLAMA_CPP_LINKED=false
if [ -n "${UNSLOTH_LOCAL_LLAMA_CPP_DIR:-}" ]; then
    if [ ! -d "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" ]; then
        # A build under an unsearchable ancestor cannot be stat'd, so report permissions
        # rather than sending the user to fix a path that is already correct.
        _report_denied_ancestor "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" "UNSLOTH_LOCAL_LLAMA_CPP_DIR"
        step "llama.cpp" "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $UNSLOTH_LOCAL_LLAMA_CPP_DIR" "$C_ERR"
        setup_fail 1 "UNSLOTH_LOCAL_LLAMA_CPP_DIR does not exist: $UNSLOTH_LOCAL_LLAMA_CPP_DIR"
    fi
    # In an if condition so a denied dir reports instead of tripping errexit.
    if ! _RESOLVED_LOCAL="$(CDPATH= cd -P -- "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" 2>/dev/null && pwd -P)"; then
        # owner-unverified: this is the user's own tree, never advise deleting it.
        _path_access_denied "$UNSLOTH_LOCAL_LLAMA_CPP_DIR" "UNSLOTH_LOCAL_LLAMA_CPP_DIR" owner-unverified
    fi
    # Canonicalize the install path the same way before comparing: _RESOLVED_LOCAL
    # is fully resolved, but LLAMA_CPP_DIR is textual ($UNSLOTH_HOME/llama.cpp). If
    # $HOME (or UNSLOTH_HOME) contains a symlink, the two never match even when the
    # user pointed the flag at the canonical install itself -- and the rm -rf below
    # would then wipe the very tree they asked to reuse. Resolve via the parent so
    # this works whether or not the leaf currently exists.
    _CANON_LLAMA_CPP_DIR="$LLAMA_CPP_DIR"
    _LLAMA_CPP_PARENT="$(dirname "$LLAMA_CPP_DIR")"
    if [ -d "$_LLAMA_CPP_PARENT" ]; then
        # Nothing can be written under a parent we cannot search, so report here
        # rather than let the link below abort raw a few lines later.
        if _canon_parent="$(CDPATH= cd -P -- "$_LLAMA_CPP_PARENT" 2>/dev/null && pwd -P)"; then
            _CANON_LLAMA_CPP_DIR="$_canon_parent/$(basename "$LLAMA_CPP_DIR")"
        else
            _path_access_denied "$_LLAMA_CPP_PARENT" "Unsloth install directory" owner-unverified
        fi
    fi
    if [ "$_RESOLVED_LOCAL" = "$_CANON_LLAMA_CPP_DIR" ]; then
        # Points at the canonical install location itself: never delete-then-link
        # it onto itself. If a usable build is already there, reuse it and skip
        # both the prebuilt download and the source build -- the prebuilt installer
        # uses os.replace() and would otherwise clobber an existing source build at
        # this path. If nothing is built there yet, fall through to the normal
        # install so it gets built in place exactly as it would without the flag.
        if _has_local_llama_server "$LLAMA_CPP_DIR"; then
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR is the canonical install location and already holds a build; reusing it"
            _link_local_llama_quantize_shim "$LLAMA_CPP_DIR"
            _LOCAL_LLAMA_CPP_LINKED=true
            _NEED_LLAMA_SOURCE_BUILD=false
            _SKIP_PREBUILT_INSTALL=true
        else
            substep "UNSLOTH_LOCAL_LLAMA_CPP_DIR points to the canonical install location with nothing built there yet; running the normal install"
        fi
    else
        # Reusing disables BOTH the prebuilt download and the source build, so the
        # linked tree must already contain a runnable llama-server in one of the
        # layouts the backend resolves (root-level or build/bin/). Fail clearly
        # rather than link an unbuilt or wrong-platform checkout and leave Unsloth
        # with no usable binary.
        if ! _has_local_llama_server "$_RESOLVED_LOCAL"; then
            step "llama.cpp" "no llama-server under $_RESOLVED_LOCAL (looked for ./llama-server and ./build/bin/llama-server) -- build llama.cpp there first, or drop --with-llama-cpp-dir" "$C_ERR"
            setup_fail 1 "No llama-server was found under $_RESOLVED_LOCAL"
        fi
        # A stale link from a previous --with-llama-cpp-dir run isn't Unsloth-owned
        # content; drop it before the ownership check so re-runs stay idempotent
        # for a custom UNSLOTH_STUDIO_HOME (the assert would otherwise follow the
        # link into the user's dir and reject it as unowned).
        [ -L "$LLAMA_CPP_DIR" ] && rm -f "$LLAMA_CPP_DIR"
        if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
        fi
        rm -rf "$LLAMA_CPP_DIR" || true
        if [ -e "$LLAMA_CPP_DIR" ]; then
            # Unreadable, not just unsearchable: mode 111 defeats the rm above and
            # would fall through to the generic message.
            if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
                _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
            fi
            step "llama.cpp" "the existing install could not be replaced with a link" "$C_ERR"
            setup_fail 3 "$LLAMA_CPP_DIR could not be replaced with a link to $_RESOLVED_LOCAL."
        fi
        ln -sfn "$_RESOLVED_LOCAL" "$LLAMA_CPP_DIR"
        _link_local_llama_quantize_shim "$LLAMA_CPP_DIR"
        step "llama.cpp" "linked local directory: $_RESOLVED_LOCAL"
        _LOCAL_LLAMA_CPP_LINKED=true
        _NEED_LLAMA_SOURCE_BUILD=false
        _SKIP_PREBUILT_INSTALL=true
    fi
fi

# Every branch below replaces $LLAMA_CPP_DIR or builds into it, and the source-build
# swap only reaches its own guards after the whole build, so check here instead.
# Local-link paths are excluded: they already replaced or reused the tree above.
if [ "$_LOCAL_LLAMA_CPP_LINKED" != true ]; then
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
    if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
        _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
fi

if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then
    : # local directory linked above; skip prebuilt install
elif [ -n "$_explicit_llama_source_backend" ] && [ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then
    step "llama.cpp" "$_explicit_llama_source_backend was explicitly requested, but this installation requires a source build" "$C_ERR"
    substep "Explicit backend selection requires a matching prebuilt bundle; allow prebuilts or unset UNSLOTH_LLAMA_CPP_BACKEND"
    setup_fail 1 "$_explicit_llama_source_backend was explicitly requested, but this installation requires a source build. Explicit backend selection requires a matching prebuilt bundle."
elif [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
    step "llama.cpp" "UNSLOTH_LLAMA_FORCE_COMPILE=1 -- skipping prebuilt" "$C_WARN"
    _NEED_LLAMA_SOURCE_BUILD=true
elif [ "${_SKIP_PREBUILT_INSTALL:-false}" = true ]; then
    substep "prebuilt install skipped -- falling back to source build"
else
    substep "installing prebuilt llama.cpp..."
    if [ -d "$LLAMA_CPP_DIR" ]; then
        substep "existing install detected -- validating update"
    fi
    # why: install_llama_prebuilt.py uses os.replace(), which would displace
    # an unrelated $UNSLOTH_STUDIO_HOME/llama.cpp before the source-build
    # ownership check below ever runs.
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
    # The ownership check above misses the default cache; stop before pathlib
    # turns an unreadable one into a traceback.
    if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
        _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
    fi
    _PREBUILT_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "$_HELPER_RELEASE_REPO"
    )
    if [ -n "${UNSLOTH_LLAMA_RELEASE_TAG:-}" ]; then
        _PREBUILT_CMD+=(--published-release-tag "$UNSLOTH_LLAMA_RELEASE_TAG")
    fi
    # Forward the gfx arch resolved above so the per-gfx ROCm prebuilt is picked
    # even when the installer's own probe cannot report it (amd-smi-only hosts,
    # name-inferred arch). Implies --has-rocm on the installer side.
    if [ -n "${_setup_gfx:-}" ]; then
        _PREBUILT_CMD+=(--rocm-gfx "$_setup_gfx")
    elif [ "$_setup_amd_detected" = true ] && \
         { command -v hipcc >/dev/null 2>&1 || [ -x /opt/rocm/bin/hipcc ] || \
           ls /opt/rocm-*/bin/hipcc >/dev/null 2>&1; }; then
        # AMD detected but gfx unknown (KFD-only host): forward --has-rocm only when
        # hipcc can actually build llama.cpp (incl. a versioned /opt/rocm-*/bin, the
        # same paths the source build uses). With no gfx the prebuilt resolver finds
        # no ROCm bundle and the source build would fail, so without hipcc fall
        # through to the CPU prebuilt instead of breaking the install.
        _PREBUILT_CMD+=(--has-rocm)
    fi
    # Reporting only: the installer reads UNSLOTH_LLAMA_CPP_BACKEND itself, and it
    # is also the only side that can see a choice recorded in the install marker,
    # so forwarding a second copy from here could only ever disagree with it. The
    # override affects llama.cpp alone, not the training backend.
    case "$_source_backend_choice" in
        cpu)
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                step "llama.cpp" "UNSLOTH_LLAMA_CPP_BACKEND=cpu has no effect on macOS (universal build; use -ngl 0 at runtime for CPU-only)" "$C_WARN" >&2
            fi
            ;;
        vulkan)
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                step "llama.cpp" "Vulkan has no effect on macOS; the universal build uses Metal" "$C_WARN" >&2
            else
                step "llama.cpp" "Vulkan selected for GGUF inference; the PyTorch training backend is unchanged" "$C_OK"
            fi
            ;;
        ""|auto|cuda|hip|rocm) ;;
        *) step "llama.cpp" "Ignoring UNSLOTH_LLAMA_CPP_BACKEND='$_source_backend_choice' (expected 'auto', 'cpu', 'cuda', 'vulkan', 'hip', or 'rocm')" "$C_WARN" >&2 ;;
    esac
    _PREBUILT_LOG="$(mktemp)"
    set +e
    if _is_verbose; then
        "${_PREBUILT_CMD[@]}" 2>&1 | tee "$_PREBUILT_LOG"
        _PREBUILT_STATUS=${PIPESTATUS[0]}
    else
        "${_PREBUILT_CMD[@]}" >"$_PREBUILT_LOG" 2>&1
        _PREBUILT_STATUS=$?
    fi
    set -e

    if [ "$_PREBUILT_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_PREBUILT_LOG"; then
            step "llama.cpp" "prebuilt up to date and validated"
        else
            step "llama.cpp" "prebuilt installed and validated"
        fi
        if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$LLAMA_CPP_DIR" ]; then
            : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
        fi
        print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
        verbose_substep "llama.cpp install dir: $LLAMA_CPP_DIR"
        rm -f "$_PREBUILT_LOG"
    elif [ "$_PREBUILT_STATUS" -eq 3 ]; then
        step "llama.cpp" "install blocked by active llama.cpp process" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "existing install was restored"
        fi
        substep "close Unsloth or other llama.cpp users and retry"
        setup_fail 3 "llama.cpp install is blocked by an active llama.cpp process"
    elif [ "$_PREBUILT_STATUS" -eq 4 ]; then
        step "llama.cpp" "not enough disk space to install llama.cpp" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        substep "free up disk or move UNSLOTH_STUDIO_HOME/TMPDIR to a larger volume, then re-run"
        _LLAMA_CPP_NO_SPACE=true
        _has_local_llama_server "$LLAMA_CPP_DIR" || _LLAMA_CPP_DEGRADED=true
        # A preserved server may not satisfy an explicit backend request, and it
        # leaves _LLAMA_CPP_DEGRADED false. Never report success on an unverified
        # backend after the requested replacement ran out of space.
        if [ -n "$_explicit_llama_source_backend" ]; then
            step "llama.cpp" "$_explicit_llama_source_backend was explicitly requested, so the installer will not keep an unverified existing backend" "$C_ERR"
            setup_fail 1 "$_explicit_llama_source_backend was explicitly requested, so the installer will not keep an unverified existing llama.cpp backend."
        fi
    elif [ "$_PREBUILT_STATUS" -eq 5 ]; then
        step "llama.cpp" "selected backend could not be installed" "$C_ERR"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "prebuilt update failed; existing install restored"
        fi
        substep "check the error above, choose another backend, or retry"
        setup_fail 1 "The selected llama.cpp backend could not be installed, so the installer will not substitute a different source backend."
    elif [ "$_PREBUILT_STATUS" -eq 2 ]; then
        step "llama.cpp" "prebuilt install failed" "$C_WARN"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "prebuilt update failed; existing install restored"
        fi
        # Exit 2 means no concrete backend was in play: a request the installer
        # could not honour -- named here or recorded in the install marker, which
        # this script cannot see -- exits 5 above instead.
        substep "falling back to source build"
        _NEED_LLAMA_SOURCE_BUILD=true
    else
        step "llama.cpp" "prebuilt helper failed unexpectedly" "$C_ERR"
        print_llama_error_log "$_PREBUILT_LOG"
        rm -f "$_PREBUILT_LOG"
        if [ -d "$LLAMA_CPP_DIR" ]; then
            substep "existing install was restored or left unchanged"
        fi
        substep "source build was not started because it cannot repair an unexpected helper or permissions error"
        setup_fail 1 "llama.cpp prebuilt helper failed unexpectedly (exit code $_PREBUILT_STATUS). Check the error above and retry setup."
    fi
fi

# Source-built llama.cpp installs do not have the prebuilt metadata used above
# for exact release matching. Reuse a complete local source build unless the
# caller explicitly requested a rebuild or a PR-specific llama.cpp checkout.
if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && \
   [ "$_LLAMA_FORCE_COMPILE" != "1" ] && \
   [ -z "$_LLAMA_PR" ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ] && \
   [ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    step "llama.cpp" "existing source build found; skipping rebuild"
    ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
    fi
    _NEED_LLAMA_SOURCE_BUILD=false
fi

if [ -n "$STAGE_ROOT" ] && [ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then
    setup_fail 1 "Background staging cannot install system build tools for llama.cpp; retry with the foreground updater."
fi

# ── 8. WSL: pre-install GGUF build dependencies for fallback source builds ──
# On WSL, sudo requires a password and can't be entered during GGUF export
# (runs in a non-interactive subprocess). Install build deps here instead.
if [ "$_NEED_LLAMA_SOURCE_BUILD" = true ] && grep -qi microsoft /proc/version 2>/dev/null; then
    _GGUF_DEPS="pciutils build-essential cmake curl git libcurl4-openssl-dev"
    apt-get update -y >/dev/null 2>&1 || true
    apt-get install -y $_GGUF_DEPS >/dev/null 2>&1 || true

    _STILL_MISSING=""
    for _pkg in $_GGUF_DEPS; do
        case "$_pkg" in
            build-essential) command -v gcc >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            pciutils) command -v lspci >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            libcurl4-openssl-dev) command -v curl-config >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
            *) command -v "$_pkg" >/dev/null 2>&1 || _STILL_MISSING="$_STILL_MISSING $_pkg" ;;
        esac
    done
    _STILL_MISSING=$(echo "$_STILL_MISSING" | sed 's/^ *//')

    if [ -z "$_STILL_MISSING" ]; then
        step "gguf deps" "installed"
    elif command -v sudo >/dev/null 2>&1; then
        step "gguf deps" "sudo required for: $_STILL_MISSING" "$C_WARN"
        if _can_read_tty; then
            printf "  %-15s" ""
            printf "accept? [Y/n] "
            # The device opened, so a failed read is EOF, not consent: decline.
            read -r REPLY </dev/tty || REPLY="n"
            case "$REPLY" in
                [nN]*)
                    substep "skipped -- run manually:"
                    substep "sudo apt-get install -y $_STILL_MISSING"
                    _SKIP_GGUF_BUILD=true
                    ;;
                *)
                    # Degrade like the no-sudo branch below rather than letting
                    # set -e abort setup on a bare apt error: missing GGUF build
                    # deps are recoverable, not fatal.
                    if sudo apt-get update -y </dev/null &&
                        sudo apt-get install -y $_STILL_MISSING </dev/null; then
                        step "gguf deps" "installed"
                    else
                        step "gguf deps" "install failed -- run manually:" "$C_WARN"
                        substep "sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                        _SKIP_GGUF_BUILD=true
                    fi
                    ;;
            esac
        else
            # Nobody can answer a prompt or type a password here, so -n makes
            # sudo refuse rather than prompt into a closed stdin, and -k ignores
            # any cached timestamp so only a real NOPASSWD rule gets through.
            # Same treatment as install.sh's _smart_apt_install. This is the WSL
            # GGUF-export case noted above, where sudo does want a password.
            if sudo -n -k apt-get update -y </dev/null &&
                sudo -n -k apt-get install -y $_STILL_MISSING </dev/null; then
                step "gguf deps" "installed (non-interactive sudo)"
            else
                step "gguf deps" "needs sudo, no terminal -- run manually:" "$C_WARN"
                substep "sudo apt-get update -y && sudo apt-get install -y $_STILL_MISSING"
                _SKIP_GGUF_BUILD=true
            fi
        fi
    else
        step "gguf deps" "missing (no sudo) -- install manually:" "$C_WARN"
        substep "apt-get install -y $_STILL_MISSING"
        _SKIP_GGUF_BUILD=true
    fi
fi

# ── 9. Build llama.cpp binaries for GGUF inference + export when prebuilt install fails ──
# Builds at ~/.unsloth/llama.cpp — a single shared location under the user's
# home directory. This is used by both the inference server and the GGUF
# export pipeline (unsloth-zoo).
#   - llama-server: for GGUF model inference
#   - llama-quantize: for GGUF export quantization (symlinked to root for check_llama_cpp())
if [ "$_NEED_LLAMA_SOURCE_BUILD" = false ]; then
    :
elif [ "${_SKIP_GGUF_BUILD:-}" = true ]; then
    step "llama.cpp" "skipped (missing build deps)" "$C_WARN"
    [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
else
{
    if ! command -v cmake &>/dev/null; then
        step "llama.cpp" "skipped (cmake not found)" "$C_WARN"
        [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
    elif ! command -v git &>/dev/null; then
        step "llama.cpp" "skipped (git not found)" "$C_WARN"
        [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
    else
        if [ -z "$_LLAMA_PR" ]; then
            _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
            if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then
                if [ "$_REQUESTED_LLAMA_TAG" = "latest" ]; then
                    _RESOLVED_SOURCE_REF="${UNSLOTH_LLAMA_FORCE_COMPILE_REF:-${_DEFAULT_LLAMA_FORCE_COMPILE_REF}}"
                    _RESOLVED_SOURCE_REF_KIND="branch"
                else
                    _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
                    _RESOLVED_SOURCE_REF_KIND="tag"
                fi
            elif [ "$_REQUESTED_LLAMA_TAG" = "latest" ]; then
                _RESOLVE_TAG_ARGS=(--resolve-llama-tag latest --published-repo "ggml-org/llama.cpp" --output-format json)
                set +e
                _RESOLVE_TAG_JSON="$(python "$SCRIPT_DIR/install_llama_prebuilt.py" "${_RESOLVE_TAG_ARGS[@]}" 2>/dev/null)"
                _RESOLVE_TAG_STATUS=$?
                set -e
                if [ "$_RESOLVE_TAG_STATUS" -eq 0 ] && [ -n "${_RESOLVE_TAG_JSON:-}" ]; then
                    _RESOLVED_SOURCE_REF="$(
                        printf '%s' "$_RESOLVE_TAG_JSON" | python -c 'import json,sys; print(json.load(sys.stdin).get("llama_tag",""))' 2>/dev/null || true
                    )"
                else
                    _RESOLVED_SOURCE_REF=""
                fi
                if [ -z "$_RESOLVED_SOURCE_REF" ]; then
                    _RESOLVED_SOURCE_REF="latest"
                fi
                _RESOLVED_SOURCE_REF_KIND="tag"
            else
                _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
                _RESOLVED_SOURCE_REF_KIND="tag"
            fi
            if [ -z "$_RESOLVED_SOURCE_URL" ]; then
                _RESOLVED_SOURCE_URL="$_LLAMA_SOURCE"
            fi
            if [ -z "$_RESOLVED_SOURCE_REF" ]; then
                _RESOLVED_SOURCE_REF="$_REQUESTED_LLAMA_TAG"
            fi
        fi
        verbose_substep "source build repo: $_RESOLVED_SOURCE_URL"
        verbose_substep "source build ref: ${_RESOLVED_SOURCE_REF:-latest} (${_RESOLVED_SOURCE_REF_KIND})"
        BUILD_OK=true
        mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
        _BUILD_TMP="${LLAMA_CPP_DIR}.build.$$"
        rm -rf "$_BUILD_TMP"
        if [ -n "$_LLAMA_PR" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_LLAMA_SOURCE}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch PR #$_LLAMA_PR" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "pull/$_LLAMA_PR/head:pr-$_LLAMA_PR" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout PR #$_LLAMA_PR" \
                    git -C "$_BUILD_TMP" checkout "pr-$_LLAMA_PR" || BUILD_OK=false
            fi
        elif [ "$_RESOLVED_SOURCE_REF_KIND" = "pull" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch source PR ref" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "$_RESOLVED_SOURCE_REF" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout source PR ref" \
                    git -C "$_BUILD_TMP" checkout -B unsloth-llama-build FETCH_HEAD || BUILD_OK=false
            fi
        elif [ "$_RESOLVED_SOURCE_REF_KIND" = "commit" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
            run_quiet_no_exit "clone llama.cpp" \
                git clone --depth 1 "${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP" || BUILD_OK=false
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "fetch source commit" \
                    git -C "$_BUILD_TMP" fetch --depth 1 origin "$_RESOLVED_SOURCE_REF" || BUILD_OK=false
            fi
            if [ "$BUILD_OK" = true ]; then
                run_quiet_no_exit "checkout source commit" \
                    git -C "$_BUILD_TMP" checkout -B unsloth-llama-build FETCH_HEAD || BUILD_OK=false
            fi
        else
            _CLONE_ARGS=(git clone --depth 1)
            if [ "$_RESOLVED_SOURCE_REF" != "latest" ] && [ -n "$_RESOLVED_SOURCE_REF" ]; then
                _CLONE_ARGS+=(--branch "$_RESOLVED_SOURCE_REF")
            fi
            _CLONE_ARGS+=("${_RESOLVED_SOURCE_URL}.git" "$_BUILD_TMP")
            run_quiet_no_exit "clone llama.cpp" \
                "${_CLONE_ARGS[@]}" || BUILD_OK=false
        fi

        if [ "$BUILD_OK" = true ]; then
            # Set Release explicitly (llama.cpp only defaults to it on non-MSVC/Xcode).
            CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=ON"
            _TRY_METAL_CPU_FALLBACK=false
            _HOST_SYSTEM="$(uname -s 2>/dev/null || true)"
            _HOST_MACHINE="$(uname -m 2>/dev/null || true)"
            _IS_MACOS_ARM64=false
            if [ "$_HOST_SYSTEM" = "Darwin" ] && { [ "$_HOST_MACHINE" = "arm64" ] || [ "$_HOST_MACHINE" = "aarch64" ]; }; then
                _IS_MACOS_ARM64=true
            fi

            # macOS: pin a low deployment target so the source build loads on
            # older macOS too (else a macOS 26 host stamps minos=26). Set before
            # CPU_FALLBACK_CMAKE_ARGS copies CMAKE_ARGS so both paths inherit it.
            if [ "$_HOST_SYSTEM" = "Darwin" ]; then
                _MACOS_DEPLOYMENT_TARGET="${UNSLOTH_MACOS_DEPLOYMENT_TARGET:-13.3}"
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_OSX_DEPLOYMENT_TARGET=${_MACOS_DEPLOYMENT_TARGET}"
                export MACOSX_DEPLOYMENT_TARGET="${_MACOS_DEPLOYMENT_TARGET}"
            fi

            if command -v ccache &>/dev/null; then
                CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
            fi
            CPU_FALLBACK_CMAKE_ARGS="$CMAKE_ARGS"

            GPU_BACKEND=""
            NVCC_PATH=""
            # Gate the CUDA toolkit search on an actually-usable NVIDIA GPU
            # (_setup_nvidia_usable, computed in the GPU summary block above;
            # already false when hidden via CUDA_VISIBLE_DEVICES=""/-1).
            # A CUDA toolkit alone (CPU-only build container, leftover packages)
            # is not proof of a GPU: building with -DGGML_CUDA=ON there yields a
            # binary that fails at runtime, so fall through to the CPU build.
            if [ "$_setup_nvidia_usable" = true ]; then
                if command -v nvcc &>/dev/null; then
                    NVCC_PATH="$(command -v nvcc)"
                    GPU_BACKEND="cuda"
                elif [ -x /usr/local/cuda/bin/nvcc ]; then
                    NVCC_PATH="/usr/local/cuda/bin/nvcc"
                    export PATH="/usr/local/cuda/bin:$PATH"
                    GPU_BACKEND="cuda"
                elif ls /usr/local/cuda-*/bin/nvcc &>/dev/null 2>&1; then
                    # Pick the newest cuda-XX.X directory
                    NVCC_PATH="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
                    export PATH="$(dirname "$NVCC_PATH"):$PATH"
                    GPU_BACKEND="cuda"
                fi
            fi

            # Check for ROCm (AMD) only if CUDA was not already selected, and
            # only when an AMD GPU was actually detected (_setup_amd_detected).
            # hipcc presence alone (HIP SDK, no GPU) must not select a HIP build.
            # NVIDIA-usable hosts never build HIP (defense in depth: the AMD
            # probes above are already skipped when NVIDIA is usable).
            ROCM_HIPCC=""
            if [ -z "$GPU_BACKEND" ] && [ "$_setup_nvidia_usable" != true ] && [ "$_setup_amd_detected" = true ]; then
                if command -v hipcc &>/dev/null; then
                    ROCM_HIPCC="$(command -v hipcc)"
                    GPU_BACKEND="rocm"
                elif [ -x /opt/rocm/bin/hipcc ]; then
                    ROCM_HIPCC="/opt/rocm/bin/hipcc"
                    export PATH="/opt/rocm/bin:$PATH"
                    GPU_BACKEND="rocm"
                elif ls /opt/rocm-*/bin/hipcc &>/dev/null 2>&1; then
                    ROCM_HIPCC="$(ls -d /opt/rocm-*/bin/hipcc 2>/dev/null | sort -V | tail -1)"
                    export PATH="$(dirname "$ROCM_HIPCC"):$PATH"
                    GPU_BACKEND="rocm"
                fi
            fi

            _BUILD_DESC="building"
            if [ "$_IS_MACOS_ARM64" = true ]; then
                # Metal takes precedence on Apple Silicon (CUDA/ROCm not functional on macOS)
                _BUILD_DESC="building (Metal)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_USE_BF16=ON -DCMAKE_INSTALL_RPATH=@loader_path -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
                CPU_FALLBACK_CMAKE_ARGS="$CPU_FALLBACK_CMAKE_ARGS -DGGML_METAL=OFF"
                _TRY_METAL_CPU_FALLBACK=true
            elif [ -n "$NVCC_PATH" ]; then
                # Returns "ok|too_old|unknown\nX.Y" on stdout.
                _NVCC_CHECK="$(_nvcc_meets_llama_minimum "$NVCC_PATH")"
                _NVCC_STATUS="$(printf '%s\n' "$_NVCC_CHECK" | sed -n '1p')"
                _NVCC_VER="$(printf '%s\n' "$_NVCC_CHECK" | sed -n '2p')"

                if [ "$_NVCC_STATUS" = "too_old" ]; then
                    substep "CUDA toolkit $_NVCC_VER is below llama.cpp minimum (12.4)." "$C_ERR"
                    substep "install a newer CUDA toolkit: https://developer.nvidia.com/cuda-toolkit-archive" "$C_WARN"
                    substep "falling back to CPU llama.cpp build for this run." "$C_WARN"
                    NVCC_PATH=""
                    GPU_BACKEND=""
                    _BUILD_DESC="building (CPU, CUDA toolkit < 12.4)"
                else
                    _DRIVER_MAX_CUDA="$(_cuda_driver_max_version)"
                    _CUDA_TOOLKIT_ALLOWED=true
                    if [ -n "$_NVCC_VER" ] && [ -n "$_DRIVER_MAX_CUDA" ] && \
                       _cuda_toolkit_major_gt_driver "$_NVCC_VER" "$_DRIVER_MAX_CUDA"; then
                        _BLOCKED_NVCC_VER="$_NVCC_VER"
                        if _ALT_NVCC_CHECK="$(_cuda_find_compatible_nvcc_for_driver "$_DRIVER_MAX_CUDA" "$NVCC_PATH")"; then
                            NVCC_PATH="$(printf '%s\n' "$_ALT_NVCC_CHECK" | sed -n '1p')"
                            _NVCC_VER="$(printf '%s\n' "$_ALT_NVCC_CHECK" | sed -n '2p')"
                            GPU_BACKEND="cuda"
                            export PATH="$(dirname "$NVCC_PATH"):$PATH"
                            substep "CUDA Toolkit $_BLOCKED_NVCC_VER is a major-version mismatch with driver CUDA $_DRIVER_MAX_CUDA; using compatible CUDA Toolkit $_NVCC_VER at $NVCC_PATH." "$C_WARN"
                        else
                            _print_cuda_driver_toolkit_mismatch "$_NVCC_VER" "$_DRIVER_MAX_CUDA"
                            substep "falling back to CPU llama.cpp build for this run." "$C_WARN"
                            NVCC_PATH=""
                            GPU_BACKEND=""
                            _BUILD_DESC="building (CPU, CUDA toolkit major > driver)"
                            _CUDA_TOOLKIT_ALLOWED=false
                        fi
                    fi

                    if [ "$_CUDA_TOOLKIT_ALLOWED" = true ]; then
                        # Resolve the arch list before committing to a CUDA build;
                        # an empty list means CPU instead of a PTX-only binary (#5854).
                        _raw_caps=""
                        # Resolve nvidia-smi as _setup_has_usable_nvidia_gpu does
                        # (PATH, then /usr/bin); `command -v` alone would miss an
                        # off-PATH binary and wrongly drop a CUDA host to CPU.
                        _smi_bin=""
                        if command -v nvidia-smi >/dev/null 2>&1; then
                            _smi_bin="nvidia-smi"
                        elif [ -x "/usr/bin/nvidia-smi" ]; then
                            _smi_bin="/usr/bin/nvidia-smi"
                        fi
                        if [ -n "$_smi_bin" ]; then
                            _raw_caps=$(_setup_run_smi "$_smi_bin" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
                        fi
                        CUDA_ARCHS="$(_resolve_cuda_archs "$_raw_caps" "${UNSLOTH_LLAMA_CUDA_ARCHS:-}")"

                        if [ -n "$CUDA_ARCHS" ]; then
                            CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
                            CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_FLAGS=--threads=0"
                            _BUILD_DESC="building (CUDA, sm_${CUDA_ARCHS//;/+sm_})"

                            # Allow a host gcc/clang newer than nvcc's whitelist (else a fresh
                            # toolkit aborts with "unsupported GNU version"); via env to avoid word-splitting.
                            export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-allow-unsupported-compiler"
                        else
                            # No detectable arch: build CPU (CMAKE_ARGS has no
                            # -DGGML_CUDA=ON yet, so clearing GPU_BACKEND yields CPU).
                            substep "could not detect a CUDA compute capability; building CPU llama.cpp instead of a PTX-only binary (set UNSLOTH_LLAMA_CUDA_ARCHS, e.g. \"120\", to force a CUDA build)." "$C_WARN"
                            GPU_BACKEND=""
                            _BUILD_DESC="building (CPU, CUDA arch undetectable)"
                        fi
                    fi
                fi
            elif [ "$GPU_BACKEND" = "rocm" ]; then
                # Resolve hipcc symlinks to find the real ROCm root
                _HIPCC_REAL="$(readlink -f "$ROCM_HIPCC" 2>/dev/null || printf '%s' "$ROCM_HIPCC")"
                ROCM_ROOT=""
                if command -v hipconfig &>/dev/null; then
                    ROCM_ROOT="$(hipconfig -R 2>/dev/null || true)"
                fi
                if [ -z "$ROCM_ROOT" ]; then
                    ROCM_ROOT="$(cd "$(dirname "$_HIPCC_REAL")/.." 2>/dev/null && pwd)"
                fi

                _BUILD_DESC="building (ROCm)"
                CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"

                # ROCm 7.x ships clang-20 which on Ubuntu 24.04+ defaults to the
                # highest-numbered gcc lib dir (/usr/lib/gcc/x86_64-linux-gnu/14/)
                # which contains runtime objects but NOT C++ headers, causing:
                #   fatal error: 'cstdlib' file not found
                # Find the newest gcc install dir that actually has both the
                # runtime dir AND /usr/include/c++/<ver> headers, then pass it
                # to clang via --gcc-install-dir so HIP builds succeed.
                _GCC_INSTALL_DIR=""
                _gcc_pm="$(gcc -print-multiarch 2>/dev/null)"
                case "$_gcc_pm" in
                    *-linux-gnu*) _GCC_MULTIARCH="$_gcc_pm" ;;
                    *) _GCC_MULTIARCH="$(uname -m)-linux-gnu" ;;
                esac
                for _gcc_ver in 14 13 12 11; do
                    if [ -d "/usr/lib/gcc/$_GCC_MULTIARCH/$_gcc_ver/include" ] && \
                       [ -d "/usr/include/c++/$_gcc_ver" ]; then
                        _GCC_INSTALL_DIR="/usr/lib/gcc/$_GCC_MULTIARCH/$_gcc_ver"
                        break
                    fi
                done
                if [ -n "$_GCC_INSTALL_DIR" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_HIP_FLAGS=--gcc-install-dir=\"$_GCC_INSTALL_DIR\""
                    substep "ROCm HIP gcc install dir: $_GCC_INSTALL_DIR"
                fi

                export ROCM_PATH="$ROCM_ROOT"
                export HIP_PATH="$ROCM_ROOT"

                # Use upstream-recommended HIP compiler (not legacy hipcc-as-CXX)
                if command -v hipconfig &>/dev/null; then
                    _HIP_CLANG_DIR="$(hipconfig -l 2>/dev/null || true)"
                    [ -n "$_HIP_CLANG_DIR" ] && export HIPCXX="$_HIP_CLANG_DIR/clang"
                fi

                # Detect AMD GPU architecture (gfx target)
                GPU_TARGETS=""
                if command -v rocminfo &>/dev/null; then
                    _gfx_list=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9]{2,4}[a-z]?' | sort -u || true)
                    _valid_gfx=""
                    for _gfx in $_gfx_list; do
                        if [[ "$_gfx" =~ ^gfx[0-9]{2,4}[a-z]?$ ]]; then
                            # Drop bare family-level targets (gfx10, gfx11, gfx12, ...)
                            # when a specific sibling is present in the same list.
                            # rocminfo on ROCm 6.1+ emits both the specific GPU and
                            # the LLVM generic family line (e.g. gfx1100 alongside
                            # gfx11-generic), and the outer grep above captures the
                            # bare family prefix from the generic line. Passing that
                            # bare prefix to -DGPU_TARGETS breaks the HIP/llama.cpp
                            # build because clang only accepts specific gfxNNN ids.
                            # No real AMD GPU has a 2-digit gfx id, so this filter
                            # can only ever drop family prefixes, never real targets.
                            if [[ "$_gfx" =~ ^gfx[0-9]{2}$ ]] \
                               && echo "$_gfx_list" | grep -qE "^${_gfx}[0-9][0-9a-z]?$"; then
                                continue
                            fi
                            _valid_gfx="${_valid_gfx}${_valid_gfx:+;}$_gfx"
                        fi
                    done
                    [ -n "$_valid_gfx" ] && GPU_TARGETS="$_valid_gfx"
                fi

                if [ -n "$GPU_TARGETS" ]; then
                    CMAKE_ARGS="$CMAKE_ARGS -DGPU_TARGETS=${GPU_TARGETS}"
                    _BUILD_DESC="building (ROCm, ${GPU_TARGETS//;/+})"
                fi
            elif [ -d /usr/local/cuda ] || _setup_run_smi nvidia-smi &>/dev/null; then
                _BUILD_DESC="building (CPU, CUDA driver found but nvcc missing)"
            elif [ -d /opt/rocm ] || command -v rocm-smi &>/dev/null; then
                _BUILD_DESC="building (CPU, ROCm driver found but hipcc missing)"
            else
                _BUILD_DESC="building (CPU)"
            fi

            substep "$_BUILD_DESC..."

            NCPU=$(_llama_build_jobs)
            verbose_substep "parallel jobs: $NCPU (RAM-capped; UNSLOTH_LLAMA_BUILD_JOBS overrides)"
            CMAKE_GENERATOR_ARGS=""
            if command -v ninja &>/dev/null; then
                CMAKE_GENERATOR_ARGS="-G Ninja"
            fi

            # GPU label for the CPU-fallback message: Metal, else GPU_BACKEND
            # (cuda/rocm). Empty on a bare CPU build (nothing to fall back from).
            _gpu_fallback_label() {
                if [ "$_TRY_METAL_CPU_FALLBACK" = true ]; then
                    echo "Metal"
                elif [ -n "$GPU_BACKEND" ]; then
                    printf '%s' "$GPU_BACKEND" | tr '[:lower:]' '[:upper:]'
                fi
            }

            if ! run_quiet_no_exit "cmake llama.cpp" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CMAKE_ARGS; then
                _FB_LABEL="$(_gpu_fallback_label)"
                if [ -n "$_FB_LABEL" ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "$_FB_LABEL configure failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL configure failed)"
                        # Now configured for CPU; clear GPU_BACKEND so a later
                        # build-step failure won't re-enter fallback on this config.
                        GPU_BACKEND=""
                    else
                        BUILD_OK=false
                    fi
                else
                    BUILD_OK=false
                fi
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            if ! run_quiet_no_exit "build llama-server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU"; then
                _FB_LABEL="$(_gpu_fallback_label)"
                if [ -n "$_FB_LABEL" ]; then
                    _TRY_METAL_CPU_FALLBACK=false
                    substep "$_FB_LABEL build failed; retrying CPU build..." "$C_WARN"
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL build failed)"
                        GPU_BACKEND=""
                        run_quiet_no_exit "build llama-server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
                    else
                        BUILD_OK=false
                    fi
                else
                    BUILD_OK=false
                fi
            fi
        fi

        if [ "$BUILD_OK" = true ]; then
            run_quiet_no_exit "build llama-quantize" cmake --build "$_BUILD_TMP/build" --config Release --target llama-quantize -j"$NCPU" || true
            # Best-effort: the DiffusionGemma visual server (an example target, present
            # on llama.cpp PR #24423). No-op when the diffusion example is not configured.
            run_quiet_no_exit "build diffusion visual server" cmake --build "$_BUILD_TMP/build" --config Release --target llama-diffusion-gemma-visual-server -j"$NCPU" || true
        fi

        # Opt-in post-build GPU smoke test (#5854 gap 2). Default off (Blackwell
        # CUDA JIT stalls). On failure, reuse the CPU fallback path so the user
        # still gets a working llama-server. Runs before the install swap.
        if [ "$BUILD_OK" = true ] && _staged_validation_enabled; then
            _FB_LABEL="$(_gpu_fallback_label)"
            _SMOKE_KIND="$(_source_smoke_install_kind)"
            if [ -n "$_FB_LABEL" ]; then
                _SMOKE_CMD=(
                    python "$SCRIPT_DIR/install_llama_prebuilt.py"
                    --validate-install "$_BUILD_TMP"
                )
                [ -n "$_SMOKE_KIND" ] && _SMOKE_CMD+=(--install-kind "$_SMOKE_KIND")
                _SMOKE_RC=0
                run_quiet_no_exit "validate source llama.cpp" "${_SMOKE_CMD[@]}" || _SMOKE_RC=$?
                # Exit 4 is a full disk, not a bad build: the CPU rebuild needs even
                # more space, so keep what we already have.
                if [ "$_SMOKE_RC" -eq 4 ]; then
                    substep "not enough disk space to validate the $_FB_LABEL build; keeping it" "$C_WARN"
                    _LLAMA_CPP_NO_SPACE=true
                elif [ "$_SMOKE_RC" -ne 0 ]; then
                    substep "$_FB_LABEL source build failed smoke test; retrying CPU build..." "$C_WARN"
                    _TRY_METAL_CPU_FALLBACK=false
                    rm -rf "$_BUILD_TMP/build"
                    if run_quiet_no_exit "cmake llama.cpp (cpu fallback)" cmake $CMAKE_GENERATOR_ARGS -S "$_BUILD_TMP" -B "$_BUILD_TMP/build" $CPU_FALLBACK_CMAKE_ARGS; then
                        _BUILD_DESC="building (CPU fallback after $_FB_LABEL smoke failed)"
                        GPU_BACKEND=""
                        run_quiet_no_exit "build llama-server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-server -j"$NCPU" || BUILD_OK=false
                        if [ "$BUILD_OK" = true ]; then
                            run_quiet_no_exit "build llama-quantize (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-quantize -j"$NCPU" || true
                            run_quiet_no_exit "build diffusion visual server (cpu fallback)" cmake --build "$_BUILD_TMP/build" --config Release --target llama-diffusion-gemma-visual-server -j"$NCPU" || true
                        fi
                    else
                        BUILD_OK=false
                    fi
                fi
            fi
        fi

        # Swap only after build succeeds -- preserves existing install on failure
        if [ "$BUILD_OK" = true ]; then
            _assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"
            # || true: without it a raw rm error aborts under errexit to a bare exit
            # code, build stranded. Keep stderr: rm names the exact subpath, we cannot.
            rm -rf "$LLAMA_CPP_DIR" || true
            if [ -e "$LLAMA_CPP_DIR" ]; then
                # Same probe as the other replace sites: the hoisted guard covers a
                # tree already denied, this catches one denied mid-build.
                if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then
                    _path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"
                fi
                step "llama.cpp" "built, but the existing install could not be replaced" "$C_ERR"
                setup_fail 3 "The llama.cpp build succeeded but $LLAMA_CPP_DIR could not be replaced. The new build is at $_BUILD_TMP."
            fi
            mv "$_BUILD_TMP" "$LLAMA_CPP_DIR"
            : > "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
            # Symlink to llama.cpp root -- check_llama_cpp() looks for the binary there
            QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
            if [ -f "$QUANTIZE_BIN" ]; then
                ln -sf build/bin/llama-quantize "$LLAMA_CPP_DIR/llama-quantize"
            fi
            # DiffusionGemma visual server, if it was built (PR #24423): link next to
            # llama-server so Unsloth serves DiffusionGemma GGUFs without DG_VISUAL_BIN.
            if [ -f "$LLAMA_CPP_DIR/build/bin/llama-diffusion-gemma-visual-server" ]; then
                ln -sf build/bin/llama-diffusion-gemma-visual-server "$LLAMA_CPP_DIR/llama-diffusion-gemma-visual-server"
            fi
        else
            rm -rf "$_BUILD_TMP"
        fi

        if [ "$BUILD_OK" = true ] && [ -f "$LLAMA_SERVER_BIN" ]; then
            step "llama.cpp" "built"
            [ -f "$LLAMA_CPP_DIR/llama-quantize" ] && step "llama-quantize" "built"
        elif [ "$BUILD_OK" = true ]; then
            step "llama.cpp" "binary not found after build" "$C_WARN"
            _LLAMA_CPP_DEGRADED=true
        else
            step "llama.cpp" "build failed" "$C_ERR"
            [ -f "$LLAMA_SERVER_BIN" ] || _LLAMA_CPP_DEGRADED=true
        fi
    fi
}
fi  # end _SKIP_GGUF_BUILD check

# ── arm64 Linux GPU: CPU prebuilt as a last resort ──
# An arm64 Linux GPU host source-builds for the GPU above. If that produced no
# binary, install the fork's arm64 CPU prebuilt (app-<tag>-linux-arm64-cpu.tar.gz)
# instead of leaving the host without llama.cpp. --cpu-fallback drops the GPU
# attributes so the CPU bundle is selected rather than re-attempting CUDA. Skipped
# on a full disk: the retry fails the same way and buries the hint.
if [ "$_LLAMA_CPP_DEGRADED" = true ] \
        && [ "$_LLAMA_CPP_NO_SPACE" != true ] \
        && [ "$_HOST_SYSTEM" = "Linux" ] \
        && { [ "$_HOST_MACHINE" = "aarch64" ] || [ "$_HOST_MACHINE" = "arm64" ]; }; then
    substep "GPU source build unavailable; trying arm64 CPU prebuilt..."
    _ARM64_CPU_CMD=(
        python "$SCRIPT_DIR/install_llama_prebuilt.py"
        --install-dir "$LLAMA_CPP_DIR"
        --llama-tag "$_REQUESTED_LLAMA_TAG"
        --published-repo "unslothai/llama.cpp"
        --cpu-fallback
    )
    # Trust the installer's exit code: it validates the server before exiting 0,
    # the same signal the primary prebuilt path above relies on.
    if run_quiet_no_exit "arm64 CPU prebuilt" "${_ARM64_CPU_CMD[@]}"; then
        step "llama.cpp" "arm64 CPU prebuilt installed (GPU build unavailable)" "$C_WARN"
        _LLAMA_CPP_DEGRADED=false
        print_installed_llama_prebuilt_release "$LLAMA_CPP_DIR"
    fi
fi

if [ ! -L "$LLAMA_CPP_DIR" ] && {
    [ "$_STUDIO_HOME_IS_CUSTOM" != true ] ||
        [ -f "$LLAMA_CPP_DIR/$_STUDIO_OWNED_MARKER" ] ||
        _studio_owned_adoptable "$LLAMA_CPP_DIR"
}; then
    _remove_agent_instruction_files "$LLAMA_CPP_DIR"
fi

# ── whisper.cpp (local speech-to-text dictation engine) ──
# Optional runtime for local dictation. Fail-open: any failure leaves the
# Transformers STT engine and browser dictation working, so it never aborts
# setup (unlike llama.cpp). Runs in 'unsloth studio update' too so the runtime
# installs/refreshes without a compiler. Installs beside llama.cpp under the
# same managed home the sidecar's _managed_whisper_cpp_dir() resolves.
WHISPER_CPP_DIR="$UNSLOTH_HOME/whisper.cpp"
if [ -n "${WHISPER_SERVER_PATH:-}" ] || [ -n "${UNSLOTH_WHISPER_CPP_PATH:-}" ]; then
    verbose_substep "whisper.cpp: using a user-configured binary/dir; skipping managed install"
elif [ "${UNSLOTH_SKIP_WHISPER_INSTALL:-0}" = "1" ]; then
    verbose_substep "whisper.cpp: install skipped (UNSLOTH_SKIP_WHISPER_INSTALL=1)"
else
    if [ "$_STUDIO_HOME_IS_CUSTOM" = true ]; then
        _assert_studio_owned_or_absent "$WHISPER_CPP_DIR" "whisper.cpp install"
    fi
    _WHISPER_CMD=(python "$SCRIPT_DIR/install_whisper_prebuilt.py" --install-dir "$WHISPER_CPP_DIR")
    if [ -n "${UNSLOTH_WHISPER_RELEASE_TAG:-}" ]; then
        _WHISPER_CMD+=(--published-release-tag "$UNSLOTH_WHISPER_RELEASE_TAG")
    fi
    if [ -n "${_setup_gfx:-}" ]; then
        _WHISPER_CMD+=(--rocm-gfx "$_setup_gfx")
    elif [ "$_setup_amd_detected" = true ]; then
        _WHISPER_CMD+=(--has-rocm)
    fi
    _WHISPER_LOG="$(mktemp)"
    set +e
    if _is_verbose; then
        "${_WHISPER_CMD[@]}" 2>&1 | tee "$_WHISPER_LOG"
        _WHISPER_STATUS=${PIPESTATUS[0]}
    else
        "${_WHISPER_CMD[@]}" >"$_WHISPER_LOG" 2>&1
        _WHISPER_STATUS=$?
    fi
    set -e
    if [ "$_WHISPER_STATUS" -eq 0 ]; then
        if grep -Fq "already matches" "$_WHISPER_LOG"; then
            step "whisper.cpp" "prebuilt up to date"
        else
            step "whisper.cpp" "prebuilt installed"
        fi
        if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$WHISPER_CPP_DIR" ]; then
            : > "$WHISPER_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
        fi
        rm -f "$_WHISPER_LOG"
    elif [ "$_WHISPER_STATUS" -eq 3 ]; then
        # A warm dictation server holds the binary; keep the old install.
        step "whisper.cpp" "install busy; keeping existing runtime" "$C_WARN"
        rm -f "$_WHISPER_LOG"
    else
        # A source build is opt-in. Keep the installer log until fallback has
        # finished so setup can distinguish release skew from an operational
        # installer failure and report the exact pairing when available.
        _WHISPER_RECOVERED=false
        _WHISPER_BUILD="$SCRIPT_DIR/../scripts/build_whisper_cpp.sh"
        if [ "${UNSLOTH_WHISPER_FORCE_COMPILE:-0}" = "1" ] && [ -f "$_WHISPER_BUILD" ] \
                && command -v cmake >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
            substep "whisper.cpp prebuilt unavailable; building from source (UNSLOTH_WHISPER_FORCE_COMPILE=1)..."
            # The source build overwrites whisper-server in the managed dir but
            # knows nothing about the prebuilt marker; a stale marker would make
            # a later setup run report "already matches" and skip repairing the
            # prebuilt over the source binary. Drop it before building.
            rm -f "$WHISPER_CPP_DIR/UNSLOTH_WHISPER_PREBUILT_INFO.json" 2>/dev/null || true
            if run_quiet_no_exit "whisper.cpp source build" \
                    env UNSLOTH_HOME="$UNSLOTH_HOME" sh "$_WHISPER_BUILD"; then
                _WHISPER_RECOVERED=true
                step "whisper.cpp" "source build installed"
                if [ "$_STUDIO_HOME_IS_CUSTOM" = true ] && [ -d "$WHISPER_CPP_DIR" ]; then
                    : > "$WHISPER_CPP_DIR/$_STUDIO_OWNED_MARKER" 2>/dev/null || true
                fi
            else
                :
            fi
        fi
        if [ "$_WHISPER_RECOVERED" != true ]; then
            if [ "$_WHISPER_STATUS" -eq 2 ]; then
                _WHISPER_REQUIRED_TAG="$(sed -n 's/.*slim bundle requires llama\.cpp \([^; ]*\).*/\1/p' "$_WHISPER_LOG" | tail -n 1)"
                _WHISPER_INSTALLED_TAG="$(python - "$UNSLOTH_HOME/llama.cpp/UNSLOTH_PREBUILT_INFO.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("release_tag", ""))
except Exception:
    pass
PY
)"
                _WHISPER_PAIRING="installed llama.cpp ${_WHISPER_INSTALLED_TAG:-unknown}; whisper requires ${_WHISPER_REQUIRED_TAG:-unknown}"
                step "whisper.cpp" "no compatible prebuilt ($_WHISPER_PAIRING); curated whisper.cpp dictation is unavailable; publish the paired releases in llama.cpp then whisper.cpp order; browser and Transformers dictation remain available" "$C_WARN"
            else
                step "whisper.cpp" "prebuilt install failed; curated whisper.cpp dictation is unavailable; retry setup or inspect verbose output; browser and Transformers dictation remain available" "$C_WARN"
            fi
        fi
        rm -f "$_WHISPER_LOG"
    fi
fi

# ── Footer ──
if [ "$_LLAMA_ONLY" = "1" ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "llama.cpp update finished (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "llama.cpp update finished"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
elif [ "$IS_COLAB" = true ]; then
    echo ""
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "Unsloth Studio Setup Complete (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Setup Complete"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    substep "from colab import start"
    substep "start()"
else
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_WARN}%s${C_RST}\n" "Unsloth Studio Installed (limited: llama.cpp unavailable)"
    else
        printf "  ${C_TITLE}%s${C_RST}\n" "Unsloth Studio Installed"
    fi
    printf "  ${C_DIM}%s${C_RST}\n" "$RULE"
    if [ "$_LLAMA_CPP_DEGRADED" = true ]; then
        printf "  ${C_DIM}%-15s${C_WARN}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    else
        printf "  ${C_DIM}%-15s${C_OK}%s${C_RST}\n" "launch" "unsloth studio -p 8888"
    fi
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 for LAN / cloud access; exposes the raw port only, not a public URL)"
    printf "  ${C_DIM}%-15s%s${C_RST}\n" "" "(add -H 0.0.0.0 --cloudflare for a public Cloudflare HTTPS link, or --secure to keep the raw port private; anyone with the API key can run code)"
fi
echo ""

# When called from install.sh (SKIP_STUDIO_BASE=1), exit non-zero so the
# installer can report the GGUF failure after finishing PATH/shortcut setup.
# When called directly via 'unsloth studio update', keep the install
# successful -- the footer above already reports the limitation and Unsloth
# is still usable for non-GGUF workflows.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" = "1" ]; then
    # In Tauri mode a non-zero exit is not "report", it is "abort": install.rs turns the
    # error into "Installation failed", so one transient prebuilt download failure (a
    # single HTTP 403 rate limit will do it) fails the whole first-launch install of an
    # app whose own footer just said Installed. Everything except GGUF inference works,
    # and whisper.cpp in this same script already degrades rather than failing for
    # exactly this case. Match it, and say what is missing and how to get it back.
    #
    # PROGRESS, not STEP: install.rs maps [TAURI:STEP] to the install-step event, and
    # use-tauri-backend.ts counts those against the seven-entry INSTALL_STEPS list that
    # install.sh already emits in full, so an eighth marker renders "Step 8 of 7" and
    # discards the payload. [TAURI:PROGRESS] becomes install-progress-detail, which
    # InstallingContent renders verbatim, so the user actually reads the limitation.
    #
    # DIAG as well: progress detail is cleared by the next install-step and is gone
    # once the install screen closes, so it cannot answer "why is GGUF missing"
    # afterwards. record_diag_marker keeps this in the support report.
    case "${UNSLOTH_TAURI_MODE:-0}" in
        1|true)
            printf '[TAURI:PROGRESS] %s\n' \
                "llama.cpp unavailable; GGUF inference is disabled until 'unsloth studio update' succeeds"
            printf '[TAURI:DIAG] %s\n' "llama_cpp=unavailable"
            ;;
        *)
            setup_fail 1 "llama.cpp setup did not produce a usable server"
            ;;
    esac
fi

# A desktop repair runs update.rs, which sets UNSLOTH_TAURI_UPDATE alone, so the
# block above is skipped and a degraded repair recorded nothing. update.rs parses
# [TAURI:DIAG] the same way. Marker only: the update contract stays successful.
if [ "$_LLAMA_CPP_DEGRADED" = true ] && [ "${SKIP_STUDIO_BASE:-0}" != "1" ]; then
    case "${UNSLOTH_TAURI_UPDATE:-0}" in
        1|true) printf '[TAURI:DIAG] %s\n' "llama_cpp=unavailable" ;;
    esac
fi
