# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: llama-server capability probe."""

import re
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import (
    Iterable,
    Optional,
)
from core.inference.llama_server_args import (
    parse_reasoning_budget_message_override,
    parse_reasoning_budget_override,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
)
from utils.code_integrity import code_integrity_block_reason
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .log import logger
from .server_log import _TRACE_VERBOSITY_HELP_RE


class _CapabilityProbeMixin:
    # ── llama-server capability probe ─────────────────────────────

    # Prebuilts based on llama.cpp b10259..b10268 advertise draft-dspark but land between
    # the reshape regression (ggml-org/llama.cpp#26531) and its fix (#26577), so a DSpark
    # launch aborts on load. Matched on the base build number, not one exact tag: every
    # release in that window is affected. Source builds carry no install marker, so only
    # managed prebuilts are gated.
    _BROKEN_DSPARK_BUILDS = range(10259, 10269)

    @classmethod
    def _dspark_release_is_broken(cls, release_tag: Optional[str]) -> bool:
        match = re.match(r"b(\d+)", str(release_tag or ""))
        return bool(match) and int(match.group(1)) in cls._BROKEN_DSPARK_BUILDS

    # Keyed on the file's revision, since `unsloth studio update` replaces it in place.
    # Nanoseconds and size, not int(st_mtime): an update landing in the same second as
    # the probe kept the key identical and got the old build's capabilities.
    _CAPABILITY_PROBE_RETRY_SECONDS = 30.0
    _CAPABILITY_PROBE_RETRY_MAX_SECONDS = 600.0
    _capability_cache: dict[tuple[str, int, int], dict[str, object]] = {}
    # (bin_path, mtime_ns, size) -> build number from `llama-server --version`, or None.
    _build_number_cache: dict[tuple[str, int, int], Optional[int]] = {}
    _capability_retry_after: dict[tuple[str, int, int], float] = {}
    _capability_retry_backoff: dict[tuple[str, int, int], float] = {}
    _capability_cache_lock = threading.Lock()

    # The value form of the flash-attention flag. Newer llama.cpp declares it
    # with an enum, "-fa, --flash-attn [on|off|auto]"; older builds take a bare
    # boolean and reject a following "on" as a stray positional, which is an
    # immediate "invalid argument" exit rather than a degraded launch.
    _FLASH_ATTN_ENUM_RE = re.compile(
        r"(?im)^[^\n]*--flash-attn[^\n]*\bon\b[ \t]*\|[ \t]*\boff\b[^\n]*$"
    )

    @classmethod
    def _flash_attn_takes_value(cls, help_text: str) -> bool:
        """Whether this build's --flash-attn accepts on/off/auto.

        Fails open: a build whose help never mentions the flag at all keeps
        today's behaviour, because the pinned prebuilt is the value form and a
        wrong answer there would break the supported path for a hypothetical
        one.
        """
        if not help_text or "--flash-attn" not in help_text:
            return True
        return bool(cls._FLASH_ATTN_ENUM_RE.search(help_text))

    # ggml-org/llama.cpp#23861 capped the output rows a context reserves at the rows it
    # can emit. Builds before it reserve a row for every micro-batch token.
    _OUTPUT_ROWS_CAPPED_BUILD = 9460

    @classmethod
    def probe_build_number(cls, binary: Optional[str] = None) -> Optional[int]:
        """The llama.cpp build number from ``llama-server --version``, cached per binary.

        Current builds print ``version: X.Y.Z (build NNNN, commit ...)``, older ones
        ``version: NNNN (commit)``. None when the binary is missing, the probe fails,
        or the build reports 1 (a source build without tags)."""
        bin_path = cls._exec_path_for_launch(binary or cls._find_llama_server_binary())
        if not bin_path or not Path(bin_path).is_file():
            return None
        try:
            binary_stat = Path(bin_path).stat()
            cache_key = (bin_path, binary_stat.st_mtime_ns, binary_stat.st_size)
        except OSError:
            cache_key = (bin_path, 0, 0)
        with cls._capability_cache_lock:
            if cache_key in cls._build_number_cache:
                return cls._build_number_cache[cache_key]
        build: Optional[int] = None
        try:
            env = cls._llama_server_env_for_binary(bin_path)
            if sys.platform == "darwin":
                env["GGML_METAL_DEVICES"] = "0"  # see probe_server_capabilities
            result = subprocess.run(
                [bin_path, "--version"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            match = re.search(r"version:[^\r\n]*\bbuild\s+(\d+)\b", output) or re.search(
                r"version:\s*(\d+)\b(?!\.)", output
            )
            if match and int(match.group(1)) > 1:
                build = int(match.group(1))
        except Exception as exc:
            logger.debug("llama-server --version probe failed: %s", exc)
        with cls._capability_cache_lock:
            cls._build_number_cache[cache_key] = build
        return build

    @classmethod
    def reserves_micro_batch_outputs(cls, binary: Optional[str] = None) -> bool:
        """Whether this llama-server reserves an output row per micro-batch token.

        Known builds before b9460, including the older-macOS b9415 fallback, do.
        Unknown builds are priced as current, matching Studio's prebuilts."""
        try:
            build = cls.probe_build_number(binary)
        except Exception:
            return False
        return build is not None and build < cls._OUTPUT_ROWS_CAPPED_BUILD

    # ggml-org/llama.cpp#23089 added continuation inside an unfinished reasoning block.
    _THOUGHT_CONTINUATION_BUILD = 9200

    @classmethod
    def resumes_thoughts(cls, binary: Optional[str] = None) -> bool:
        """Whether this llama-server continues a trailing turn that holds only reasoning.

        Unknown builds are treated as current, matching Studio's prebuilts."""
        try:
            build = cls.probe_build_number(binary)
        except Exception:
            return True
        return build is None or build >= cls._THOUGHT_CONTINUATION_BUILD

    _ADVERTISED_DEFAULT_RE = re.compile(r"\(default:\s*(-?\d+)")

    @classmethod
    def _advertised_int_default(cls, help_block: Optional[str]) -> Optional[int]:
        """The integer a flag's help block declares as its default, else None.

        llama.cpp wraps the block across columns, so the caller passes the joined
        text. None means "this build did not say", never a numeric guess.
        """
        if not help_block:
            return None
        match = cls._ADVERTISED_DEFAULT_RE.search(help_block)
        return int(match.group(1)) if match else None

    @classmethod
    def probe_server_capabilities(cls, binary: Optional[str] = None) -> dict[str, object]:
        """Parse `llama-server --help` for feature flags. Returns
        {found, mtp_token, supports_mtp, supports_dspark, supports_dflash,
        ngram_mod_flavor, supports_ngram_mod, spec_draft_n_max_flag, cache flag
        support}.

        ``ngram_mod_flavor``: ``"new"`` when the post-rename
        ``--spec-ngram-mod-n-match / -n-min / -n-max`` are real args;
        ``"legacy"`` when only the pre-rename
        ``--spec-ngram-size-n / --draft-min / --draft-max`` are real (the
        rename ships stub removal entries for legacy names, told apart by
        the "argument has been removed" description); ``None`` if neither
        set is usable.

        ``spec_draft_n_max_flag``: the flag the binary accepts --
        ``--spec-draft-n-max`` post-rename, ``--draft-max`` on legacy.
        ``None`` means n_max cannot be set.
        """
        # Not only in load_model: startup, status and preflight probe directly,
        # and a wrapper loses DYLD_* to SIP before reaching the real server. The
        # inconclusive capabilities that follow clamp slots and disable
        # DSpark/DFlash sizing on a server that supports both (#8566).
        bin_path = cls._exec_path_for_launch(binary or cls._find_llama_server_binary())
        if not bin_path or not Path(bin_path).is_file():
            return {
                "found": False,
                "mtp_token": None,
                "supports_mtp": False,
                "supports_dspark": False,
                "supports_dflash": False,
                "supports_eagle3": False,
                "mtp_probe_inconclusive": True,
                "ngram_mod_flavor": None,
                "supports_ngram_mod": False,
                "spec_draft_n_max_flag": None,
                "spec_draft_n_max_default": None,
                "supports_kv_unified": False,
                # Fail OPEN, unlike every key above. These three are emitted on
                # every launch today, so a failed probe must keep emitting them;
                # only a build whose --help positively lacks one drops it.
                "supports_no_context_shift": True,
                "supports_jinja": True,
                "supports_api_key_file": True,
                "supports_flash_attn": True,
                "flash_attn_takes_value": True,
                "supports_fit_ctx": False,
                "supports_fit_target": False,
                "supports_cache_ram": False,
                "supports_ctx_checkpoints": False,
                "ctx_checkpoints_flag": None,
                "ctx_checkpoints_default": None,
                "supports_no_cache_prompt": False,
                "supports_metrics": False,
                "supports_slot_save": False,
                "supports_reasoning_budget": False,
                "supports_reasoning_budget_message": False,
                "reasoning_budget_probe_inconclusive": True,
                # Fails CLOSED, unlike --jinja: a build without --reasoning exits on it.
                "supports_reasoning_flag": False,
                "supports_no_mmproj_offload": False,
                "supports_trace_verbosity": False,
                "supports_video_fps": False,
                "supports_load_mode": False,
                "supports_moe_cache": False,
                "supports_moe_cache_auto": False,
                "supports_lazy_mode": False,
                "lazy_mode_flag": None,
                "spec_draft_ngl_flag": None,
                "spec_draft_cache_k_flag": None,
                "spec_draft_cache_v_flag": None,
                "flags": {},
                "help_probe_ok": False,
            }
        try:
            binary_stat = Path(bin_path).stat()
            mtime_ns = binary_stat.st_mtime_ns
            size = binary_stat.st_size
        except OSError:
            mtime_ns = 0
            size = 0
        cache_key = (bin_path, mtime_ns, size)
        with cls._capability_cache_lock:
            cached = cls._capability_cache.get(cache_key)
            if cached is not None:
                if not cached.get("mtp_probe_inconclusive"):
                    return cached
                retry_after = cls._capability_retry_after.get(cache_key)
                if retry_after is not None and time.monotonic() < retry_after:
                    return cached
                cls._capability_cache.pop(cache_key, None)
                cls._capability_retry_after.pop(cache_key, None)

        mtp_token: Optional[str] = None
        supports_dspark = False
        supports_dflash = False
        supports_eagle3 = False
        ngram_mod_flavor: Optional[str] = None
        spec_draft_n_max_flag: Optional[str] = None
        spec_draft_n_max_default: Optional[int] = None
        supports_kv_unified = False
        supports_trace_verbosity = False
        # See the fallback dict: these fail open.
        supports_no_context_shift = True
        supports_jinja = True
        supports_api_key_file = True
        supports_flash_attn = True
        flash_attn_takes_value = True
        supports_fit_ctx = False
        supports_fit_target = False
        supports_cache_ram = False
        supports_ctx_checkpoints = False
        ctx_checkpoints_flag = None
        ctx_checkpoints_default = None
        supports_no_cache_prompt = False
        supports_metrics = False
        supports_slot_save = False
        supports_reasoning_budget = False
        supports_reasoning_budget_message = False
        supports_reasoning_flag = False
        supports_no_mmproj_offload = False
        supports_video_fps = False
        supports_load_mode = False
        supports_moe_cache = False
        supports_moe_cache_auto = False
        supports_lazy_mode = False
        lazy_mode_flag = None
        spec_draft_ngl_flag = None
        spec_draft_cache_k_flag = None
        spec_draft_cache_v_flag = None
        saw_spec_type = False
        probe_ok = False
        help_text = ""
        # Every flag this build documents, with its help text. Pre-initialised for
        # the same reason as the booleans above: a failed probe must fall back to
        # "nothing known", not raise.
        blocks: dict[str, str] = {}
        # Same reason, and the reason it lives out here rather than beside the parse
        # loop: a probe that times out leaves the loop unrun, and the catalogue below
        # is still built. Reading an unset local there turned a slow binary into an
        # UnboundLocalError instead of "nothing known about this build".
        takes_value: dict[str, bool] = {}

        def _is_real(flag: str) -> bool:
            """True if the flag exists AND is not a removal stub."""
            desc = blocks.get(flag)
            if desc is None:
                return False
            return "argument has been removed" not in desc

        try:
            probe_env = cls._llama_server_env_for_binary(bin_path)
            # Capability detection describes the binary, independently of the
            # user's launch defaults. In particular, a device selected through
            # LLAMA_ARG_* becomes invalid when the macOS probe disables Metal.
            for name in tuple(probe_env):
                if name.startswith("LLAMA_ARG_"):
                    probe_env.pop(name, None)
            if sys.platform == "darwin":
                # Building --help asks llama_supports_rpc(), which otherwise
                # initializes Metal and compiles its embedded shaders. A cold
                # compile exceeds this probe's timeout; no device is needed to
                # enumerate the command-line flags.
                probe_env["GGML_METAL_DEVICES"] = "0"
            code_integrity_blocked: Optional[str] = None
            result = subprocess.run(
                [bin_path, "--help"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
                env = probe_env,
                # Else the probe flashes a console window on every status poll.
                **_windows_hidden_subprocess_kwargs(),
            )
            probe_ok = result.returncode == 0
            help_text = (result.stdout or "") + "\n" + (result.stderr or "")
            if not probe_ok:
                # The LOADER kills the created process: the NTSTATUS arrives as
                # a return code, or in the output when a DLL was refused.
                code_integrity_blocked = code_integrity_block_reason(
                    result.returncode
                ) or code_integrity_block_reason(help_text)
                if code_integrity_blocked is not None:
                    logger.warning(
                        "llama-server is blocked by Windows code integrity policy: "
                        f"{code_integrity_blocked}. Binary: {bin_path}"
                    )
            # Split into per-flag blocks (each --flag line + its indented
            # continuation), so the "argument has been removed" description
            # sits with its flag.
            current_flags: list[str] = []
            current_desc: list[str] = []
            # Flag -> whether its declaration shows a value placeholder.
            current_takes_value = False
            for line in help_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("-") and not line.startswith(" "):
                    # New flag line; flush previous.
                    if current_flags:
                        desc = " ".join(current_desc)
                        for f in current_flags:
                            blocks[f] = desc
                            takes_value[f] = current_takes_value
                    current_flags = []
                    current_desc = [stripped]
                    # Whether the flag takes a value at all, which is what tells
                    # "--verbose foo" (a typo llama-server refuses) from "--numa
                    # distribute". A placeholder follows its flag by a SINGLE space
                    # ("--threads N"); the description sits in an aligned column two
                    # or more spaces away. Only the yes/no is taken from this: how
                    # many values, and of what shape, is not worth guessing at from
                    # help text that wraps.
                    current_takes_value = False
                    _previous_end = None
                    for _match in re.finditer(r"\S+", stripped):
                        _token = _match.group()
                        _flag_shaped = _token.startswith("-") and re.match(
                            r"-{1,2}[A-Za-z][A-Za-z0-9_-]*,?$", _token
                        )
                        if _flag_shaped:
                            _previous_end = _match.end()
                            continue
                        # The first token that is not an alias: a placeholder when it
                        # is one space away, the description when it is further.
                        current_takes_value = (
                            _previous_end is not None and _match.start() - _previous_end == 1
                        )
                        break
                    # Extract long-form flag tokens from the DECLARATION
                    # prefix only (comma-separated aliases). Stop at the
                    # first non-flag token so flag references inside
                    # descriptions are ignored.
                    for tok in re.split(r"[,\s]+", stripped):
                        if tok.startswith("--") and re.match(r"--[A-Za-z][A-Za-z0-9_-]*$", tok):
                            current_flags.append(tok)
                        elif tok.startswith("-") and re.match(r"-[A-Za-z][A-Za-z0-9_-]*$", tok):
                            # A short alias (-fa, -t, -ngl). Kept, not just skipped:
                            # the catalogue is what the editor checks a typed flag
                            # against, and -t is as valid as --threads, so dropping
                            # it warned that a correct flag was unknown.
                            current_flags.append(tok)
                        elif tok.startswith("-") and len(tok) > 1:
                            # A value placeholder like -1; keep scanning aliases.
                            continue
                        else:
                            # First non-flag token marks end of decl.
                            break
                else:
                    current_desc.append(stripped)
            if current_flags:
                desc = " ".join(current_desc)
                for f in current_flags:
                    blocks[f] = desc
                    takes_value[f] = current_takes_value

            # MTP token from the full --spec-type help block (decl + indented
            # continuation). First-line-only probing missed builds putting the
            # enum on the next line (#7302). Prefer draft-mtp (PR #22673) over mtp.
            spec_help = blocks.get("--spec-type") or ""
            if not spec_help:
                # Fallback: join --spec-type lines, avoiding incidental "mtp" in --help.
                spec_help = "\n".join(
                    line for line in help_text.splitlines() if "--spec-type" in line
                )
            mtp_token = cls._mtp_token_from_spec_help(spec_help)
            supports_dspark = bool(
                re.search(
                    r"(?<![a-z0-9_-])draft-dspark(?![a-z0-9_-])",
                    spec_help.lower(),
                )
            )
            supports_eagle3 = bool(
                re.search(r"(?<![a-z0-9_-])draft-eagle3(?![a-z0-9_-])", spec_help.lower())
            )
            # Same word-boundary match as DSpark: draft-dflash only exists on
            # builds that carry the arch, and emitting a --spec-type the binary
            # does not know aborts the launch instead of falling back.
            supports_dflash = bool(
                re.search(
                    r"(?<![a-z0-9_-])draft-dflash(?![a-z0-9_-])",
                    spec_help.lower(),
                )
            )
            # Only a resolved --spec-type block confirms missing MTP; empty/crash
            # leaves saw_spec_type False so supports_mtp fails open.
            saw_spec_type = bool(spec_help.strip()) and "--spec-type" in spec_help

            # ngram-mod flag flavor. Post-rename builds advertise both new
            # args (real) and legacy ones (stubs); pre-rename builds only
            # have legacy ones as real.
            new_ngram_real = (
                _is_real("--spec-ngram-mod-n-match")
                and _is_real("--spec-ngram-mod-n-min")
                and _is_real("--spec-ngram-mod-n-max")
            )
            legacy_ngram_real = (
                _is_real("--spec-ngram-size-n")
                and _is_real("--draft-max")
                and _is_real("--draft-min")
            )
            if new_ngram_real:
                ngram_mod_flavor = "new"
            elif legacy_ngram_real:
                ngram_mod_flavor = "legacy"

            # n_max flag: prefer post-rename, fall back to legacy.
            if _is_real("--spec-draft-n-max"):
                spec_draft_n_max_flag = "--spec-draft-n-max"
            elif _is_real("--draft-max"):
                spec_draft_n_max_flag = "--draft-max"
            # And the build's OWN default for it, off the same help line. Needed
            # where the extras own --spec-type: Unsloth emits no depth then, so the
            # child runs on this number and the Hybrid Mamba rollback reserve
            # scales by it. None when the line carries no default.
            if spec_draft_n_max_flag:
                _n_max_default = re.search(
                    r"default:\s*(\d+)", blocks.get(spec_draft_n_max_flag) or ""
                )
                if _n_max_default:
                    spec_draft_n_max_default = int(_n_max_default.group(1))

            supports_kv_unified = _is_real("--kv-unified")
            # Fails closed: an older build's level 4 may be per-token debug on the decode path.
            supports_trace_verbosity = bool(
                probe_ok
                and _is_real("--verbosity")
                and _TRACE_VERBOSITY_HELP_RE.search(blocks.get("--verbosity") or "")
            )
            # Only once the help actually parsed AND `--help` succeeded. Empty
            # `blocks` means the probe told us nothing, and a nonzero exit means
            # the listing may be a fragment; neither may read as "absent" for a
            # flag we would otherwise always send. A real llama-server prints
            # usage and exits 0, so this costs no supported build anything.
            if probe_ok and blocks:
                supports_no_context_shift = _is_real("--no-context-shift")
                supports_jinja = _is_real("--jinja")
                supports_api_key_file = _is_real("--api-key-file")
                # Two answers, not one: whether the flag exists, and whether its
                # declaration takes a value. A build predating flash attention
                # has neither, and emitting the flag there is an immediate exit.
                supports_flash_attn = _is_real("--flash-attn")
                flash_attn_takes_value = cls._flash_attn_takes_value(help_text)
            supports_fit_ctx = _is_real("--fit-ctx")
            supports_fit_target = _is_real("--fit-target")
            supports_cache_ram = _is_real("--cache-ram")
            # --swa-checkpoints is the older spelling, kept as an alias on newer
            # builds. Probing the modern name alone drops the Checkpoints control on
            # a build carrying only the old one, so record WHICH name this build has,
            # like the draft-cache pair below.
            for _alias in ("--ctx-checkpoints", "--swa-checkpoints"):
                if _is_real(_alias):
                    ctx_checkpoints_flag = _alias
                    break
            supports_ctx_checkpoints = ctx_checkpoints_flag is not None
            # The default shipped at 3 (ggml-org/llama.cpp#15293) and is 32 today, so a
            # hardcoded 32 would RAISE an older build's count instead of capping it.
            if ctx_checkpoints_flag:
                ctx_checkpoints_default = cls._advertised_int_default(
                    blocks.get(ctx_checkpoints_flag)
                )
            supports_no_cache_prompt = _is_real("--no-cache-prompt")
            supports_metrics = _is_real("--metrics")
            supports_slot_save = _is_real("--slot-save-path")
            supports_reasoning_budget = _is_real("--reasoning-budget")
            supports_reasoning_budget_message = _is_real("--reasoning-budget-message")
            # Catalogue, not help text: --reasoning-format / -budget / -preserve contain it.
            supports_reasoning_flag = _is_real("--reasoning")
            supports_no_mmproj_offload = _is_real("--no-mmproj-offload")
            supports_video_fps = _is_real("--video-fps")
            # --load-mode supersedes --mlock / --no-mmap, which are deprecated.
            # Pre-initialised above: a failed probe must fall back, not raise.
            supports_load_mode = _is_real("--load-mode")
            # "auto" is a fork addition; upstream builds exit on it, so fail closed.
            supports_moe_cache = bool(probe_ok and _is_real("--moe-cache-mib"))
            supports_moe_cache_auto = bool(
                supports_moe_cache
                and re.search(
                    r"(?<![a-z0-9_-])auto(?![a-z0-9_-])",
                    (blocks.get("--moe-cache-mib") or "").lower(),
                )
            )
            if probe_ok:
                lazy_mode_flag = next(
                    (f for f in ("--lazy-mode", "--tensor-read-lazy") if _is_real(f)), None
                )
            supports_lazy_mode = lazy_mode_flag is not None
            # Record WHICH alias this build has: --spec-draft-ngl only landed in
            # b8955, and a build exposing only --gpu-layers-draft would refuse to
            # start on the newer name. Long forms only, since the block parser above
            # skips short aliases, so "-ngld" could never have matched.
            for _alias in ("--spec-draft-ngl", "--gpu-layers-draft", "--n-gpu-layers-draft"):
                if _is_real(_alias):
                    spec_draft_ngl_flag = _alias
                    break
            # Same alias dance for the draft KV cache pair: --spec-draft-type-* is
            # the modern spelling, --cache-type-*-draft the older, and a build with
            # only one exits on the other. K and V probed independently (upstream
            # renamed them together; a fork need not have). Long forms only.
            for _alias in ("--spec-draft-type-k", "--cache-type-k-draft"):
                if _is_real(_alias):
                    spec_draft_cache_k_flag = _alias
                    break
            for _alias in ("--spec-draft-type-v", "--cache-type-v-draft"):
                if _is_real(_alias):
                    spec_draft_cache_v_flag = _alias
                    break
        except (OSError, subprocess.SubprocessError) as exc:
            blocked = code_integrity_block_reason(exc)
            code_integrity_blocked = blocked
            if blocked is not None:
                # Warning, not debug: the only other symptom is a silent probe.
                logger.warning(
                    f"llama-server is blocked by Windows code integrity policy: {blocked}. "
                    f"Binary: {bin_path}"
                )
            else:
                logger.debug(f"llama-server --help probe failed: {exc}")
            saw_spec_type = False
            probe_ok = False
            help_text = ""

        help_nonempty = bool(help_text.strip())
        # Confirmed only when a successful --help lists a --spec-type block with
        # mtp/draft-mtp; nonempty --help without it is a definitive pre-spec
        # binary; failed/empty probes stay inconclusive (#7302).
        if saw_spec_type and probe_ok:
            supports_mtp = mtp_token is not None
            mtp_probe_inconclusive = False
        elif help_nonempty and probe_ok:
            supports_mtp = False
            mtp_probe_inconclusive = False
        else:
            supports_mtp = False
            mtp_probe_inconclusive = True

        if supports_dspark:
            try:
                from utils.llama_cpp_freshness import read_install_marker

                marker = read_install_marker(bin_path) or {}
                release_tag = marker.get("release_tag") or marker.get("tag")
                if cls._dspark_release_is_broken(release_tag):
                    supports_dspark = False
                    logger.info(
                        "Disabling DSpark for known-broken llama.cpp prebuilt %s",
                        release_tag,
                    )
            except Exception as exc:
                logger.debug("Could not inspect llama.cpp prebuilt identity: %s", exc)

        info = {
            "found": True,
            "mtp_token": mtp_token,
            "supports_mtp": supports_mtp,
            "supports_dspark": bool(supports_dspark and saw_spec_type and probe_ok),
            "supports_dflash": bool(supports_dflash and saw_spec_type and probe_ok),
            "supports_eagle3": bool(supports_eagle3 and saw_spec_type and probe_ok),
            "mtp_probe_inconclusive": mtp_probe_inconclusive,
            "ngram_mod_flavor": ngram_mod_flavor,
            "supports_ngram_mod": ngram_mod_flavor is not None,
            "spec_draft_n_max_flag": spec_draft_n_max_flag,
            "spec_draft_n_max_default": spec_draft_n_max_default,
            "supports_kv_unified": supports_kv_unified,
            "supports_no_context_shift": supports_no_context_shift,
            "supports_jinja": supports_jinja,
            "supports_api_key_file": supports_api_key_file,
            "supports_flash_attn": supports_flash_attn,
            "flash_attn_takes_value": flash_attn_takes_value,
            "supports_fit_ctx": supports_fit_ctx,
            "supports_fit_target": supports_fit_target,
            "supports_cache_ram": supports_cache_ram,
            "supports_ctx_checkpoints": supports_ctx_checkpoints,
            "ctx_checkpoints_flag": ctx_checkpoints_flag,
            "ctx_checkpoints_default": ctx_checkpoints_default,
            "supports_no_cache_prompt": supports_no_cache_prompt,
            "supports_metrics": supports_metrics,
            "supports_slot_save": supports_slot_save,
            "supports_reasoning_budget": supports_reasoning_budget,
            "supports_reasoning_budget_message": supports_reasoning_budget_message,
            "reasoning_budget_probe_inconclusive": not (probe_ok and help_nonempty),
            "supports_reasoning_flag": supports_reasoning_flag,
            "supports_no_mmproj_offload": supports_no_mmproj_offload,
            "supports_trace_verbosity": supports_trace_verbosity,
            "supports_video_fps": supports_video_fps,
            "supports_load_mode": supports_load_mode,
            "supports_moe_cache": supports_moe_cache,
            "supports_moe_cache_auto": supports_moe_cache_auto,
            "supports_lazy_mode": supports_lazy_mode,
            "lazy_mode_flag": lazy_mode_flag,
            "spec_draft_ngl_flag": spec_draft_ngl_flag,
            "spec_draft_cache_k_flag": spec_draft_cache_k_flag,
            "spec_draft_cache_v_flag": spec_draft_cache_v_flag,
            # The whole parsed catalogue, not just the booleans above. The UI
            # validates pass-through args against THIS build rather than a list
            # bundled with Unsloth, since a custom or newer llama.cpp is exactly
            # the case where a bundled list would be wrong. Removal stubs are
            # excluded: a build that still lists --draft-max only to say the
            # argument has been removed would otherwise read as supporting it,
            # and the editor would stay quiet about a flag the load then refuses.
            "flags": {flag: desc for flag, desc in blocks.items() if _is_real(flag)},
            # The flags that take no value, so the editor can tell "--verbose foo"
            # (a typo llama-server refuses) from "--numa distribute".
            "switch_flags": sorted(
                flag for flag, takes in takes_value.items() if not takes and _is_real(flag)
            ),
            # Whether --help actually succeeded. A run that exits nonzero after
            # printing part of its help still leaves a non-empty catalogue, and a
            # caller that equated "parsed something" with "read the whole thing"
            # would call every flag past the failure point unknown.
            "help_probe_ok": bool(probe_ok),
        }
        with cls._capability_cache_lock:
            published = cls._capability_cache.get(cache_key)
            if published is not None and (
                not published.get("mtp_probe_inconclusive") or mtp_probe_inconclusive
            ):
                # Concurrent callers may finish out of order. Never replace a
                # conclusive result with a timeout, or a newer retry result with
                # another result of the same confidence.
                return published
            cls._capability_cache[cache_key] = info
            if mtp_probe_inconclusive:
                # Bound both failure modes: do not pin a transient failure for
                # the process lifetime, and do not make every caller repeat a
                # 10-second timeout while a persistent failure remains (#8317).
                # Only a CONFIRMED block doubles, being permanent; escalating on
                # an inconclusive probe would strand a busy machine on stale caps.
                if code_integrity_blocked is not None:
                    delay = cls._capability_retry_backoff.get(
                        cache_key, cls._CAPABILITY_PROBE_RETRY_SECONDS
                    )
                    cls._capability_retry_backoff[cache_key] = min(
                        delay * 2.0, cls._CAPABILITY_PROBE_RETRY_MAX_SECONDS
                    )
                else:
                    delay = cls._CAPABILITY_PROBE_RETRY_SECONDS
                    cls._capability_retry_backoff.pop(cache_key, None)
                cls._capability_retry_after[cache_key] = time.monotonic() + delay
            else:
                cls._capability_retry_after.pop(cache_key, None)
                cls._capability_retry_backoff.pop(cache_key, None)
            return info

    @classmethod
    def validate_reasoning_budget_capabilities(
        cls,
        binary: Optional[str],
        *,
        extra_args: Optional[Iterable[str]],
        reasoning_budget: int,
        reasoning_budget_message: str,
    ) -> dict[str, object]:
        """Reject configured reasoning flags before replacing a live server.

        Explicit config only: the first-class fields and the passthrough extras.
        LLAMA_ARG_THINK_BUDGET* is a machine-wide llama.cpp default no Studio control
        can clear, so gating on it would fail loads with no way out of the UI.
        llama-server validates its own env; effective state still reports it.
        """
        effective_budget = resolve_reasoning_budget(extra_args, reasoning_budget)
        effective_message = resolve_reasoning_budget_message(extra_args, reasoning_budget_message)
        needs_budget, needs_message = cls.reasoning_budget_settings_requested(
            extra_args = extra_args,
            reasoning_budget = effective_budget,
            reasoning_budget_message = effective_message,
        )
        if not needs_budget and not needs_message:
            return cls.probe_server_capabilities(binary) if binary else {}
        if not binary:
            raise ValueError(
                "Reasoning budget settings require a llama-server runtime. "
                "Install or update llama.cpp, then try again."
            )

        caps = cls.probe_server_capabilities(binary)
        missing = []
        if needs_budget and not caps.get("supports_reasoning_budget"):
            missing.append("--reasoning-budget")
        if needs_message and not caps.get("supports_reasoning_budget_message"):
            missing.append("--reasoning-budget-message")
        if missing:
            verdict = (
                "could not verify support for"
                if caps.get("reasoning_budget_probe_inconclusive")
                else "does not support"
            )
            raise ValueError(
                f"llama-server at {binary} {verdict} {', '.join(missing)}. "
                "Update llama.cpp or clear the Reasoning Budget settings."
            )
        if needs_budget and effective_budget > 0:
            value_key = f"supports_reasoning_budget_value:{effective_budget}"
            if value_key not in caps:
                probe_binary = cls._exec_path_for_launch(binary)
                probe_env = cls._llama_server_env_for_binary(probe_binary)
                # Same shape as the --help probe: llama-server parses LLAMA_ARG_* before
                # argv, so a stale inherited value would read as "rejects the budget".
                for name in tuple(probe_env):
                    if name.startswith("LLAMA_ARG_"):
                        probe_env.pop(name, None)
                if sys.platform == "darwin":
                    probe_env["GGML_METAL_DEVICES"] = "0"
                try:
                    result = subprocess.run(
                        [probe_binary, "--reasoning-budget", str(effective_budget), "--help"],
                        capture_output = True,
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        timeout = 10,
                        check = False,
                        env = probe_env,
                    )
                    caps[value_key] = result.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    caps[value_key] = False
            if not caps[value_key]:
                raise ValueError(
                    f"llama-server at {binary} does not accept positive reasoning budgets. "
                    "Update llama.cpp, use 0 or -1, or clear the Reasoning Budget setting."
                )
        return caps

    @staticmethod
    def reasoning_budget_settings_requested(
        *, extra_args: Optional[Iterable[str]], reasoning_budget: int, reasoning_budget_message: str
    ) -> tuple[bool, bool]:
        budget_override = parse_reasoning_budget_override(extra_args)
        message_override = parse_reasoning_budget_message_override(extra_args)
        return (
            reasoning_budget != -1 or budget_override is not None,
            bool(reasoning_budget_message) or message_override is not None,
        )

    @staticmethod
    def _mtp_token_from_spec_help(spec_help: str) -> Optional[str]:
        """Extract ``draft-mtp`` / ``mtp`` from a ``--spec-type`` help snippet.

        Prefers ``draft-mtp`` (llama.cpp PR #22673) over the later bare ``mtp``
        rename. Returns ``None`` when neither token appears as an enum value.
        """
        text = spec_help or ""
        if "draft-mtp" in text:
            return "draft-mtp"
        # Bare `mtp` enum token (`|mtp|`, `,mtp,`, ...), not a substring.
        if re.search(r"(?<![A-Za-z0-9_-])mtp(?![A-Za-z0-9_-])", text):
            return "mtp"
        return None
