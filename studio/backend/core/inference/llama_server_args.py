# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security boundary for user-supplied ``llama-server`` arguments.

The policy is deliberately deny-known: every currently documented argument
that can reach the filesystem, network, another process, or server control
plane is classified and blocked here.  Compute, sampling, and decoding flags
remain available, and an option unknown to this build keeps the historical
pass-through behaviour.  Callers must revalidate remembered values because a
previously unclassified flag can become blocked after a llama.cpp update.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, NamedTuple, Optional

# Valid llama-server --parallel range, shared with LoadRequest.n_parallel.
# Mirrored by callers that cannot import this: run.py and unsloth_cli/commands/
# studio.py (_PARALLEL_MIN/MAX), per-model-config.ts (N_PARALLEL_MIN/MAX);
# test_parallel_slots_per_load.py pins them together.
PARALLEL_MIN = 1
PARALLEL_MAX = 64

# --batch-size / --ubatch-size range, mirrored by N_BATCH_MIN/MAX in per-model-config.ts
BATCH_MIN = 1
BATCH_MAX = 65536

# Keep the argv boundary bounded even when a caller bypasses the Studio UI.
# These limits are mirrored by the raw editor, but this validator remains the
# authoritative load-time guard for API and remembered-setting callers.
EXTRA_ARGS_MAX_TOKENS = 256
EXTRA_ARGS_MAX_UTF8_BYTES = 32 * 1024

class LlamaServerFlagPolicy(NamedTuple):
    """Canonical alias and arity record for a blocked capability."""

    canonical: str
    aliases: tuple[str, ...] = ()
    value_arity: int = 0
    category: str = "Server administration"

    @property
    def spellings(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


def _blocked(
    canonical: str,
    *aliases: str,
    value_arity: int = 0,
    category: str,
) -> LlamaServerFlagPolicy:
    return LlamaServerFlagPolicy(canonical, aliases, value_arity, category)


def _safe(
    canonical: str,
    *aliases: str,
    value_arity: int = 1,
    category: str,
) -> LlamaServerFlagPolicy:
    """Describe a documented pass-through flag whose syntax is unambiguous."""

    return LlamaServerFlagPolicy(canonical, aliases, value_arity, category)


def _safe_compute(
    canonical: str, *aliases: str, value_arity: int = 1
) -> LlamaServerFlagPolicy:
    return _safe(
        canonical,
        *aliases,
        value_arity = value_arity,
        category = "Compute/placement",
    )


def _safe_sampling(
    canonical: str, *aliases: str, value_arity: int = 1
) -> LlamaServerFlagPolicy:
    return _safe(
        canonical,
        *aliases,
        value_arity = value_arity,
        category = "Sampling/decoding",
    )


def _safe_speculative(
    canonical: str, *aliases: str, value_arity: int = 1
) -> LlamaServerFlagPolicy:
    return _safe(
        canonical,
        *aliases,
        value_arity = value_arity,
        category = "Speculative decoding",
    )


def _safe_runtime(
    canonical: str, *aliases: str, value_arity: int = 1
) -> LlamaServerFlagPolicy:
    return _safe(
        canonical,
        *aliases,
        value_arity = value_arity,
        category = "Runtime/decoding",
    )


# Checked against the installed llama-server help (b10360 at implementation
# time).  Historical aliases remain because Studio can use system/prebuilt
# binaries from adjacent llama.cpp releases.
BLOCKED_FLAG_POLICIES: tuple[LlamaServerFlagPolicy, ...] = (
    # Non-serving terminal actions.
    _blocked("--help", "-h", "--usage", category = "Terminal action"),
    _blocked("--version", category = "Terminal action"),
    _blocked("--list-devices", category = "Terminal action"),
    _blocked("--cache-list", "-cl", category = "Terminal action"),
    _blocked("--completion-bash", category = "Terminal action"),
    # Local model/material reads and writes.
    _blocked("--model", "-m", value_arity = 1, category = "Filesystem read"),
    _blocked("--mmproj", "-mm", value_arity = 1, category = "Filesystem read"),
    _blocked("--lora", value_arity = 1, category = "Filesystem read"),
    _blocked("--lora-scaled", value_arity = 1, category = "Filesystem read"),
    _blocked("--lora-init-without-apply", category = "Server administration"),
    _blocked("--control-vector", value_arity = 1, category = "Filesystem read"),
    _blocked("--control-vector-scaled", value_arity = 1, category = "Filesystem read"),
    _blocked("--control-vector-layer-range", value_arity = 2, category = "Filesystem read"),
    _blocked("--grammar-file", value_arity = 1, category = "Filesystem read"),
    _blocked("--json-schema-file", "-jf", value_arity = 1, category = "Filesystem read"),
    _blocked("--chat-template-file", value_arity = 1, category = "Filesystem read"),
    _blocked("--lookup-cache-static", "-lcs", value_arity = 1, category = "Filesystem read"),
    _blocked("--lookup-cache-dynamic", "-lcd", value_arity = 1, category = "Filesystem read/write"),
    _blocked("--log-file", value_arity = 1, category = "Filesystem write"),
    _blocked("--log-prompts-dir", value_arity = 1, category = "Filesystem write"),
    _blocked("--slot-save-path", value_arity = 1, category = "Filesystem write"),
    _blocked("--media-path", value_arity = 1, category = "Filesystem read"),
    _blocked("--models-dir", value_arity = 1, category = "Filesystem read"),
    _blocked("--models-preset", value_arity = 1, category = "Filesystem read"),
    # Download/RPC capability, including the built-in downloadable presets.
    _blocked("--rpc", value_arity = 1, category = "Network/RPC"),
    _blocked("--model-url", "-mu", value_arity = 1, category = "Network/downloader"),
    _blocked("--docker-repo", "-dr", value_arity = 1, category = "Network/downloader"),
    _blocked("--hf-repo", "-hf", "-hfr", value_arity = 1, category = "Network/downloader"),
    _blocked("--hf-file", "-hff", value_arity = 1, category = "Network/downloader"),
    _blocked("--hf-repo-v", "-hfv", "-hfrv", value_arity = 1, category = "Network/downloader"),
    _blocked("--hf-file-v", "-hffv", value_arity = 1, category = "Network/downloader"),
    _blocked("--mmproj-url", "-mmu", value_arity = 1, category = "Network/downloader"),
    _blocked("--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft", value_arity = 1, category = "Network/downloader"),
    _blocked("--embd-gemma-default", category = "Network/downloader"),
    _blocked("--fim-qwen-1.5b-default", category = "Network/downloader"),
    _blocked("--fim-qwen-3b-default", category = "Network/downloader"),
    _blocked("--fim-qwen-7b-default", category = "Network/downloader"),
    _blocked("--fim-qwen-7b-spec", category = "Network/downloader"),
    _blocked("--fim-qwen-14b-spec", category = "Network/downloader"),
    _blocked("--fim-qwen-30b-default", category = "Network/downloader"),
    _blocked("--gpt-oss-20b-default", category = "Network/downloader"),
    _blocked("--gpt-oss-120b-default", category = "Network/downloader"),
    _blocked("--vision-gemma-4b-default", category = "Network/downloader"),
    _blocked("--vision-gemma-12b-default", category = "Network/downloader"),
    # Credentials and transport security.
    _blocked("--api-key", value_arity = 1, category = "Authentication"),
    _blocked("--api-key-file", value_arity = 1, category = "Authentication"),
    _blocked("--hf-token", "-hft", value_arity = 1, category = "Authentication"),
    _blocked("--ssl-key-file", value_arity = 1, category = "TLS"),
    _blocked("--ssl-cert-file", value_arity = 1, category = "TLS"),
    # Listening, routing, UI and media exposure.
    _blocked("--host", value_arity = 1, category = "Routing/listening"),
    _blocked("--port", value_arity = 1, category = "Routing/listening"),
    _blocked("--reuse-port", category = "Routing/listening"),
    _blocked("--path", value_arity = 1, category = "Routing/listening"),
    _blocked("--api-prefix", value_arity = 1, category = "Routing/listening"),
    _blocked("--cors-origins", value_arity = 1, category = "Routing/listening"),
    _blocked("--cors-methods", value_arity = 1, category = "Routing/listening"),
    _blocked("--cors-headers", value_arity = 1, category = "Routing/listening"),
    _blocked("--cors-credentials", "--no-cors-credentials", category = "Routing/listening"),
    _blocked("--alias", "-a", value_arity = 1, category = "Routing/listening"),
    _blocked("--tags", value_arity = 1, category = "Routing/listening"),
    _blocked("--ui", "--webui", "--no-ui", "--no-webui", category = "UI/media"),
    _blocked("--ui-config", "--webui-config", value_arity = 1, category = "UI/media"),
    _blocked("--ui-config-file", "--webui-config-file", value_arity = 1, category = "UI/media"),
    _blocked("--ui-mcp-proxy", "--webui-mcp-proxy", "--no-ui-mcp-proxy", "--no-webui-mcp-proxy", category = "UI/media"),
    _blocked("--mmproj-auto", "--no-mmproj", "--no-mmproj-auto", category = "UI/media"),
    # Tool/process execution surfaces.
    _blocked("--tools", value_arity = 1, category = "Tools/agent/process"),
    _blocked("--tools-runtime", value_arity = 1, category = "Tools/agent/process"),
    _blocked("--mcp-servers-config", value_arity = 1, category = "Tools/agent/process"),
    _blocked("--mcp-servers-json", value_arity = 1, category = "Tools/agent/process"),
    _blocked("--agent", "-ag", "-no-ag", "--no-agent", category = "Tools/agent/process"),
    # Server lifecycle, mode and administrative endpoints.
    _blocked("--parallel", "-np", "--n-parallel", value_arity = 1, category = "Server administration"),
    _blocked("--embedding", "--embeddings", category = "Server administration"),
    _blocked("--rerank", "--reranking", category = "Server administration"),
    _blocked("--pooling", value_arity = 1, category = "Server administration"),
    _blocked("--timeout", "-to", value_arity = 1, category = "Server administration"),
    _blocked("--sse-ping-interval", value_arity = 1, category = "Server administration"),
    _blocked("--threads-http", value_arity = 1, category = "Server administration"),
    _blocked("--metrics", category = "Server administration"),
    _blocked("--props", category = "Server administration"),
    _blocked("--slots", "--no-slots", category = "Server administration"),
    _blocked("--models-max", value_arity = 1, category = "Server administration"),
    _blocked("--models-autoload", "--no-models-autoload", category = "Server administration"),
    _blocked("--sleep-idle-seconds", value_arity = 1, category = "Server administration"),
    _blocked("--log-disable", category = "Server administration"),
)

# Documented compute, sampling, decoding and performance flags that require a
# following value in the installed llama-server.  They remain pass-through;
# this inventory only lets Studio reject an incomplete argv before teardown.
# Optional-value switches such as ``--flash-attn [on|off|auto]`` and ordinary
# booleans are deliberately absent. Unknown future flags remain pass-through.
KNOWN_SAFE_FLAG_POLICIES: tuple[LlamaServerFlagPolicy, ...] = (
    _safe_compute("--threads", "-t"),
    _safe_compute("--threads-batch", "-tb"),
    _safe_compute("--cpu-mask", "-C"),
    _safe_compute("--cpu-range", "-Cr"),
    _safe_compute("--cpu-strict"),
    _safe_compute("--prio"),
    _safe_compute("--poll"),
    _safe_compute("--cpu-mask-batch", "-Cb"),
    _safe_compute("--cpu-range-batch", "-Crb"),
    _safe_compute("--cpu-strict-batch"),
    _safe_compute("--prio-batch"),
    _safe_compute("--poll-batch"),
    _safe_compute("--ctx-size", "-c"),
    _safe_compute("--predict", "-n", "--n-predict"),
    _safe_compute("--batch-size", "-b"),
    _safe_compute("--ubatch-size", "-ub"),
    _safe_compute("--keep"),
    _safe_compute("--rope-scaling"),
    _safe_compute("--rope-scale"),
    _safe_compute("--rope-freq-base"),
    _safe_compute("--rope-freq-scale"),
    _safe_compute("--yarn-orig-ctx"),
    _safe_compute("--yarn-ext-factor"),
    _safe_compute("--yarn-attn-factor"),
    _safe_compute("--yarn-beta-slow"),
    _safe_compute("--yarn-beta-fast"),
    _safe_compute("--cache-type-k", "-ctk"),
    _safe_compute("--cache-type-v", "-ctv"),
    _safe_compute("--defrag-thold", "-dt"),
    _safe_compute("--load-mode", "-lm"),
    _safe_compute("--numa"),
    _safe_compute("--device", "-dev"),
    _safe_compute("--override-tensor", "-ot"),
    _safe_compute("--n-cpu-moe", "-ncmoe"),
    _safe_compute("--gpu-layers", "-ngl", "--n-gpu-layers"),
    _safe_compute("--split-mode", "-sm"),
    _safe_compute("--tensor-split", "-ts"),
    _safe_compute("--main-gpu", "-mg"),
    _safe_compute("--fit-target", "-fitt"),
    _safe_compute("--fit-ctx", "-fitc"),
    _safe_compute("--override-kv"),
    _safe_runtime("--verbosity", "-lv", "--log-verbosity"),
    _safe_speculative("--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"),
    _safe_speculative("--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"),
    _safe_sampling("--samplers"),
    _safe_sampling("--seed", "-s"),
    _safe_sampling("--sampler-seq", "--sampling-seq"),
    _safe_sampling("--temp", "--temperature"),
    _safe_sampling("--top-k"),
    _safe_sampling("--top-p"),
    _safe_sampling("--min-p"),
    _safe_sampling("--top-nsigma", "--top-n-sigma"),
    _safe_sampling("--xtc-probability"),
    _safe_sampling("--xtc-threshold"),
    _safe_sampling("--typical", "--typical-p"),
    _safe_sampling("--repeat-last-n"),
    _safe_sampling("--repeat-penalty"),
    _safe_sampling("--presence-penalty"),
    _safe_sampling("--frequency-penalty"),
    _safe_sampling("--dry-multiplier"),
    _safe_sampling("--dry-base"),
    _safe_sampling("--dry-allowed-length"),
    _safe_sampling("--dry-penalty-last-n"),
    _safe_sampling("--dry-sequence-breaker"),
    _safe_sampling("--adaptive-target"),
    _safe_sampling("--adaptive-decay"),
    _safe_sampling("--dynatemp-range"),
    _safe_sampling("--dynatemp-exp"),
    _safe_sampling("--mirostat"),
    _safe_sampling("--mirostat-lr"),
    _safe_sampling("--mirostat-ent"),
    _safe_sampling("--logit-bias", "-l"),
    _safe_runtime("--grammar"),
    _safe_runtime("--json-schema", "-j"),
    # PR #8702 compatibility: an explicitly supplied local drafter remains a
    # pass-through choice. Keep it typed so missing values fail before launch,
    # and classified so its path can still be redacted from startup logs.
    _safe(
        "--spec-draft-model",
        "-md",
        "--model-draft",
        category = "Filesystem read",
    ),
    _safe_speculative("--spec-draft-threads", "-td", "--threads-draft"),
    _safe_speculative("--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"),
    _safe_speculative("--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"),
    _safe_speculative("--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"),
    _safe_speculative("--spec-draft-cpu-strict", "--cpu-strict-draft"),
    _safe_speculative("--spec-draft-prio", "--prio-draft"),
    _safe_speculative("--spec-draft-poll", "--poll-draft"),
    _safe_speculative("--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"),
    _safe_speculative("--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"),
    _safe_speculative("--spec-draft-prio-batch", "--prio-batch-draft"),
    _safe_speculative("--spec-draft-poll-batch", "--poll-batch-draft"),
    _safe_speculative("--spec-draft-override-tensor", "-otd", "--override-tensor-draft"),
    _safe_speculative("--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"),
    _safe_speculative("--spec-draft-n-max"),
    _safe_speculative("--spec-draft-n-min"),
    _safe_speculative("--spec-draft-p-split", "--draft-p-split"),
    _safe_speculative("--spec-draft-p-min", "--draft-p-min"),
    _safe_speculative("--spec-draft-device", "-devd", "--device-draft"),
    _safe_speculative("--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"),
    _safe_speculative("--spec-type"),
    _safe_speculative("--spec-ngram-mod-n-min"),
    _safe_speculative("--spec-ngram-mod-n-max"),
    _safe_speculative("--spec-ngram-mod-n-match"),
    _safe_speculative("--spec-ngram-simple-size-n"),
    _safe_speculative("--spec-ngram-simple-size-m"),
    _safe_speculative("--spec-ngram-simple-min-hits"),
    _safe_speculative("--spec-ngram-map-k-size-n"),
    _safe_speculative("--spec-ngram-map-k-size-m"),
    _safe_speculative("--spec-ngram-map-k-min-hits"),
    _safe_speculative("--spec-ngram-map-k4v-size-n"),
    _safe_speculative("--spec-ngram-map-k4v-size-m"),
    _safe_speculative("--spec-ngram-map-k4v-min-hits"),
    _safe_compute("--ctx-checkpoints", "-ctxcp", "--swa-checkpoints"),
    _safe_compute("--checkpoint-min-step", "-cms"),
    _safe_compute("--cache-ram", "-cram"),
    _safe_runtime("--reverse-prompt", "-r"),
    _safe_runtime("--image-min-tokens"),
    _safe_runtime("--image-max-tokens"),
    _safe_runtime("--mtmd-batch-max-tokens"),
    _safe_runtime("--embd-normalize"),
    _safe_runtime("--chat-template-kwargs"),
    _safe_compute("--cache-reuse"),
    _safe_runtime("--reasoning-format"),
    _safe_runtime("--reasoning-budget"),
    _safe_runtime("--reasoning-budget-message"),
    _safe_runtime("--chat-template"),
    _safe_runtime("--slot-prompt-similarity", "-sps"),
)

# Current documented switches which take no value, or accept an optional value.
# They remain outside KNOWN_SAFE_FLAG_POLICIES so required-value validation and
# the help-derived optional-value metadata stay unchanged. This inventory exists
# to classify known options in the Studio catalog; future options still display
# as Unclassified until reviewed.
KNOWN_SAFE_SWITCH_POLICIES: tuple[LlamaServerFlagPolicy, ...] = (
    _safe_compute("--swa-full", value_arity = 0),
    _safe_compute("--flash-attn", "-fa", value_arity = 0),
    _safe_compute("--perf", "--no-perf", value_arity = 0),
    _safe_runtime("--escape", "-e", "--no-escape", value_arity = 0),
    _safe_compute(
        "--kv-offload", "-kvo", "-nkvo", "--no-kv-offload", value_arity = 0
    ),
    _safe_compute("--repack", "-nr", "--no-repack", value_arity = 0),
    _safe_compute("--no-host", value_arity = 0),
    _safe_compute("--mlock", value_arity = 0),
    _safe_compute("--mmap", "--no-mmap", "-no-mmap", value_arity = 0),
    _safe_compute(
        "--direct-io",
        "-dio",
        "-ndio",
        "--no-direct-io",
        value_arity = 0,
    ),
    _safe_compute("--cpu-moe", "-cmoe", value_arity = 0),
    _safe_compute("--fit", "-fit", value_arity = 0),
    _safe_compute("--check-tensors", value_arity = 0),
    _safe_compute("--op-offload", "--no-op-offload", value_arity = 0),
    _safe_runtime("--log-colors", value_arity = 0),
    _safe_runtime("--verbose", "-v", "--log-verbose", value_arity = 0),
    _safe_runtime("--offline", value_arity = 0),
    _safe_runtime("--log-prefix", "--no-log-prefix", value_arity = 0),
    _safe_runtime("--log-timestamps", "--no-log-timestamps", value_arity = 0),
    _safe_sampling("--ignore-eos", value_arity = 0),
    _safe_sampling("--backend-sampling", "-bs", value_arity = 0),
    _safe_speculative(
        "--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft", value_arity = 0
    ),
    _safe_speculative(
        "--spec-draft-backend-sampling",
        "--no-spec-draft-backend-sampling",
        value_arity = 0,
    ),
    _safe_speculative("--spec-ngram-", value_arity = 0),
    _safe_compute(
        "--kv-unified", "-kvu", "-no-kvu", "--no-kv-unified", value_arity = 0
    ),
    _safe_compute(
        "--cache-idle-slots", "--no-cache-idle-slots", value_arity = 0
    ),
    _safe_compute("--context-shift", "--no-context-shift", value_arity = 0),
    _safe_runtime("--special", "-sp", value_arity = 0),
    _safe_compute("--warmup", "--no-warmup", value_arity = 0),
    _safe_runtime("--spm-infill", value_arity = 0),
    _safe_compute(
        "--cont-batching", "-cb", "-nocb", "--no-cont-batching", value_arity = 0
    ),
    _safe_compute("--mmproj-offload", "--no-mmproj-offload", value_arity = 0),
    _safe_compute("--cache-prompt", "--no-cache-prompt", value_arity = 0),
    _safe_runtime("--jinja", "--no-jinja", value_arity = 0),
    _safe_runtime("--reasoning", "-rea", value_arity = 0),
    _safe_runtime(
        "--reasoning-preserve", "--no-reasoning-preserve", value_arity = 0
    ),
    _safe_runtime(
        "--skip-chat-parsing", "--no-skip-chat-parsing", value_arity = 0
    ),
    _safe_runtime(
        "--prefill-assistant", "--no-prefill-assistant", value_arity = 0
    ),
    _safe_speculative("--spec-default", value_arity = 0),
)

_POLICY_BY_SPELLING: dict[str, LlamaServerFlagPolicy] = {
    spelling: policy
    for policy in BLOCKED_FLAG_POLICIES
    for spelling in policy.spellings
}
_SAFE_POLICY_BY_SPELLING: dict[str, LlamaServerFlagPolicy] = {
    spelling: policy
    for policy in KNOWN_SAFE_FLAG_POLICIES
    for spelling in policy.spellings
}
_SAFE_SWITCH_POLICY_BY_SPELLING: dict[str, LlamaServerFlagPolicy] = {
    spelling: policy
    for policy in KNOWN_SAFE_SWITCH_POLICIES
    for spelling in policy.spellings
}
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(policy.spellings) for policy in BLOCKED_FLAG_POLICIES
)
_DENYLIST: frozenset[str] = frozenset(_POLICY_BY_SPELLING)
# Documented no/optional-value aliases are part of syntax resolution even
# though they do not need an arity policy.  Keeping them here prevents a
# shorter attached alias from stealing an exact token (``-m`` must never claim
# ``-mlock``; the same invariant covers every installed multi-character alias).
_DECLARED_SAFE_SHORT_ALIASES: frozenset[str] = frozenset(
    {
        "-bs",
        "-cb",
        "-cmoe",
        "-cmoed",
        "-dio",
        "-e",
        "-fa",
        "-fit",
        "-kvo",
        "-kvu",
        "-mlock",
        "-ncmoed",
        "-ndio",
        "-nkvo",
        "-no-kvu",
        "-no-mmap",
        "-nocb",
        "-nr",
        "-rea",
        "-sp",
        "-v",
    }
)
_DECLARED_EXACT_SPELLINGS: frozenset[str] = frozenset(
    set(_POLICY_BY_SPELLING)
    | set(_SAFE_POLICY_BY_SPELLING)
    | set(_SAFE_SWITCH_POLICY_BY_SPELLING)
    | set(_DECLARED_SAFE_SHORT_ALIASES)
)
_ATTACHED_SHORT_SPELLINGS: tuple[str, ...] = tuple(
    sorted(
        {
            spelling
            for spelling, policy in {
                **_POLICY_BY_SPELLING,
                **_SAFE_POLICY_BY_SPELLING,
            }.items()
            if policy.value_arity > 0
            and spelling.startswith("-")
            and not spelling.startswith("--")
        },
        key = len,
        reverse = True,
    )
)


class LlamaServerArgsError(ValueError):
    """Safe, typed admission failure.  It never stores or renders a flag value."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        canonical_flag: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.canonical_flag = canonical_flag
        self.category = category


class ExtraArgsAssessment(NamedTuple):
    """Non-throwing result used to quarantine a complete remembered list."""

    args: tuple[str, ...]
    error: Optional[LlamaServerArgsError] = None

    @property
    def quarantined(self) -> bool:
        return self.error is not None


def managed_flag_groups() -> tuple[tuple[str, ...], ...]:
    """Stable managed aliases exposed to diagnostics/catalog consumers."""
    return tuple(tuple(sorted(group)) for group in _DENYLIST_GROUPS)


def managed_flags() -> tuple[str, ...]:
    """Every stable flag blocked by the authoritative admission policy."""
    return tuple(sorted(_DENYLIST))


def declared_exact_aliases() -> tuple[str, ...]:
    """All aliases which must win before attached-short value parsing."""

    return tuple(sorted(_DECLARED_EXACT_SPELLINGS))


def resolve_flag_alias(token: str) -> Optional[str]:
    """Flag name for ``token``, or None if it isn't a flag.

    Peels `--key=value` to `--key`, normalises long-option underscores like
    llama.cpp, treats `-1`/`-0.5` as values (shorts always start with a letter),
    and normalises attached blocked short forms such as ``-np8`` and
    ``-mC:\\model.gguf`` to their declared alias.
    """
    token = token.strip()
    if not token.startswith("-") or token in {"-", "--"}:
        return None
    if len(token) >= 2 and (token[1].isdigit() or token[1] == "."):
        return None
    name = token.split("=", 1)[0]
    if name.startswith("--"):
        name = name.replace("_", "-")
    # Exact declared aliases always win over attached-short interpretation.
    if name in _DECLARED_EXACT_SPELLINGS:
        return name
    # Longest prefix across both safe and blocked aliases prevents a shorter
    # safe alias from hiding a longer blocked one (``-t`` versus ``-to10``),
    # while still preserving safe forms such as ``-mg0`` over blocked ``-m``.
    for short in _ATTACHED_SHORT_SPELLINGS:
        if name.startswith(short) and len(name) > len(short):
            return short
    return name


def _flag_name(token: str) -> Optional[str]:
    """Compatibility name for the centralized exact-first alias resolver."""

    return resolve_flag_alias(token)


def flag_policy(flag: str) -> Optional[LlamaServerFlagPolicy]:
    """Return the canonical blocked policy for any supported token spelling."""
    normalised = _flag_name(flag)
    return _POLICY_BY_SPELLING.get(normalised) if normalised is not None else None


def safe_flag_policy(flag: str) -> Optional[LlamaServerFlagPolicy]:
    """Return syntax metadata for a documented pass-through flag."""

    normalised = _flag_name(flag)
    return _SAFE_POLICY_BY_SPELLING.get(normalised) if normalised is not None else None


def safe_flag_category(flag: str) -> Optional[str]:
    """User-facing category for a reviewed pass-through option."""

    normalised = _flag_name(flag)
    if normalised is None:
        return None
    policy = _SAFE_POLICY_BY_SPELLING.get(normalised)
    if policy is None:
        policy = _SAFE_SWITCH_POLICY_BY_SPELLING.get(normalised)
    return policy.category if policy is not None else None


def _attached_short_value(token: str, flag: Optional[str]) -> Optional[str]:
    """Return an attached short-option value, or None for separate/long forms."""

    stripped = token.strip()
    if (
        flag is not None
        and flag.startswith("-")
        and not flag.startswith("--")
        and stripped.startswith(flag)
        and len(stripped) > len(flag)
        and stripped[len(flag)] != "="
    ):
        return stripped[len(flag) :]
    return None


def _has_forbidden_characters(token: str) -> bool:
    # Tabs remain legal argv data.  Every line separator and C0/C1 control is
    # rejected so neither a request nor a remembered value can forge log lines.
    # Lone UTF-16 surrogates cannot be encoded into a Windows child command line.
    return any(
        char != "\t"
        and (
            ord(char) < 32
            or 0x7F <= ord(char) <= 0x9F
            or 0xD800 <= ord(char) <= 0xDFFF
            or char in {"\u2028", "\u2029"}
        )
        for char in token
    )


def _coerce_args(args: Optional[Iterable[str]]) -> list[str]:
    if args is None:
        return []
    if isinstance(args, (str, bytes)):
        raise LlamaServerArgsError(
            "malformed", "llama-server extra args must be a list of strings"
        )
    out: list[str] = []
    try:
        for raw in args:
            if not isinstance(raw, str):
                raise LlamaServerArgsError(
                    "malformed",
                    "llama-server extra args must be a list of strings",
                )
            out.append(raw)
    except TypeError as exc:
        raise LlamaServerArgsError(
            "malformed", "llama-server extra args must be an iterable of strings"
        ) from exc
    return out


def validate_argv_tokens(
    args: Optional[Iterable[str]], *, enforce_custom_limits: bool = False
) -> list[str]:
    """Validate token encoding/shape without applying capability policy."""

    out = _coerce_args(args)
    if not out:
        return []
    if enforce_custom_limits and len(out) > EXTRA_ARGS_MAX_TOKENS:
        raise LlamaServerArgsError(
            "too_many_tokens",
            f"llama-server extra args exceed the {EXTRA_ARGS_MAX_TOKENS}-token limit"
        )

    total_bytes = 0
    for token in out:
        if _has_forbidden_characters(token):
            raise LlamaServerArgsError(
                "forbidden_character",
                "llama-server extra args cannot contain forbidden control characters, "
                "line separators, or invalid Unicode; horizontal tabs are allowed",
            )
        if not token or token != token.strip() or token in {"-", "--"}:
            raise LlamaServerArgsError(
                "malformed",
                "llama-server extra args cannot contain empty or whitespace-padded tokens",
            )
        token_bytes = len(token.encode("utf-8", "surrogatepass"))
        total_bytes += token_bytes
        if enforce_custom_limits and total_bytes > EXTRA_ARGS_MAX_UTF8_BYTES:
            raise LlamaServerArgsError(
                "too_large",
                "llama-server extra args exceed the "
                f"{EXTRA_ARGS_MAX_UTF8_BYTES}-byte total UTF-8 limit"
            )
    return out


def validate_extra_args(args: Optional[Iterable[str]]) -> list[str]:
    """Validate user-supplied llama-server args. Returns a flat list ready to
    extend the llama-server command; raises ``ValueError`` naming the
    offending flag on the first managed token."""
    out = validate_argv_tokens(args, enforce_custom_limits = True)
    if not out:
        return []
    for token in out:
        policy = flag_policy(token)
        if policy is not None:
            matched = _flag_name(token)
            alias_note = (
                f" via alias '{matched}'"
                if matched is not None and matched != policy.canonical
                else ""
            )
            raise LlamaServerArgsError(
                "blocked_flag",
                f"llama-server flag '{policy.canonical}'{alias_note} is blocked "
                f"({policy.category}) and managed by Unsloth Studio",
                canonical_flag = policy.canonical,
                category = policy.category,
            )
    for index, token in enumerate(out):
        policy = safe_flag_policy(token)
        if policy is None or policy.value_arity <= 0:
            continue
        if "=" in token:
            if token.split("=", 1)[1] == "":
                raise LlamaServerArgsError(
                    "malformed",
                    f"llama-server flag '{policy.canonical}' requires a value",
                    canonical_flag = policy.canonical,
                )
            continue
        matched = _flag_name(token)
        if (
            matched is not None
            and matched.startswith("-")
            and not matched.startswith("--")
            and token.strip() != matched
        ):
            # llama.cpp accepts attached short values such as ``-c4096`` and
            # ``-mg0``. The suffix satisfies the option's arity.
            continue
        for offset in range(1, policy.value_arity + 1):
            value_index = index + offset
            if value_index >= len(out) or _flag_name(out[value_index]) is not None:
                raise LlamaServerArgsError(
                    "malformed",
                    f"llama-server flag '{policy.canonical}' requires a value",
                    canonical_flag = policy.canonical,
                )
    try:
        parse_ctx_override(out)
        parse_cache_override(out)
        parse_split_mode_override(out)
        parse_gpu_layers_override(out)
    except ValueError as exc:
        raise LlamaServerArgsError("malformed", str(exc)) from exc
    return out


def validate_stored_extra_args(args: object) -> list[str]:
    """Validate a present persisted field; only an actual list is authoritative."""

    if not isinstance(args, list):
        raise LlamaServerArgsError(
            "malformed",
            "saved llama-server extra args must be a list of strings",
        )
    return validate_extra_args(args)


def assess_extra_args(args: Optional[Iterable[str]]) -> ExtraArgsAssessment:
    """Classify a complete list without exposing values or dropping a subset."""
    try:
        validated = validate_extra_args(args)
    except LlamaServerArgsError as exc:
        return ExtraArgsAssessment((), exc)
    except ValueError as exc:
        # Existing numeric/enum parsers also reject malformed known safe flags.
        return ExtraArgsAssessment(
            (),
            LlamaServerArgsError("malformed", str(exc)),
        )
    return ExtraArgsAssessment(tuple(validated))


def drop_managed_flags(
    args: Optional[Iterable[str]],
) -> tuple[list[str], list[str]]:
    """Deprecated compatibility shim; invalid lists now fail atomically.

    Callers handling persistence should use :func:`assess_extra_args` and mark
    ``assessment.quarantined``.  Returning a permitted subset is intentionally
    forbidden because it changes the saved command's meaning.
    """
    return validate_extra_args(args), []


def is_managed_flag(flag: str) -> bool:
    """True if ``flag`` is Unsloth-managed. Normalises via ``_flag_name`` so
    `-np8` / `--parallel=8` classify like the canonical tokens."""
    return flag_policy(flag) is not None


def overlaps_studio_control(flag: str) -> bool:
    """True when ``flag`` can override a first-class Studio setting.

    Unlike :func:`is_managed_flag`, overlap is advisory: custom arguments are
    appended last deliberately, so expert users may still override these
    controls. Keeping this policy beside the validator lets catalog consumers
    display warnings without duplicating backend flag groups.
    """
    normalised = _flag_name(flag)
    return normalised is not None and normalised in _STUDIO_CONTROL_OVERLAP_FLAGS


# Pass-through flags that shadow first-class LoadRequest fields; stripped
# from inherited extras so they can't last-wins-override an Apply that
# re-sets the same field.
_CONTEXT_FLAGS: frozenset[str] = frozenset({"-c", "--ctx-size"})
_CACHE_TYPE_K_FLAGS: frozenset[str] = frozenset({"-ctk", "--cache-type-k"})
_CACHE_TYPE_V_FLAGS: frozenset[str] = frozenset({"-ctv", "--cache-type-v"})
_CACHE_FLAGS: frozenset[str] = _CACHE_TYPE_K_FLAGS | _CACHE_TYPE_V_FLAGS
_SPEC_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--spec-type",
        "--spec-ngram-size-n",
        "--spec-ngram-size",
        "--draft-min",
        "--draft-max",
        # MTP path (llama.cpp #22673). The drafter selectors (local --model-draft
        # and HF --spec-draft-hf aliases) are Unsloth-managed since the separate-
        # drafter support (Gemma 4): an inherited copy must not last-wins-override
        # the auto-detected drafter. Explicit extras for the current load are never
        # stripped. The per-drafter tuning knobs (--spec-draft-type-*, -ngld,
        # --spec-draft-device) are deliberately NOT stripped: the VRAM budget reads
        # them via the same parsers the child honors, so they stay consistent on
        # inherit, and stripping them would silently move a CPU-offloaded drafter
        # back onto the GPU.
        "--model-draft",
        "-md",
        "--spec-draft-model",
        "--spec-draft-hf",
        "-hfd",
        "-hfrd",
        "--hf-repo-draft",
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
        "--spec-ngram-mod-n-match",
        "--spec-ngram-mod-n-min",
        "--spec-ngram-mod-n-max",
    }
)
_TEMPLATE_FLAGS: frozenset[str] = frozenset(
    {
        "--chat-template",
        "--chat-template-file",
        "--chat-template-kwargs",
        "--jinja",
        "--no-jinja",
    }
)
# Multi-GPU split mode shadows the Tensor Parallelism toggle
# (--split-mode tensor). Pass-through stays allowed so users keep the
# row/none/layer modes the toggle doesn't expose, but it's stripped on
# inherit and reconciled into the round-tripped tensor_parallel state.
# --tensor-split is coupled to the split mode and is stripped with it: Unsloth
# owns the tensor-mode split ratios, so an inherited/stale --tensor-split must
# not last-wins-override Unsloth's computed asymmetric split.
_SPLIT_MODE_FLAGS: frozenset[str] = frozenset({"-sm", "--split-mode"})
_TENSOR_SPLIT_FLAGS: frozenset[str] = frozenset({"-ts", "--tensor-split"})
_SPLIT_SHADOWING_FLAGS: frozenset[str] = _SPLIT_MODE_FLAGS | _TENSOR_SPLIT_FLAGS
# llama.cpp placement flags. Opt-in (users may pass them under auto-select):
# stripped only when gpu_ids is set, so they cannot override the selected pool
# or choose a main GPU outside it (#7188).
_DEVICE_FLAGS: frozenset[str] = frozenset({"--device", "-dev", "--main-gpu", "-mg"})

# GPU-offload flags. Stripped only when the GPU Memory mode owns offload
# (manual emits --fit / --gpu-layers / --n-cpu-moe); in auto, a user's
# inherited -ngl is respected (the offload_overridden path), so this group is
# opt-in, not default. Layer flags are shared with llama_cpp's override
# detection; the MoE flags are strip-only (manual's --n-cpu-moe slider owns them).
_GPU_LAYER_FLAGS: frozenset[str] = frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers"})
# inherited copies of these shadow n_batch / n_ubatch, stripped only when the field is set
_BATCH_FLAGS: frozenset[str] = frozenset({"-b", "--batch-size"})
_UBATCH_FLAGS: frozenset[str] = frozenset({"-ub", "--ubatch-size"})
_FIT_FLAGS: frozenset[str] = frozenset({"-fit", "--fit"})
_LAYER_OFFLOAD_FLAGS: frozenset[str] = _GPU_LAYER_FLAGS | _FIT_FLAGS
_MOE_OFFLOAD_FLAGS: frozenset[str] = frozenset({"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"})
_OFFLOAD_SHADOWING_FLAGS: frozenset[str] = _LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS

# Host-memory placement flags. Both can create full-model RAM reservations
# (--mlock pins it, --no-mmap mallocs a copy). Studio may emit defaults for the
# same load-mode group, but explicit custom argv remains authoritative.
_MLOCK_FLAGS: frozenset[str] = frozenset({"--mlock", "-mlock"})
# Modern spelling of both, as an enum value. Takes a value, so NOT boolean.
_LOAD_MODE_FLAGS: frozenset[str] = frozenset({"--load-mode", "-lm"})
_NO_MMAP_FLAGS: frozenset[str] = frozenset({"--no-mmap", "-no-mmap"})
# Deprecated selectors for the same load-mode enum. Measured: ANY of them
# trailing the managed flag resets the WHOLE mode and drops the mlock, in both
# polarities ("--mmap" and "--no-direct-io" do it too). Affirmative dio streams
# and holds no full copy; the negative spellings are NOT plain mmap, upstream
# maps them to mode `none` like --no-mmap, so no-reserve must veto those too.
_DIO_ON_FLAGS: frozenset[str] = frozenset({"--direct-io", "-dio"})
_DIO_OFF_FLAGS: frozenset[str] = frozenset({"--no-direct-io", "-ndio"})
_DIO_FLAGS: frozenset[str] = _DIO_ON_FLAGS | _DIO_OFF_FLAGS
_LOAD_MODE_ALIAS_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | frozenset({"--mmap"}) | _DIO_FLAGS
# Every spelling that asks for a full-model host buffer.
_RAM_RESERVING_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | _DIO_OFF_FLAGS
# llama.cpp reads these before argv, so an inherited value survives stripping the
# equivalent tokens. Scrubbed whenever a toggle is on, like the spec/placement
# env groups, so the setting owns memory placement outright.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "LLAMA_ARG_MLOCK",
    "LLAMA_ARG_MMAP",
    "LLAMA_ARG_LOAD_MODE",
    "LLAMA_ARG_DIO",
    # Legacy negative aliases, honoured by PRESENCE whatever the value.
    "LLAMA_ARG_NO_MMAP",
    "LLAMA_ARG_NO_DIO",
)

# llama.cpp gives environment handlers the same authority as argv.  A partial
# list inevitably misses new options, so every LLAMA_ARG_* is inherited-denied.
# The fixed names are authentication inputs documented outside that prefix.
LLAMA_SERVER_AUTH_ENV_VARS: frozenset[str] = frozenset(
    {
        "LLAMA_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
# Compatibility/export for older tests and callers.  The wildcard prefix is
# enforced by scrub_llama_server_env rather than represented in this tuple.
DENIED_ENV_VARS: tuple[str, ...] = tuple(sorted(LLAMA_SERVER_AUTH_ENV_VARS))

_SHADOWING_FLAGS: frozenset[str] = (
    _CONTEXT_FLAGS | _CACHE_FLAGS | _SPEC_FLAGS | _TEMPLATE_FLAGS | _SPLIT_SHADOWING_FLAGS
)

# Shadowing flags that take no value -- strip the flag only, not the next token.
_BOOLEAN_SHADOWING_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--jinja",
        "--no-jinja",
        "-cmoe",
        "--cpu-moe",
        "--mlock",
        "-mlock",
        "--no-mmap",
        "-no-mmap",
        "--mmap",
        "--direct-io",
        "-dio",
        "--no-direct-io",
        "-ndio",
    }
)


def parse_ctx_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``-c`` / ``--ctx-size`` value.

    Mirrors llama.cpp's last-wins parsing for the one numeric knob Unsloth's
    load-time fit logic needs.
    """
    if not args:
        return None

    tokens = [str(a) for a in args]
    override: Optional[int] = None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in _CONTEXT_FLAGS:
            i += 1
            continue

        attached = _attached_short_value(tok, flag)
        if attached is not None:
            raw_value = attached
            i += 1
        elif "=" in tok:
            raw_value = tok.split("=", 1)[1]
            i += 1
        else:
            if i + 1 >= n or _flag_name(tokens[i + 1]) is not None:
                raise ValueError(f"llama-server flag '{flag}' requires an integer value")
            raw_value = tokens[i + 1]
            i += 2

        try:
            value = int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(f"llama-server flag '{flag}' requires an integer value") from exc
        if value < 0:
            raise ValueError(f"llama-server flag '{flag}' requires a non-negative integer value")
        override = value

    return override


def resolve_requested_ctx(args: Optional[Iterable[str]], fallback_n_ctx: int) -> int:
    """Return the context size load_model should treat as requested.

    Single source of truth for load_model's ctx-override conditional so
    tests don't reimplement and assert against their own logic.
    """
    override = parse_ctx_override(args)
    return override if override is not None else fallback_n_ctx


def _last_flag_value(args: Optional[Iterable[str]], flags: frozenset[str]) -> Optional[str]:
    """Return the last-wins string value among ``flags`` in extras, or None.

    Handles both ``--flag=value`` and ``--flag value`` forms and raises if a
    matched flag has no (or an empty) value. Shared by the single-knob
    last-wins parsers (cache type, split mode).
    """
    if not args:
        return None

    tokens = [str(a) for a in args]
    override: Optional[str] = None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in flags:
            i += 1
            continue

        attached = _attached_short_value(tok, flag)
        if attached is not None:
            raw_value = attached
            i += 1
        elif "=" in tok:
            raw_value = tok.split("=", 1)[1]
            i += 1
        else:
            if i + 1 >= n or _flag_name(tokens[i + 1]) is not None:
                raise ValueError(f"llama-server flag '{flag}' requires a value")
            raw_value = tokens[i + 1]
            i += 2

        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"llama-server flag '{flag}' requires a non-empty value")
        override = value

    return override


def parse_cache_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins cache type if extras pass cache flags.

    Mirrors parse_ctx_override but for cache type. Recognises both -ctk
    (key) and -ctv (value). When both flags appear, returns the last-wins
    value, treating key and value cache flags as the same setting because
    Unsloth's KV estimate has a single cache_type_kv knob.
    """
    return _last_flag_value(args, _CACHE_FLAGS)


def parse_gpu_layers_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied GPU layer count from extras.

    Manual GPU memory mode strips llama.cpp offload flags because the
    first-class load fields own them. Callers use this parser first to preserve
    an explicit ``-ngl`` / ``--gpu-layers`` / ``--n-gpu-layers`` value when
    translating the extras into those fields.
    """
    raw_value = _last_flag_value(args, _GPU_LAYER_FLAGS)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("llama-server GPU layers flag requires an integer value") from exc
    if value < -1:
        raise ValueError("llama-server GPU layers flag requires an integer value of at least -1")
    return value


def fit_is_enabled_in(args: Optional[Iterable[str]]) -> bool:
    """Whether the last ``--fit`` in extras turns the fitter ON.

    Only ``--fit on`` hands placement back to llama.cpp; ``--fit off`` disables
    it and so cannot move weights to the CPU. Upstream requires a value and
    rejects anything that is neither truthy nor falsey, so an absent or
    unreadable value is not an enable.
    """
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    return raw_value is not None and raw_value.strip().lower() in _ENV_TRUE_VALUES


def fit_is_effectively_on(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the fitter actually runs, over the WHOLE argv and the env twin.

    ``fit_is_enabled_in`` answers for the extras alone; this answers for the
    child. llama.cpp defaults the fitter ON and applies the env before argv, so
    only an explicit "off" turns it off, and an unreadable value keeps it on.
    """
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT")
    if raw_value is None:
        return True
    return str(raw_value).strip().lower() not in _ENV_FALSE_VALUES


def parse_cache_override_per_axis(
    args: Optional[Iterable[str]],
) -> tuple[Optional[str], Optional[str]]:
    """Last-wins --cache-type-k / --cache-type-v values kept apart, as (k, v).

    parse_cache_override collapses both axes to one last-wins value; this keeps
    them separate so an asymmetric K/V can be budgeted by its heavier axis.
    """
    return (
        _last_flag_value(args, _CACHE_TYPE_K_FLAGS),
        _last_flag_value(args, _CACHE_TYPE_V_FLAGS),
    )


def resolve_cache_type_kv(
    args: Optional[Iterable[str]], fallback_cache_type_kv: Optional[str]
) -> Optional[str]:
    """Return the cache type load_model should treat as requested.

    Single source of truth for ``load_model``'s cache override conditional.
    """
    override = parse_cache_override(args)
    return override if override is not None else fallback_cache_type_kv


def parse_split_mode_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins ``--split-mode`` / ``-sm`` value from extras.

    Mirrors parse_cache_override for the multi-GPU split mode. Returns the
    raw mode string (e.g. ``tensor`` / ``row`` / ``none`` / ``layer``), or
    None when extras don't set it.
    """
    return _last_flag_value(args, _SPLIT_MODE_FLAGS)


def resolve_tensor_parallel(args: Optional[Iterable[str]], fallback_tensor_parallel: bool) -> bool:
    """Return the tensor-parallel state load_model should treat as requested.

    A user-supplied ``--split-mode`` in extras last-wins-overrides the
    toggle, so reconcile it back into the boolean: any explicit split mode
    means tensor-parallel is on iff that mode is ``tensor``. Falls back to
    the toggle value when extras don't set it.
    """
    override = parse_split_mode_override(args)
    if override is None:
        return fallback_tensor_parallel
    return override.strip().lower() == "tensor"


def _env_split_mode_is_tensor(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the inherited LLAMA_ARG_SPLIT_MODE env selects tensor. Unsloth
    emits --split-mode only on its tensor branch, so a tensor env on the layer
    path would run the child tensor-parallel unbudgeted; this flips the budget
    to tensor. Only tensor is heavier, so other modes are ignored."""
    raw = (os.environ if env is None else env).get("LLAMA_ARG_SPLIT_MODE")
    return bool(raw) and raw.strip().lower() == "tensor"


def _effective_tensor_parallel(
    extra_args: Optional[Iterable[str]],
    tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Tensor-parallel decision including the inherited LLAMA_ARG_SPLIT_MODE env.

    resolve_tensor_parallel (extras + toggle), flipped on when extras set no split
    mode but the child inherits a tensor split env. Shared by load_model (which
    budgets and launches it) and the tensor-fallback wrapper (so an env-only
    tensor crash still retries layer split)."""
    resolved = resolve_tensor_parallel(extra_args, tensor_parallel)
    if (
        not resolved
        and parse_split_mode_override(extra_args) is None
        and _env_split_mode_is_tensor(env)
    ):
        return True
    return resolved


def _tensor_parallel_matches_loaded(
    extra_args: Optional[Iterable[str]],
    requested_tensor_parallel: bool,
    loaded_tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether a duplicate load request matches a loaded server's tensor state.

    Env-only tensor mode is a launch hint load_model may downgrade to layer split
    (capacity/buffer), scrubbing the child env. So only let an inherited tensor env
    raise a match against a server that *actually* launched tensor; on a downgraded
    (layer) server the env is ignored, and an identical request would downgrade the
    same way -- avoiding an endless reload of a healthy server."""
    requested = resolve_tensor_parallel(extra_args, requested_tensor_parallel)
    if (
        loaded_tensor_parallel
        and not requested
        and parse_split_mode_override(extra_args) is None
        and _env_split_mode_is_tensor(env)
    ):
        requested = True
    return requested == loaded_tensor_parallel


_MMPROJ_DISABLE_FLAGS: frozenset[str] = frozenset({"--no-mmproj", "--no-mmproj-auto"})
_MMPROJ_ENABLE_FLAGS: frozenset[str] = frozenset({"--mmproj-auto"})


_STUDIO_CONTROL_OVERLAP_FLAGS: frozenset[str] = frozenset().union(
    _CONTEXT_FLAGS,
    _CACHE_FLAGS,
    _SPEC_FLAGS,
    _TEMPLATE_FLAGS,
    _SPLIT_SHADOWING_FLAGS,
    _DEVICE_FLAGS,
    _OFFLOAD_SHADOWING_FLAGS,
    _MLOCK_FLAGS,
    _LOAD_MODE_FLAGS,
    _LOAD_MODE_ALIAS_FLAGS,
    _BATCH_FLAGS,
    _UBATCH_FLAGS,
)

_MODEL_MEMORY_OVERLAP_FLAGS: frozenset[str] = frozenset().union(
    _MLOCK_FLAGS,
    _NO_MMAP_FLAGS,
    _DIO_ON_FLAGS,
    _DIO_OFF_FLAGS,
    frozenset({"--mmap"}),
    _LOAD_MODE_FLAGS,
    _LOAD_MODE_ALIAS_FLAGS,
)


def extra_args_override_model_memory(args: Optional[Iterable[str]]) -> bool:
    """Whether custom argv explicitly owns the Model Memory launch mode."""
    return any(
        (_flag_name(str(raw)) or "") in _MODEL_MEMORY_OVERLAP_FLAGS
        for raw in (args or ())
    )


def extra_args_disable_mmproj(args: Optional[Iterable[str]]) -> bool:
    """True when pass-through args opt out of vision mmproj loading.

    llama-server parses --mmproj-auto / --no-mmproj / --no-mmproj-auto as one
    boolean with last-wins semantics; mirror that here.
    """
    if not args:
        return False
    disabled = False
    for raw in args:
        flag = _flag_name(str(raw))
        if flag in _MMPROJ_DISABLE_FLAGS:
            disabled = True
        elif flag in _MMPROJ_ENABLE_FLAGS:
            disabled = False
    return disabled


def strip_shadowing_flags(
    args: Iterable[str],
    *,
    strip_context: bool = True,
    strip_cache: bool = True,
    strip_spec: bool = True,
    strip_template: bool = True,
    strip_split_mode: bool = True,
    strip_tensor_split: bool = False,
    strip_offload: bool = False,
    strip_device: bool = False,
    strip_mlock: bool = False,
    strip_no_mmap: bool = False,
    strip_load_mode_aliases: bool = False,
    strip_load_mode: bool = False,
    strip_batch: bool = False,
    strip_ubatch: bool = False,
) -> list[str]:
    """Low-level removal of selected llama.cpp argument groups.

    Ordinary Run Settings do not call this to rewrite custom arguments: custom
    argv is authoritative. The helper remains for explicit recovery paths (for
    example a proven-crashing tensor attempt or virtualised-Metal CPU rescue)
    and focused normalization utilities. Each ``strip_*`` toggle controls one
    group.

    ``strip_split_mode`` removes both ``--split-mode`` and the coupled
    ``--tensor-split`` (the Tensor Parallelism toggle owns the whole split).
    ``strip_tensor_split`` removes ``--tensor-split`` *alone*, so manual mode can
    replace an inherited per-GPU ratio while leaving the user's ``--split-mode``
    row/none/layer choice intact. ``strip_no_mmap`` covers every spelling of
    mode `none`, so the negative DirectIO forms go with it. Boolean groups drop
    only the token; valued groups also drop their value.
    """
    shadowing: set[str] = set()
    if strip_context:
        shadowing |= _CONTEXT_FLAGS
    if strip_cache:
        shadowing |= _CACHE_FLAGS
    if strip_spec:
        shadowing |= _SPEC_FLAGS
    if strip_template:
        shadowing |= _TEMPLATE_FLAGS
    if strip_split_mode:
        shadowing |= _SPLIT_SHADOWING_FLAGS
    if strip_tensor_split:
        shadowing |= _TENSOR_SPLIT_FLAGS
    if strip_offload:
        shadowing |= _OFFLOAD_SHADOWING_FLAGS
    if strip_device:
        shadowing |= _DEVICE_FLAGS
    if strip_mlock:
        shadowing |= _MLOCK_FLAGS
    if strip_no_mmap:
        shadowing |= _RAM_RESERVING_FLAGS
    if strip_load_mode_aliases:
        shadowing |= _LOAD_MODE_ALIAS_FLAGS
    if strip_load_mode:
        shadowing |= _LOAD_MODE_FLAGS
    if strip_batch:
        shadowing |= _BATCH_FLAGS
    if strip_ubatch:
        shadowing |= _UBATCH_FLAGS

    tokens = [str(a) for a in (args or [])]
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in shadowing:
            out.append(tok)
            i += 1
            continue
        # Drop the flag; also consume the next token unless it's boolean,
        # already inline (`-c=4096`), or another flag.
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out


def strip_split_mode_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Remove the split-mode group (``--split-mode`` / ``-sm`` and the coupled
    ``--tensor-split`` / ``-ts``) from ``args``, keeping every other shadow flag.
    Preserves a None/empty input so the inherit-vs-explicit-empty distinction
    survives. Used where tensor mode is being forced off (downgrade / fallback)."""
    if not args:
        return args
    return strip_shadowing_flags(
        args,
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = True,
    )


def apply_model_memory_policy(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool = False,
    weights_in_host_memory: bool = True,
) -> tuple[list[str], list[str]]:
    """Resolve the Model Memory settings into llama-server flags.

    Returns ``(managed_flags, extras)``: what Unsloth emits itself, followed by
    the user's unchanged extras. Explicit custom arguments are authoritative;
    the Model Memory UI supplies defaults but never deletes a user flag.

    "Keep model in GPU memory" page-locks the weights (``--load-mode mmap+mlock``,
    or the deprecated ``--mlock``) but ONLY when ``weights_in_host_memory``.
    mlock pins a whole mapping in host RAM, so for a model fully offloaded to a
    discrete GPU it would hold a second full copy of the weights in system RAM
    without doing anything for VRAM residency; there, residency is carried by
    the idle-unload veto alone. A later custom load-mode flag may replace that
    default. Likewise, "Don't reserve system RAM" does not silently delete an
    explicit custom reservation; the custom flag wins the overlap.
    """
    try:
        from utils.model_memory_settings import get_model_memory_settings
    except Exception:
        # Settings unavailable (bare unit-test import): behave as before.
        return [], list(extra_args or [])

    # One snapshot for both decisions so a concurrent save cannot mix states.
    keep_resident, no_ram_reserve = get_model_memory_settings()
    tokens = list(extra_args or [])

    managed: list[str] = []
    if keep_resident and not no_ram_reserve and weights_in_host_memory:
        # Before the extras, like the rest of the managed block. mmap+mlock, not
        # bare mlock: it matches what --mlock meant alongside the default mmap.
        managed.extend(["--load-mode", "mmap+mlock"] if supports_load_mode else ["--mlock"])
    return managed, tokens


def model_memory_owns_placement() -> bool:
    """True when either toggle is on, so the child env must be scrubbed."""
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return False
    return get_keep_resident() or get_no_ram_reserve()


def _env_var_locks_or_reserves(name: str, value: str) -> bool:
    """Whether this inherited var, as set, locks or reserves host RAM.

    Mirrors the argv rule: the settings own the RESERVATION, not the loader, so
    a DirectIO or mmap choice made through the environment survives the same way
    ``--load-mode dio`` does. An unrecognised value is left alone.
    """
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        return normalized in _ENV_TRUE_VALUES
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        # Presence alone selects mode "none", which is a full host buffer.
        return True
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        # Falsy selects "none"; truthy selects mmap / dio, neither of which
        # holds a full copy.
        return normalized in _ENV_FALSE_VALUES
    if name == "LLAMA_ARG_LOAD_MODE":
        return normalized in _LOAD_MODE_MLOCK_VALUES or normalized in _LOAD_MODE_RESERVING_VALUES
    return False


def scrub_memory_env(env: dict) -> list[str]:
    """Drop inherited memory placement the settings override.

    Returns the names removed, for logging. A no-op with both toggles off, so an
    existing LLAMA_ARG_MLOCK deployment keeps working untouched. Only the values
    that actually lock or reserve go: an inherited ``LLAMA_ARG_DIO=1`` is a
    loader choice, not a reservation, and no-reserve has no quarrel with it.
    """
    if not model_memory_owns_placement():
        return []
    removed = [
        name
        for name in MEMORY_ENV_VARS
        if name in env and _env_var_locks_or_reserves(name, env[name])
    ]
    for name in removed:
        env.pop(name, None)
    return removed


def scrub_llama_server_env(
    env: dict[str, str],
    *,
    managed_env: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Create the final Studio-owned llama-server environment boundary.

    All inherited ``LLAMA_ARG_*`` and documented llama authentication inputs
    are removed.  A caller may then add a small explicit ``managed_env`` map;
    this makes the exception auditable instead of trusting ambient process
    state.  Unrelated entries (PATH, CUDA visibility, loader paths, etc.) are
    preserved.
    """
    removed = sorted(
        name
        for name in tuple(env)
        if name.startswith("LLAMA_ARG_") or name in LLAMA_SERVER_AUTH_ENV_VARS
    )
    for name in removed:
        env.pop(name, None)
    if managed_env:
        env.update({str(name): str(value) for name, value in managed_env.items()})
    return removed


def scrub_denied_env(env: dict[str, str]) -> list[str]:
    """Backward-compatible name for the complete environment scrub."""
    return scrub_llama_server_env(env)


# Mirrors llama_cpp's _LLAMA_ARG_TRUE/FALSE_VALUES; duplicated so this module
# stays dependency-free (llama_cpp imports from here, not the other way).
_ENV_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_ENV_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})

_LOAD_MODE_MLOCK_VALUES = frozenset({"mlock", "mmap+mlock"})
# Modes that read the weights into a full host buffer. "dio" streams via
# DirectIO and "mmap" maps, so neither reserves RAM for the whole model.
_LOAD_MODE_RESERVING_VALUES = frozenset({"none", "mlock"})


def resolve_effective_memory_state(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[bool, bool]:
    """``(mlock, reserves_ram)`` the child will actually run with.

    Mirrors llama.cpp: env supplies defaults, argv overrides last-wins. Used to
    compare a running process against the current settings, so the reload hint
    reflects the launched state rather than only what Unsloth emitted.
    """
    env = env or {}
    mlock = False
    reserves_ram = False
    # Each var runs the SAME handler as its flag, so it assigns the whole mode
    # and a later one overwrites an earlier one, in llama.cpp's registration
    # order. Measured: LLAMA_ARG_MLOCK=1 with LLAMA_ARG_MMAP=on or
    # LLAMA_ARG_DIO=0 leaves the child unlocked.
    # Only the mlock bit, like the argv --mlock below: "mlock" vs "mmap+mlock"
    # is not observable and changes no decision.
    if str(env.get("LLAMA_ARG_MLOCK", "")).strip().lower() in _ENV_TRUE_VALUES:
        mlock = True
    # Every option with a negative form also answers to LLAMA_ARG_NO_<NAME>:
    # upstream rewrites the name and, if that var EXISTS, forces the value
    # falsey whatever it says, before reading the affirmative one. Measured:
    # LLAMA_ARG_NO_MMAP=0 still disables mmap, and it beats LLAMA_ARG_MMAP=on.
    # --mlock has no negative form, so LLAMA_ARG_NO_MLOCK does nothing.
    # LLAMA_ARG_MMAP is whether to mmap, so "off" means mmap disabled ("none").
    _mmap_env = "0" if "LLAMA_ARG_NO_MMAP" in env else str(env.get("LLAMA_ARG_MMAP", ""))
    _mmap_env = _mmap_env.strip().lower()
    if _mmap_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram = False, False
    elif _mmap_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram = False, True
    # LLAMA_ARG_DIO likewise: on selects DirectIO, off selects "none".
    _dio_env = "0" if "LLAMA_ARG_NO_DIO" in env else str(env.get("LLAMA_ARG_DIO", ""))
    _dio_env = _dio_env.strip().lower()
    if _dio_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram = False, False
    elif _dio_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram = False, True
    _mode_env = str(env.get("LLAMA_ARG_LOAD_MODE", "")).strip().lower()
    if _mode_env:
        mlock = _mode_env in _LOAD_MODE_MLOCK_VALUES
        reserves_ram = _mode_env in _LOAD_MODE_RESERVING_VALUES

    tokens = [str(a) for a in (argv or [])]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None:
            i += 1
            continue
        if flag in _MLOCK_FLAGS:
            # Only the mlock bit: the enum has both "mlock" and "mmap+mlock" and
            # which one this maps to is not observable. It changes no decision,
            # since mlock alone already counts as a reservation for no-reserve.
            mlock = True
            i += 1
        elif flag in _NO_MMAP_FLAGS:
            # Deprecated selector for the whole "none" mode, so it clears the
            # mlock too: measured, "--mlock --no-mmap" leaves the child unlocked
            # while "--no-mmap --mlock" locks it.
            mlock = False
            reserves_ram = True
            i += 1
        elif flag in _DIO_ON_FLAGS:
            # Deprecated load-mode selector: resets the mode, so the mlock goes.
            # DirectIO streams the weights, so it holds no full host copy.
            mlock = False
            reserves_ram = False
            i += 1
        elif flag in _DIO_OFF_FLAGS:
            # Not "plain mmap": upstream maps these to mode `none`, like
            # --no-mmap, which reads the weights into a full host buffer.
            mlock = False
            reserves_ram = True
            i += 1
        elif flag == "--mmap":
            mlock = False
            reserves_ram = False
            i += 1
        elif flag in _LOAD_MODE_FLAGS:
            if "=" in tok:
                value, step = tok.split("=", 1)[1], 1
            elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
                value, step = tokens[i + 1], 2
            else:
                value, step = "", 1
            value = value.strip().lower()
            if value:
                mlock = value in _LOAD_MODE_MLOCK_VALUES
                reserves_ram = value in _LOAD_MODE_RESERVING_VALUES
            i += step
        else:
            i += 1
    return mlock, reserves_ram


def memory_state_satisfies_settings(
    state: Optional[tuple[bool, bool]],
    policy_active: bool = False,
    mlock_applicable: bool = True,
) -> bool:
    """True when a launched ``(mlock, reserves_ram)`` matches the settings.

    Shared by the duplicate-load comparator (so toggling a setting forces a real
    relaunch instead of returning already-loaded) and the settings route (so the
    reload hint agrees with it).

    ``state`` is None for a process this policy does not govern, such as the
    diffusion runner, which has no load-mode of its own; nothing about it can
    contradict the settings, so it always matches.

    ``policy_active`` says the launch differed from an unmanaged one, because a
    flag was emitted, a requested one suppressed, or an inherited env var
    scrubbed. With both toggles off the policy no longer applies, so any of
    those has to be undone on the next launch, while a launch it never touched
    is left alone.

    ``mlock_applicable`` is False when the launch is fully offloaded to a
    discrete GPU, where page-locking host RAM buys nothing and is deliberately
    not emitted. Residency there is the idle-unload veto, which needs no
    relaunch, so demanding mlock would ask for a reload that can never satisfy
    the check.
    """
    if state is None:
        return True
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return True
    mlock, reserves_ram = state
    if get_no_ram_reserve():
        # mlock_applicable only excuses a MISSING lock; a live reservation
        # still has to go, wherever the weights are.
        return not (mlock or reserves_ram)
    if get_keep_resident():
        return mlock or not mlock_applicable
    return not policy_active
