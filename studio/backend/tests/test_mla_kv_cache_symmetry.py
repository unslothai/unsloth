# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MLA model rejects different K and V cache types.

llama-context.cpp gained this check, and it sits ABOVE the V-quantization
check, so it decides first:

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4)
            && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("model does not support different K (%s) and V (%s)
                         cache types");
        return nullptr;
    }

is_mla() covers DeepSeek V2/V3/R1, Kimi K2 and GLM-4.7/5.x, which Unsloth
already recognises through kv_lora_rank.

The flash-attention-off retry resets a quantized V cache to f16 and
deliberately leaves K quantized, because a quantized K needs no FA and
resetting it enlarges the cache. On an MLA model that produces K=q8_0 V=f16,
which is a hard abort for a DIFFERENT reason than the one being avoided: the
retry that exists to recover from an FA crash fails on the K/V mismatch
instead of recovering.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

QUANT_KV = [
    "llama-server",
    "-m",
    "ds.gguf",
    "--flash-attn",
    "on",
    "--cache-type-k",
    "q8_0",
    "--cache-type-v",
    "q8_0",
]


def _types_of(
    cmd,
    k_flag = "--cache-type-k",
    v_flag = "--cache-type-v",
):
    return cmd[cmd.index(k_flag) + 1], cmd[cmd.index(v_flag) + 1]


class TestTheRetryKeepsKAndVEqualOnMla:
    def test_both_axes_come_down_together(self):
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV), mla = True)
        k, v = _types_of(out)
        assert k == v == "f16", (k, v)

    def test_the_inline_equals_spelling_is_handled(self):
        cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k=q8_0", "--cache-type-v=q8_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert "--cache-type-k=f16" in out and "--cache-type-v=f16" in out

    def test_the_draft_pair_is_handled_too(self):
        cmd = [*QUANT_KV, "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert _types_of(out) == ("f16", "f16")
        assert _types_of(out, "--spec-draft-type-k", "--spec-draft-type-v") == ("f16", "f16")

    def test_an_unquantized_kv_is_left_alone(self):
        """Nothing to reset, so nothing moves and no cache grows."""
        cmd = [
            "llama-server",
            "--flash-attn",
            "on",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
        ]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert _types_of(out) == ("f16", "f16")

    def test_a_quantized_k_alone_still_comes_down(self):
        """A quantized K with no V flag at all is NOT already symmetric: V falls
        back to the f16 default, so llama.cpp sees K=q8_0 against V=f16 and
        rejects it on MLA. Lowering K is what makes the retry start."""
        cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k", "q8_0"]
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        assert out[out.index("--cache-type-k") + 1] == "f16"

    def test_argv_k_comes_down_when_v_is_quantized_only_in_the_env(self):
        """The reset cannot see the environment, so it must not make lowering K
        conditional on having just lowered V on argv. Here V is quantized purely
        through LLAMA_ARG_CACHE_TYPE_V, which _drop_env_quantized_v_cache removes;
        a K left at q8_0 would then abort against the resulting f16 V."""
        cmd = ["llama-server", "--flash-attn", "on", "--cache-type-k", "q8_0"]
        env = {"LLAMA_ARG_CACHE_TYPE_V": "q8_0"}
        out = LlamaCppBackend._with_flash_attn_off(cmd, mla = True)
        LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True)
        assert out[out.index("--cache-type-k") + 1] == "f16"
        assert env == {}


class TestTheEnvDropKeepsMlaKAndVEqual:
    """llama.cpp reads LLAMA_ARG_* before parsing argv, so an inherited quantized
    K env survives an argv-only fix and reintroduces the mismatch."""

    def test_the_k_env_goes_with_the_v_env_on_mla(self):
        env = {
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
        }
        assert LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True) is True
        assert env == {}

    def test_the_draft_pair_goes_too(self):
        env = {
            "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q4_0",
            "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V": "q4_0",
        }
        assert LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True) is True
        assert env == {}

    def test_a_quantized_k_env_alone_is_dropped_on_mla(self):
        """V is f16 by then, so a lone quantized K env is a guaranteed abort."""
        env = {"LLAMA_ARG_CACHE_TYPE_K": "q8_0"}
        assert LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True) is True
        assert env == {}

    def test_non_mla_keeps_the_quantized_k_env(self):
        """The size argument is unchanged off MLA: a quantized K runs fine
        without flash attention, so its env var is preserved."""
        env = {
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
        }
        assert LlamaCppBackend._drop_env_quantized_v_cache(env, mla = False) is True
        assert env == {"LLAMA_ARG_CACHE_TYPE_K": "q8_0"}

    def test_the_default_is_the_non_mla_behaviour(self):
        """mla defaults False, so an un-updated caller keeps today's answer."""
        env = {"LLAMA_ARG_CACHE_TYPE_K": "q8_0", "LLAMA_ARG_CACHE_TYPE_V": "q8_0"}
        LlamaCppBackend._drop_env_quantized_v_cache(env)
        assert env == {"LLAMA_ARG_CACHE_TYPE_K": "q8_0"}

    def test_an_unquantized_k_env_is_left_alone_on_mla(self):
        """f16/bf16/f32 satisfy the V rule already; dropping them would silently
        change a launch the user configured deliberately."""
        env = {"LLAMA_ARG_CACHE_TYPE_K": "bf16", "LLAMA_ARG_CACHE_TYPE_V": "bf16"}
        assert LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True) is False
        assert env == {"LLAMA_ARG_CACHE_TYPE_K": "bf16", "LLAMA_ARG_CACHE_TYPE_V": "bf16"}


class TestNonMlaBehaviourIsUnchanged:
    """The size argument still holds everywhere else: resetting K needlessly
    enlarges it and can OOM a memory-constrained config."""

    def test_k_stays_quantized(self):
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV), mla = False)
        assert _types_of(out) == ("q8_0", "f16")

    def test_the_default_is_the_non_mla_behaviour(self):
        """mla defaults False, so an un-updated caller keeps today's answer."""
        out = LlamaCppBackend._with_flash_attn_off(list(QUANT_KV))
        assert _types_of(out) == ("q8_0", "f16")


class TestTheSignalStudioAlreadyHas:
    def test_every_call_site_passes_both_per_model_signals(self):
        """The target and the drafter are separate models with separate contexts,
        so every reset site gates each side on its own metadata rather than
        reusing the target's answer for both."""
        import inspect

        src = inspect.getsource(LlamaCppBackend.load_model)
        assert src.count("mla = self._target_kv_symmetry()") >= 2
        assert src.count("draft_mla = self._draft_kv_symmetry(") == src.count(
            "mla = self._target_kv_symmetry()"
        )

    def test_kv_lora_rank_is_the_mla_marker(self):
        """kv_lora_rank stands in for is_mla(): the MLA archs' converter writes it
        alongside the MLA head dims that is_mla() actually reads."""
        assert LlamaCppBackend._requires_symmetric_kv(512, "deepseek2") is True
        assert LlamaCppBackend._requires_symmetric_kv(None, "llama") is False

    def test_deepseek4_is_recognized_without_kv_lora_rank(self):
        """Upstream checks `is_mla() || arch == LLM_ARCH_DEEPSEEK4`. DeepSeek4 has
        its own KV cache, never sets the MLA head dims, and its converter writes
        q_lora_rank but no kv_lora_rank, so the rank probe alone misses it."""
        assert LlamaCppBackend._requires_symmetric_kv(None, "deepseek4") is True
        assert LlamaCppBackend._requires_symmetric_kv(None, "DeepSeek4") is True
        assert LlamaCppBackend._requires_symmetric_kv(None, " deepseek4 ") is True

    def test_a_non_mla_arch_without_a_rank_is_not_symmetric(self):
        assert LlamaCppBackend._requires_symmetric_kv(None, None) is False
        assert LlamaCppBackend._requires_symmetric_kv(None, "deepseek") is False
        assert LlamaCppBackend._requires_symmetric_kv(None, "deepseek2") is False


class TestTheDrafterIsGatedOnItsOwnModel:
    """llama.cpp applies the K == V restriction per context, and the drafter is a
    separate model, so the target's answer must not decide the draft flags."""

    DRAFT_CMD = [
        "llama-server",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--spec-draft-type-k",
        "q8_0",
        "--spec-draft-type-v",
        "q8_0",
    ]

    def _draft_k(self, out):
        return out[out.index("--spec-draft-type-k") + 1]

    def _main_k(self, out):
        return out[out.index("--cache-type-k") + 1]

    def test_an_mla_target_leaves_a_non_mla_drafters_k_quantized(self):
        """Resetting it would needlessly double the drafter's K cache, which is
        the OOM the size argument warns about."""
        out = LlamaCppBackend._with_flash_attn_off(list(self.DRAFT_CMD), mla = True, draft_mla = False)
        assert self._main_k(out) == "f16"
        assert self._draft_k(out) == "q8_0"

    def test_a_non_mla_target_still_brings_an_mla_drafters_k_down(self):
        """The mirror case, and the more serious one: leaving the draft K
        quantized against an f16 draft V aborts the draft context."""
        out = LlamaCppBackend._with_flash_attn_off(list(self.DRAFT_CMD), mla = False, draft_mla = True)
        assert self._main_k(out) == "q8_0"
        assert self._draft_k(out) == "f16"

    def test_an_unknown_drafter_falls_back_to_the_target(self):
        """None means the drafter's GGUF could not be read. An unnecessary reset
        only costs memory; a missing one aborts, so it follows the target."""
        out = LlamaCppBackend._with_flash_attn_off(list(self.DRAFT_CMD), mla = True, draft_mla = None)
        assert self._main_k(out) == "f16"
        assert self._draft_k(out) == "f16"

    def test_the_env_pair_is_split_the_same_way(self):
        env = {
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
            "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V": "q8_0",
        }
        LlamaCppBackend._drop_env_quantized_v_cache(env, mla = True, draft_mla = False)
        assert env == {"LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K": "q8_0"}


class TestTheDraftSignalComesFromTheLaunchCommand:
    """The drafter that matters is the one llama.cpp actually loads, which is the
    one named on the command being launched."""

    def _backend(self, drafters):
        """A backend whose drafter metadata lookup is stubbed, not read from disk."""
        b = LlamaCppBackend.__new__(LlamaCppBackend)
        b._kv_lora_rank = None
        b._architecture = "llama"
        b._mtp_draft_path = None

        def _draft_backend_for(path):
            meta = drafters.get(path)
            if meta is None:
                return None
            db = LlamaCppBackend.__new__(LlamaCppBackend)
            db._kv_lora_rank, db._architecture = meta
            return db

        b._draft_backend_for = _draft_backend_for
        return b

    def test_an_extra_args_drafter_is_the_one_that_counts(self):
        """--model-draft in extras is the drafter llama.cpp loads, so an MLA one
        must be seen even though the managed sidecar path is unset."""
        b = self._backend({"/d/mla.gguf": (512, "deepseek2")})
        cmd = ["llama-server", "-m", "t.gguf", "--model-draft", "/d/mla.gguf"]
        assert b._draft_kv_symmetry(cmd, {}) is True

    def test_the_managed_sidecar_is_seen_through_the_same_flag(self):
        """_build_speculative_flags emits --model-draft <sidecar> into the same
        command, so one source covers both."""
        b = self._backend({"/d/plain.gguf": (None, "llama")})
        cmd = ["llama-server", "-m", "t.gguf", "--model-draft", "/d/plain.gguf"]
        assert b._draft_kv_symmetry(cmd, {}) is False

    def test_the_stored_path_is_not_consulted(self):
        """_mtp_draft_path is assigned after the launch-site reset has already run,
        so it is stale there. Only the command decides."""
        b = self._backend({"/d/mla.gguf": (512, "deepseek2")})
        b._mtp_draft_path = "/d/mla.gguf"
        assert b._draft_kv_symmetry(["llama-server", "-m", "t.gguf"], {}) is None

    def test_the_last_draft_flag_wins(self):
        b = self._backend({"/d/mla.gguf": (512, "deepseek2"), "/d/plain.gguf": (None, "llama")})
        cmd = [
            "llama-server",
            "--model-draft",
            "/d/mla.gguf",
            "--model-draft",
            "/d/plain.gguf",
        ]
        assert b._draft_kv_symmetry(cmd, {}) is False

    def test_an_hf_repo_drafter_takes_the_conservative_path(self):
        """A repo id is not a local file, so its metadata cannot be read. Falling
        back to the target would leave an MLA drafter's K quantized against an f16
        draft V and abort; resetting a non-MLA drafter's K only costs memory."""
        b = self._backend({})
        cmd = ["llama-server", "--hf-repo-draft", "org/drafter-GGUF"]
        assert b._draft_kv_symmetry(cmd, {}) is True

    def test_an_env_supplied_drafter_is_seen(self):
        b = self._backend({"/d/mla.gguf": (512, "deepseek2")})
        env = {"LLAMA_ARG_SPEC_DRAFT_MODEL": "/d/mla.gguf"}
        assert b._draft_kv_symmetry(["llama-server"], env) is True

    def test_no_drafter_at_all_is_none(self):
        b = self._backend({})
        assert b._draft_kv_symmetry(["llama-server", "-m", "t.gguf"], {}) is None

    def test_an_unreadable_drafter_takes_the_conservative_path(self):
        """Named but unreadable is still a drafter that will launch."""
        b = self._backend({})
        cmd = ["llama-server", "--model-draft", "/d/missing.gguf"]
        assert b._draft_kv_symmetry(cmd, {}) is True

    def test_a_raising_metadata_read_takes_the_conservative_path(self):
        b = self._backend({})

        def _boom(path):
            raise OSError("unreadable")

        b._draft_backend_for = _boom
        cmd = ["llama-server", "--model-draft", "/d/broken.gguf"]
        assert b._draft_kv_symmetry(cmd, {}) is True

    def test_only_the_absence_of_a_drafter_is_none(self):
        """None means "no draft flags exist", so the value cannot matter."""
        b = self._backend({})
        assert b._draft_kv_symmetry(["llama-server", "-m", "t.gguf"], {}) is None

    def test_every_call_site_passes_a_command(self):
        """A bare _draft_kv_symmetry() would silently read no drafter at all."""
        import inspect

        src = inspect.getsource(LlamaCppBackend.load_model)
        assert "self._draft_kv_symmetry()" not in src
        assert src.count("self._draft_kv_symmetry(") == src.count("self._target_kv_symmetry()")


class TestLoadModelNeverReadsEnvBeforeItExists:
    """A launch-site fixup that reads the child environment raises
    UnboundLocalError and kills every load through that path.

    This is a static check on purpose. The block-extraction harness used by the
    flagless tests seeds ``env`` into the exec scope, so it would happily pass
    while the real function raised, which is exactly what happened once.
    """

    def test_env_is_never_loaded_before_it_is_assigned(self):
        import ast
        import inspect
        import textwrap

        fn = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))
        stores = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "env" and isinstance(n.ctx, ast.Store)
        ]
        loads = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "env" and isinstance(n.ctx, ast.Load)
        ]
        assert stores, "load_model no longer builds a child env; update this test"
        early = [line for line in loads if line < min(stores)]
        assert not early, f"env read before assignment at offsets {early}"
