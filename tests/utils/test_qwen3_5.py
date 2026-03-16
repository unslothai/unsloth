# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for unsloth/models/qwen3_5.py — fix for issue #4188.

These tests use CPU tensors and mock out GPU-only dependencies (unsloth_fused_ce_loss,
EMPTY_LOGITS) so they run without a CUDA device or real model weights.
"""

import os
import types
import unittest
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers to import the module under test with mocked unsloth internals
# ---------------------------------------------------------------------------


def _make_fake_unsloth_fused_ce_loss():
    """Return a mock that records calls and returns a scalar loss tensor."""
    mock = MagicMock(return_value=torch.tensor(1.23))
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
VOCAB_SIZE = 64


def _make_self(bsz=2, q_len=8, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
    """
    Build a minimal mock `self` (the model instance) with:
    - lm_head: a real nn.Linear (CPU) so the matmul paths work
    - loss_function: a MagicMock returning a fixed scalar tensor
    - accelerator_scaler: None
    - config.text_config.vocab_size / config.vocab_size: vocab_size
    """
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    cfg_text = MagicMock()
    cfg_text.vocab_size = vocab_size
    cfg = MagicMock()
    cfg.vocab_size = vocab_size
    cfg.text_config = cfg_text

    self = MagicMock()
    self.lm_head = lm_head
    self.config = cfg
    self.accelerator_scaler = None
    self.loss_function = MagicMock(return_value=torch.tensor(0.99))
    return self


def _make_outputs(bsz=2, q_len=8, hidden_dim=HIDDEN_DIM):
    """Return a mock outputs object whose [0] is a random hidden_states tensor."""
    hidden = torch.randn(bsz, q_len, hidden_dim)
    outputs = MagicMock()
    outputs.__getitem__ = lambda self, idx: hidden if idx == 0 else None
    outputs.past_key_values = None
    outputs.hidden_states = None
    outputs.attentions = None
    return outputs, hidden


# ---------------------------------------------------------------------------
# Tests for _qwen3_5_compute_loss_or_logits
# ---------------------------------------------------------------------------


class TestComputeLossOrLogits(unittest.TestCase):
    """Tests for the shared helper that houses all four forward paths."""

    def setUp(self):
        import unsloth.models.qwen3_5 as mod

        self.mod = mod
        self.orig_fused_ce = mod.unsloth_fused_ce_loss
        self.orig_empty_logits = mod.EMPTY_LOGITS
        # Install fresh mocks for each test
        self.mock_fused_ce = MagicMock(return_value=torch.tensor(1.23))
        self.mock_empty_logits = torch.zeros(1)
        mod.unsloth_fused_ce_loss = self.mock_fused_ce
        mod.EMPTY_LOGITS = self.mock_empty_logits

    def tearDown(self):
        self.mod.unsloth_fused_ce_loss = self.orig_fused_ce
        self.mod.EMPTY_LOGITS = self.orig_empty_logits

    # -- single-token decode path -------------------------------------------

    def test_single_token_decode_uses_mv_not_lm_head(self):
        """bsz=1, q_len=1 → fast torch.mv, no full lm_head call, no loss."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self(bsz=1, q_len=1)
        hidden = torch.randn(1, 1, HIDDEN_DIM)

        with patch.object(torch, "mv", wraps=torch.mv) as mv_spy:
            loss, logits = helper(
                self_, hidden, labels=None, logits_to_keep=0, vocab_size=VOCAB_SIZE
            )

        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (1, 1, VOCAB_SIZE))
        self.mock_fused_ce.assert_not_called()
        self_.loss_function.assert_not_called()

    # -- partial-logits path ------------------------------------------------

    def test_logits_to_keep_slices_last_n_tokens(self):
        """logits_to_keep=3 → output has exactly 3 token positions."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)

        loss, logits = helper(
            self_, hidden, labels=None, logits_to_keep=3, vocab_size=VOCAB_SIZE
        )

        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (2, 3, VOCAB_SIZE))
        self.mock_fused_ce.assert_not_called()

    # -- training / fused-CE path -------------------------------------------

    def test_training_path_calls_fused_ce_returns_early(self):
        """labels present + UNSLOTH_RETURN_LOGITS unset → fused CE, lm_head NOT called."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            loss, logits = helper(
                self_, hidden, labels=labels, logits_to_keep=0, vocab_size=VOCAB_SIZE
            )

        self.assertIsNotNone(loss)
        self.assertIs(logits, self.mock_empty_logits)
        self.mock_fused_ce.assert_called_once()
        # Verify key arguments passed to fused CE
        _, kwargs = self.mock_fused_ce.call_args
        self.assertIs(kwargs["lm_head_weight"], self_.lm_head.weight)
        self.assertEqual(kwargs["logit_softcapping"], 0)
        self_.loss_function.assert_not_called()
        # Core OOM guarantee: lm_head must NOT be called in the training path
        # — the whole point of this fix is to avoid the 7.68 GB logits tensor.
        self_.lm_head.assert_not_called()

    def test_training_path_passes_num_items_in_batch(self):
        """num_items_in_batch kwarg is forwarded to unsloth_fused_ce_loss."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            helper(
                self_,
                hidden,
                labels=labels,
                logits_to_keep=0,
                vocab_size=VOCAB_SIZE,
                num_items_in_batch=16,
            )

        _, kwargs = self.mock_fused_ce.call_args
        self.assertEqual(kwargs["n_items"], 16)

    # -- UNSLOTH_RETURN_LOGITS override ------------------------------------

    def test_return_logits_env_var_bypasses_fused_ce(self):
        """UNSLOTH_RETURN_LOGITS=1 → materialise full logits even during training."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            loss, logits = helper(
                self_, hidden, labels=labels, logits_to_keep=0, vocab_size=VOCAB_SIZE
            )

        self.assertEqual(logits.shape, (2, 8, VOCAB_SIZE))
        self.mock_fused_ce.assert_not_called()
        self_.loss_function.assert_called_once()

    # -- eval / inference path (no labels) ----------------------------------

    def test_no_labels_returns_full_logits_no_loss(self):
        """No labels → full logits computed, loss=None."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            loss, logits = helper(
                self_, hidden, labels=None, logits_to_keep=0, vocab_size=VOCAB_SIZE
            )

        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (2, 8, VOCAB_SIZE))
        self.mock_fused_ce.assert_not_called()
        self_.loss_function.assert_not_called()

    # -- n_items fallback ---------------------------------------------------

    def test_n_items_kwarg_used_when_num_items_absent(self):
        """n_items kwarg is the fallback when num_items_in_batch is absent."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            helper(
                self_,
                hidden,
                labels=labels,
                logits_to_keep=0,
                vocab_size=VOCAB_SIZE,
                n_items=8,
            )

        _, kwargs = self.mock_fused_ce.call_args
        self.assertEqual(kwargs["n_items"], 8)

    def test_num_items_in_batch_zero_does_not_fall_through_to_n_items(self):
        """num_items_in_batch=0 must NOT fall through to n_items (or-bug regression)."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            helper(
                self_,
                hidden,
                labels=labels,
                logits_to_keep=0,
                vocab_size=VOCAB_SIZE,
                num_items_in_batch=0,
                n_items=99,
            )

        _, kwargs = self.mock_fused_ce.call_args
        # 0 is falsy; the old `or` expression would have returned 99 — must be 0.
        self.assertEqual(kwargs["n_items"], 0)

    # -- batch decode (bsz > 1, q_len == 1) ---------------------------------

    def test_batch_decode_falls_through_to_eval_path(self):
        """bsz=4, q_len=1 must NOT use the single-token mv path; goes to eval path."""
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self(bsz=4, q_len=1)
        hidden = torch.randn(4, 1, HIDDEN_DIM)

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "0"}):
            with patch.object(torch, "mv", wraps=torch.mv) as mv_spy:
                loss, logits = helper(
                    self_, hidden, labels=None, logits_to_keep=0, vocab_size=VOCAB_SIZE
                )

        mv_spy.assert_not_called()  # fast path is bsz==1 AND q_len==1 only
        self.assertEqual(logits.shape, (4, 1, VOCAB_SIZE))

    # -- labels silently ignored when logits_to_keep is set -----------------

    def test_labels_ignored_when_logits_to_keep_nonzero(self):
        """
        When logits_to_keep != 0 the function returns early with partial logits
        and no loss, even if labels are provided.  This matches the llama.py
        behaviour and is intentional (speculative-decoding path).
        """
        helper = self.mod._qwen3_5_compute_loss_or_logits
        self_ = _make_self()
        hidden = torch.randn(2, 8, HIDDEN_DIM)
        labels = torch.zeros(2, 8, dtype=torch.long)

        loss, logits = helper(
            self_, hidden, labels=labels, logits_to_keep=3, vocab_size=VOCAB_SIZE
        )

        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (2, 3, VOCAB_SIZE))
        self.mock_fused_ce.assert_not_called()
        self_.loss_function.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for num_logits_to_keep normalisation and return_dict=False handling
# ---------------------------------------------------------------------------


class TestForwardFunctionBehaviour(unittest.TestCase):
    """P1/P2 regression tests for the outer forward wrappers."""

    def _make_outputs_tuple(self, bsz=2, q_len=8, hidden_dim=HIDDEN_DIM):
        """Simulate self.model(...) with return_dict=False → returns a tuple."""
        hidden = torch.randn(bsz, q_len, hidden_dim)
        past_kv = MagicMock(name="past_key_values")
        # HF tuple convention: (last_hidden_state, past_key_values)
        return (hidden, past_kv)

    def _make_outputs_dict(self, bsz=2, q_len=8, hidden_dim=HIDDEN_DIM):
        """Simulate self.model(...) with return_dict=True → returns a ModelOutput."""
        hidden = torch.randn(bsz, q_len, hidden_dim)
        outputs = MagicMock()
        outputs.__getitem__ = lambda s, idx: hidden if idx == 0 else None
        outputs.past_key_values = MagicMock(name="past_key_values")
        outputs.hidden_states = None
        outputs.attentions = None
        outputs.rope_deltas = None
        return outputs

    # -- P1: num_logits_to_keep normalisation --------------------------------

    def test_num_logits_to_keep_respected_in_conditional_generation(self):
        """num_logits_to_keep=3 must produce logits for exactly 3 token positions."""
        from unsloth.models.qwen3_5 import Qwen3_5ForConditionalGeneration_fast_forward

        self_ = _make_self(bsz=1, q_len=8)
        outputs = self._make_outputs_dict(bsz=1, q_len=8)
        self_.model = MagicMock(return_value=outputs)
        self_.config.use_return_dict = True

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            result = Qwen3_5ForConditionalGeneration_fast_forward(
                self_,
                input_ids=torch.zeros(1, 8, dtype=torch.long),
                num_logits_to_keep=3,
                logits_to_keep=0,
            )

        # Only the last 3 token positions should appear in logits
        self.assertEqual(result.logits.shape, (1, 3, VOCAB_SIZE))

    def test_num_logits_to_keep_respected_in_causal_lm(self):
        """num_logits_to_keep=2 must produce logits for exactly 2 token positions."""
        from unsloth.models.qwen3_5 import Qwen3_5ForCausalLM_fast_forward

        self_ = _make_self(bsz=1, q_len=8)
        outputs = self._make_outputs_dict(bsz=1, q_len=8)
        self_.model = MagicMock(return_value=outputs)
        self_.config.use_return_dict = True

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            result = Qwen3_5ForCausalLM_fast_forward(
                self_,
                input_ids=torch.zeros(1, 8, dtype=torch.long),
                num_logits_to_keep=2,
                logits_to_keep=0,
            )

        self.assertEqual(result.logits.shape, (1, 2, VOCAB_SIZE))

    # -- P2: return_dict=False -----------------------------------------------

    def test_return_dict_false_returns_tuple_not_dataclass(self):
        """return_dict=False must return a plain tuple, not raise AttributeError."""
        from unsloth.models.qwen3_5 import Qwen3_5ForCausalLM_fast_forward

        self_ = _make_self(bsz=2, q_len=8)
        tup = self._make_outputs_tuple(bsz=2, q_len=8)
        self_.model = MagicMock(return_value=tup)
        self_.config.use_return_dict = False

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            result = Qwen3_5ForCausalLM_fast_forward(
                self_,
                input_ids=torch.zeros(2, 8, dtype=torch.long),
                return_dict=False,
            )

        self.assertIsInstance(result, tuple, "return_dict=False must yield a tuple")

    def test_return_dict_false_does_not_access_dot_attributes(self):
        """
        When return_dict=False the model returns a tuple; accessing .past_key_values
        would raise AttributeError.  Verify no AttributeError is raised.
        """
        from unsloth.models.qwen3_5 import Qwen3_5ForConditionalGeneration_fast_forward

        self_ = _make_self(bsz=2, q_len=8)
        tup = self._make_outputs_tuple(bsz=2, q_len=8)
        self_.model = MagicMock(return_value=tup)
        self_.config.use_return_dict = False
        self_.config.text_config = MagicMock()
        self_.config.text_config.vocab_size = VOCAB_SIZE

        with patch.dict(os.environ, {"UNSLOTH_RETURN_LOGITS": "1"}):
            try:
                result = Qwen3_5ForConditionalGeneration_fast_forward(
                    self_,
                    input_ids=torch.zeros(2, 8, dtype=torch.long),
                    return_dict=False,
                )
            except AttributeError as exc:
                self.fail(f"return_dict=False raised AttributeError: {exc}")

        self.assertIsInstance(result, tuple)

    # -- UNSLOTH_RETURN_HIDDEN_STATES ----------------------------------------

    def test_return_hidden_states_causal_lm(self):
        """UNSLOTH_RETURN_HIDDEN_STATES=1 → logits field contains hidden states, no loss."""
        from unsloth.models.qwen3_5 import Qwen3_5ForCausalLM_fast_forward

        self_ = _make_self(bsz=1, q_len=8)
        outputs = self._make_outputs_dict(bsz=1, q_len=8)
        self_.model = MagicMock(return_value=outputs)
        self_.config.use_return_dict = True

        with patch.dict(os.environ, {"UNSLOTH_RETURN_HIDDEN_STATES": "1"}):
            result = Qwen3_5ForCausalLM_fast_forward(
                self_,
                input_ids=torch.zeros(1, 8, dtype=torch.long),
            )

        self.assertIsNone(result.loss)
        # logits field carries hidden states (shape [bsz, q_len, hidden_dim])
        self.assertEqual(result.logits.shape, (1, 8, HIDDEN_DIM))

    def test_return_hidden_states_sliced_by_logits_to_keep(self):
        """UNSLOTH_RETURN_HIDDEN_STATES=1 with logits_to_keep=2 → last 2 positions."""
        from unsloth.models.qwen3_5 import Qwen3_5ForConditionalGeneration_fast_forward

        self_ = _make_self(bsz=1, q_len=8)
        outputs = self._make_outputs_dict(bsz=1, q_len=8)
        self_.model = MagicMock(return_value=outputs)
        self_.config.use_return_dict = True
        self_.config.text_config = MagicMock()
        self_.config.text_config.vocab_size = VOCAB_SIZE

        with patch.dict(os.environ, {"UNSLOTH_RETURN_HIDDEN_STATES": "1"}):
            result = Qwen3_5ForConditionalGeneration_fast_forward(
                self_,
                input_ids=torch.zeros(1, 8, dtype=torch.long),
                logits_to_keep=2,
            )

        self.assertEqual(result.logits.shape, (1, 2, HIDDEN_DIM))


# ---------------------------------------------------------------------------
# Tests for FastQwen3_5Model.pre_patch()
# ---------------------------------------------------------------------------


class TestPrePatch(unittest.TestCase):
    """pre_patch() must assign the fast-forward functions to both model classes."""

    def test_pre_patch_replaces_conditional_generation_forward(self):
        from unsloth.models.qwen3_5 import (
            FastQwen3_5Model,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5ForConditionalGeneration_fast_forward,
        )

        original = Qwen3_5ForConditionalGeneration.forward
        try:
            FastQwen3_5Model.pre_patch()
            self.assertIs(
                Qwen3_5ForConditionalGeneration.forward,
                Qwen3_5ForConditionalGeneration_fast_forward,
            )
        finally:
            Qwen3_5ForConditionalGeneration.forward = original

    def test_pre_patch_replaces_causal_lm_forward(self):
        from unsloth.models.qwen3_5 import (
            FastQwen3_5Model,
            Qwen3_5ForCausalLM,
            Qwen3_5ForCausalLM_fast_forward,
        )

        original = Qwen3_5ForCausalLM.forward
        try:
            FastQwen3_5Model.pre_patch()
            self.assertIs(
                Qwen3_5ForCausalLM.forward,
                Qwen3_5ForCausalLM_fast_forward,
            )
        finally:
            Qwen3_5ForCausalLM.forward = original


# ---------------------------------------------------------------------------
# Tests for loader routing
# ---------------------------------------------------------------------------


class TestFromPretrained(unittest.TestCase):
    """from_pretrained must call FastLlamaModel, not FastQwen3Model."""

    def test_from_pretrained_calls_llama_not_qwen3(self):
        """
        FastQwen3Model.from_pretrained hardcodes model_patcher=FastQwen3Model,
        which would apply incompatible Qwen3 attention patches to Qwen3.5.
        FastQwen3_5Model.from_pretrained must bypass it and call FastLlamaModel
        directly so that only FastQwen3_5Model.pre_patch() is applied.
        """
        from unsloth.models.qwen3_5 import FastQwen3_5Model
        from unsloth.models.llama import FastLlamaModel

        with patch.object(
            FastLlamaModel, "from_pretrained", return_value=("model", "tok")
        ) as llama_mock:
            FastQwen3_5Model.from_pretrained(model_name="Qwen/Qwen3.5-0.6B-Base")

            llama_mock.assert_called_once()

            # model_patcher must be FastQwen3_5Model, not FastQwen3Model
            _, kwargs = llama_mock.call_args
            self.assertIs(
                kwargs.get("model_patcher"),
                FastQwen3_5Model,
                "model_patcher must be FastQwen3_5Model so only its pre_patch() runs",
            )


class TestLoaderRouting(unittest.TestCase):
    """model_type == 'qwen3_5' must route to FastQwen3_5Model."""

    def test_qwen3_5_routes_to_fast_model(self):
        from unsloth.models import loader as loader_mod
        from unsloth.models.qwen3_5 import FastQwen3_5Model

        self.assertTrue(
            hasattr(loader_mod, "SUPPORTS_QWEN3_5"),
            "loader.py must define SUPPORTS_QWEN3_5",
        )
        self.assertTrue(
            loader_mod.SUPPORTS_QWEN3_5,
            "SUPPORTS_QWEN3_5 should be True with transformers >= 4.53.0",
        )
        # FastQwen3_5Model must be importable from loader (conditional import succeeded)
        self.assertTrue(
            hasattr(loader_mod, "FastQwen3_5Model"),
            "FastQwen3_5Model must be imported into loader.py when SUPPORTS_QWEN3_5",
        )
        self.assertIs(loader_mod.FastQwen3_5Model, FastQwen3_5Model)

    def test_qwen3_5_in_force_float32_list(self):
        """Qwen3.5 RMSNorm overflows float16 — must stay in FORCE_FLOAT32."""
        from unsloth.models import loader as loader_mod

        self.assertIn(
            "qwen3_5",
            loader_mod.FORCE_FLOAT32,
            "qwen3_5 must remain in FORCE_FLOAT32 (RMSNorm uses (1+w) pattern)",
        )


# ---------------------------------------------------------------------------
# Tests for __init__.py exports
# ---------------------------------------------------------------------------


class TestInitExports(unittest.TestCase):
    """FastQwen3_5Model must be exported from unsloth.models."""

    def test_fast_qwen3_5_model_importable(self):
        try:
            from unsloth.models import FastQwen3_5Model  # noqa: F401
        except ImportError:
            self.fail(
                "FastQwen3_5Model should be importable from unsloth.models "
                "when transformers >= 4.53.0 is installed"
            )

    def test_init_except_clause_is_import_error(self):
        """
        The try/except around qwen3_5 in __init__.py must catch ImportError,
        not bare except (which would silently swallow unrelated exceptions).
        """
        import ast
        from pathlib import Path

        init_path = (
            Path(__file__).resolve().parents[2] / "unsloth" / "models" / "__init__.py"
        )
        tree = ast.parse(init_path.read_text())

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Find the try block that imports qwen3_5
            source = ast.unparse(node)
            if "qwen3_5" not in source:
                continue
            for handler in node.handlers:
                if handler.type is None:
                    self.fail(
                        "The try/except around qwen3_5 in __init__.py uses bare "
                        "`except:` — must use `except ImportError:` instead"
                    )
                self.assertEqual(
                    ast.unparse(handler.type),
                    "ImportError",
                    "Handler must be `except ImportError:`, got something else",
                )


if __name__ == "__main__":
    unittest.main()
