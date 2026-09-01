# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for packed-attention mask helpers with sliding-window logic."""

import math
import weakref

import pytest
import torch

from unsloth.utils import attention_dispatch
from unsloth.utils import packing as packing_utils


def _make_seq_info(lengths):
    lengths = torch.tensor(lengths, dtype = torch.int32)
    cu = torch.cat(
        [
            torch.zeros(1, dtype = torch.int32),
            torch.cumsum(lengths, dim = 0, dtype = torch.int32),
        ]
    )
    max_len = int(lengths.max().item())
    return lengths, cu, max_len


def test_sdpa_packed_attention_mask_sliding_window():
    seq_info = _make_seq_info([5, 3])
    mask = packing_utils.build_sdpa_packed_attention_mask(
        seq_info,
        dtype = torch.float32,
        device = torch.device("cpu"),
        sliding_window = 3,
    )

    assert mask.shape == (1, 1, 8, 8)

    block_first = mask[0, 0, :5, :5]
    upper = torch.triu(torch.ones_like(block_first), diagonal = 1).bool()
    assert torch.all(block_first[upper] == float("-inf"))
    assert block_first[3, 0].item() == float("-inf")
    assert block_first[4, 1].item() == float("-inf")
    assert block_first[4, 2].item() > -math.inf
    assert mask[0, 0, 0, 6].item() == float("-inf")


def test_xformers_block_mask_sliding_window(monkeypatch):
    class _FakeMask:
        def __init__(
            self,
            lengths,
            window = None,
            device = None,
        ):
            self.lengths = lengths
            self.window = window
            self.device = torch.device(device)

        @classmethod
        def from_seqlens(cls, lengths):
            return cls(tuple(lengths), device = "cuda:0")

        def make_local_attention(self, window_size):
            return _FakeMask(self.lengths, window = window_size, device = self.device)

        def to(self, device):
            return _FakeMask(self.lengths, window = self.window, device = device)

    monkeypatch.setattr(packing_utils, "_XFormersBlockMask", _FakeMask, raising = False)
    packing_utils.clear_packed_caches()

    seq_info = _make_seq_info([4, 4])
    mask = packing_utils.build_xformers_block_causal_mask(
        seq_info,
        sliding_window = 2,
    )

    assert isinstance(mask, _FakeMask)
    assert mask.window == 2
    assert mask.device == torch.device("cpu")
    packing_utils.clear_packed_caches()


def test_xformers_block_mask_cache_is_scoped_to_device(monkeypatch):
    class _FakeMask:
        def __init__(self, lengths, device):
            self.lengths = tuple(lengths)
            self.device = torch.device(device)

        @classmethod
        def from_seqlens(cls, lengths):
            return cls(lengths, "cuda:0")

        def to(self, device):
            return _FakeMask(self.lengths, device)

    monkeypatch.setattr(packing_utils, "_XFormersBlockMask", _FakeMask, raising = False)
    packing_utils.clear_packed_caches()

    lengths = (4, 4)
    cuda_0 = torch.device("cuda:0")
    cuda_1 = torch.device("cuda:1")
    first = packing_utils._get_cached_block_mask(lengths, None, cuda_0)
    second = packing_utils._get_cached_block_mask(lengths, None, cuda_1)

    assert first.device == cuda_0
    assert second.device == cuda_1
    assert second is not first
    assert packing_utils._get_cached_block_mask(lengths, None, cuda_0) is first

    packing_utils.clear_packed_caches()
    assert not packing_utils._XFORMERS_MASK_CACHE


def test_xformers_bias_move_supports_legacy_in_place_metadata():
    class _LegacySeqInfo:
        def __init__(self, device):
            self.device = torch.device(device)

        def to(self, device):
            self.device = torch.device(device)

    class _LegacyBias:
        def __init__(self):
            self.q_seqinfo = _LegacySeqInfo("cuda:0")
            self.k_seqinfo = self.q_seqinfo

    bias = _LegacyBias()
    moved = packing_utils.move_xformers_attention_bias(bias, torch.device("cuda:1"))

    assert moved is not bias
    assert moved.q_seqinfo is moved.k_seqinfo
    assert moved.q_seqinfo.device == torch.device("cuda:1")
    assert bias.q_seqinfo is bias.k_seqinfo
    assert bias.q_seqinfo.device == torch.device("cuda:0")


def test_xformers_bias_move_replaces_all_shared_metadata_aliases():
    class _FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

    class _ReturningSeqInfo:
        def __init__(self, device):
            self.seqstart = _FakeTensor(device)

        def to(self, device):
            return _ReturningSeqInfo(device)

    class _Bias:
        def __init__(self):
            self.q_seqinfo = _ReturningSeqInfo("cuda:0")
            self.k_seqinfo = self.q_seqinfo

    bias = _Bias()
    original = bias.q_seqinfo
    moved = packing_utils.move_xformers_attention_bias(bias, torch.device("cuda:1"))

    assert moved is not bias
    assert moved.q_seqinfo is moved.k_seqinfo
    assert moved.q_seqinfo is not original
    assert moved.q_seqinfo.seqstart.device == torch.device("cuda:1")
    assert bias.q_seqinfo is bias.k_seqinfo
    assert bias.q_seqinfo is original
    assert bias.q_seqinfo.seqstart.device == torch.device("cuda:0")


def test_xformers_bias_move_preserves_causal_type_when_to_demotes():
    class _FakeTensor:
        def __init__(self, device):
            self.device = torch.device(device)

    class _ReturningSeqInfo:
        def __init__(self, device):
            self.seqstart = _FakeTensor(device)

        def to(self, device):
            return _ReturningSeqInfo(device)

    class _BaseBias:
        def __init__(self, seqinfo):
            self.q_seqinfo = seqinfo
            self.k_seqinfo = seqinfo

        def to(self, device):
            return _BaseBias(self.q_seqinfo.to(device))

    class _CausalBias(_BaseBias):
        pass

    bias = _CausalBias(_ReturningSeqInfo("cuda:0"))
    first = packing_utils.move_xformers_attention_bias(bias, torch.device("cuda:1"))
    second = packing_utils.move_xformers_attention_bias(bias, torch.device("cuda:2"))

    assert first is not bias
    assert type(first) is _CausalBias
    assert first.q_seqinfo is first.k_seqinfo
    assert first.q_seqinfo.seqstart.device == torch.device("cuda:1")
    assert second is not bias
    assert type(second) is _CausalBias
    assert second.q_seqinfo is second.k_seqinfo
    assert second.q_seqinfo.seqstart.device == torch.device("cuda:2")
    assert first.q_seqinfo.seqstart.device == torch.device("cuda:1")
    assert bias.q_seqinfo is bias.k_seqinfo
    assert bias.q_seqinfo.seqstart.device == torch.device("cuda:0")


def test_xformers_bias_move_skips_matching_metadata_device():
    class _SeqInfo:
        def __init__(self):
            self.seqstart = torch.empty(0)

    class _Bias:
        def __init__(self):
            self.q_seqinfo = _SeqInfo()
            self.k_seqinfo = self.q_seqinfo

        def to(self, device):
            raise AssertionError("matching metadata should not be moved")

    bias = _Bias()
    assert packing_utils.move_xformers_attention_bias(bias, torch.device("cpu")) is bias


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or packing_utils._XFormersBlockMask is None,
    reason = "needs xFormers and two CUDA devices",
)
def test_real_xformers_packed_mask_validates_on_each_device():
    from xformers.ops.fmha.common import Inputs
    packing_utils.clear_packed_caches()
    try:
        masks = []
        for index in (0, 1):
            device = torch.device(f"cuda:{index}")
            lengths = torch.tensor([4, 4], dtype = torch.int32, device = device)
            masks.append(
                packing_utils.build_xformers_block_causal_mask(
                    (lengths, torch.empty(0, dtype = torch.int32, device = device), 4)
                )
            )

        assert masks[0].q_seqinfo.seqstart.device == torch.device("cuda:0")
        assert masks[1].q_seqinfo.seqstart.device == torch.device("cuda:1")
        assert masks[1] is not masks[0]

        config = attention_dispatch.AttentionConfig(
            backend = attention_dispatch.XFORMERS,
            n_kv_heads = 1,
            n_groups = 1,
        )
        context = attention_dispatch.AttentionContext(
            bsz = 1,
            q_len = 8,
            kv_seq_len = 8,
            n_heads = 1,
            head_dim = 64,
            requires_grad = True,
            seq_info = None,
            attention_mask = None,
            causal_mask = masks[0],
        )
        queries = []
        outputs = []
        for index in (0, 1):
            device = torch.device(f"cuda:{index}")
            query = torch.zeros(
                (1, 8, 1, 64), dtype = torch.float16, device = device, requires_grad = True
            )
            Inputs(query = query, key = query, value = query, attn_bias = masks[index]).validate_inputs()
            model_query = query.transpose(1, 2)
            outputs.append(
                attention_dispatch.run_attention(
                    config = config,
                    context = context,
                    Q = model_query,
                    K = model_query,
                    V = model_query,
                )
            )
            queries.append(query)

        assert masks[0].q_seqinfo.seqstart.device == torch.device("cuda:0")
        for index, output in enumerate(outputs):
            assert output.device == torch.device(f"cuda:{index}")
            assert bool(torch.isfinite(output).all())

        # Start backward only after the second shard has consumed the shared source mask, matching model-parallel layer
        for query, output in zip(queries, outputs):
            output.sum().backward()
            assert query.grad is not None
            assert bool(torch.isfinite(query.grad).all())
    finally:
        packing_utils.clear_packed_caches()


def test_run_attention_sdpa_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([3, 2])
    sliding_window = 2

    original_builder = attention_dispatch.build_sdpa_packed_attention_mask
    captured = {}

    def _capture_builder(
        seq_info_arg,
        *,
        dtype,
        device,
        sliding_window = None,
    ):
        captured["window"] = sliding_window
        return original_builder(
            seq_info_arg,
            dtype = dtype,
            device = device,
            sliding_window = sliding_window,
        )

    monkeypatch.setattr(
        attention_dispatch,
        "build_sdpa_packed_attention_mask",
        _capture_builder,
    )

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.SDPA,
        n_kv_heads = 1,
        n_groups = 1,
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 5,
        kv_seq_len = 5,
        n_heads = 1,
        head_dim = 1,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 5, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["window"] == sliding_window
    mask = captured["mask"]
    assert mask is not None and mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 4, 1].item() == float("-inf")


def test_run_attention_xformers_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3

    class _FakeBias:
        def __init__(self, device = "cuda:0"):
            self.device = torch.device(device)

        def to(self, device):
            return _FakeBias(device)

    captured = {}

    def _fake_builder(
        seq_info_arg,
        *,
        sliding_window = None,
        base_mask = None,
    ):
        captured["window"] = sliding_window
        captured["base"] = base_mask
        return _FakeBias()

    def _fake_attention(
        Q,
        K,
        V,
        attn_bias = None,
        **_,
    ):
        captured["bias"] = attn_bias
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "build_xformers_block_causal_mask", _fake_builder)
    monkeypatch.setattr(attention_dispatch, "xformers_attention", _fake_attention, raising = False)
    monkeypatch.setattr(attention_dispatch, "XFORMERS_BLOCK_DIAG_CLS", _FakeBias, raising = False)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.XFORMERS,
        n_kv_heads = 1,
        n_groups = 1,
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 1,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["window"] == sliding_window
    assert isinstance(captured["bias"], _FakeBias)
    assert captured["bias"].device == torch.device("cpu")


def test_run_attention_flash_varlen_receives_window_and_softcap(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3
    softcap = 0.5
    window_tuple = (sliding_window, sliding_window)

    captured = {}

    def _fake_flash_varlen(Q, K, V, cu_q, cu_k, max_q, max_k, **kwargs):
        captured["kwargs"] = kwargs
        return torch.zeros_like(Q)

    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_varlen_func",
        _fake_flash_varlen,
    )
    monkeypatch.setattr(attention_dispatch, "HAS_FLASH_ATTENTION", True)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.FLASH_VARLEN,
        n_kv_heads = 1,
        n_groups = 1,
        flash_varlen_kwargs = {
            "dropout_p": 0.0,
            "softmax_scale": 1.0,
            "causal": True,
            "softcap": softcap,
            "window_size": window_tuple,
        },
    )

    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 2,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        sliding_window = sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 2)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config = config,
        context = context,
        Q = Q,
        K = K,
        V = V,
    )

    assert captured["kwargs"]["softcap"] == softcap
    assert captured["kwargs"]["window_size"] == window_tuple


"""Unit tests for packed-attention mask helpers with sliding-window logic."""


def test_run_attention_sdpa_windows_an_unpacked_unmasked_batch(monkeypatch):
    """No packing, no padding mask: the case that had nothing to hang the window off.

    SDPA's ``is_causal`` is FULL causal -- it has no window -- so with neither the xformers
    bias nor flash's ``window_size`` in play, a model whose config declares a sliding window
    attended its entire causal history. That is reachable from a Mistral training step the
    moment xFormers is disabled and FlashAttention is absent, which is precisely what the
    kernel probe can now decide.
    """
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        captured["is_causal"] = kwargs.get("is_causal")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.SDPA,
        n_kv_heads = 1,
        n_groups = 1,
    )
    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 6,
        kv_seq_len = 6,
        n_heads = 1,
        head_dim = 1,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        sliding_window = 3,
    )
    Q = torch.zeros(1, 1, 6, 1)

    attention_dispatch.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)

    mask = captured["mask"]
    assert mask is not None, "a declared window must not fall through to plain is_causal"
    assert captured["is_causal"] is False
    assert mask.shape == (1, 1, 6, 6)
    # Row 5 sees 3, 4, 5 and nothing older;
    assert [bool(v) for v in mask[0, 0, 5]] == [False, False, False, True, True, True]


def test_run_attention_sdpa_leaves_a_short_sequence_alone(monkeypatch):
    # Shorter than the window: nothing to clamp, and the cheap is_causal path must survive.
    captured = {}
    monkeypatch.setattr(
        attention_dispatch,
        "scaled_dot_product_attention",
        lambda Q, K, V, **kw: (captured.update(kw), torch.zeros_like(Q))[1],
    )
    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.SDPA, n_kv_heads = 1, n_groups = 1
    )
    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 1,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        sliding_window = 8,
    )
    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    assert captured["attn_mask"] is None and captured["is_causal"] is True


def test_mistral_hands_the_dispatcher_its_configured_window():
    """The context Mistral builds omitted `sliding_window` entirely, so even a correct SDPA
    window path had nothing to act on."""
    import ast
    from pathlib import Path

    src = Path(attention_dispatch.__file__).resolve().parents[1] / "models" / "mistral.py"
    tree = ast.parse(src.read_text(encoding = "utf-8"))
    contexts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AttentionContext"
    ]
    assert contexts, "AttentionContext construction not found in mistral.py"
    for call in contexts:
        assert "sliding_window" in {kw.arg for kw in call.keywords}


def test_a_zero_configured_window_is_full_causal_not_a_blank_mask():
    """`sliding_window = 0` means "no local attention", the same as absent -- which is how
    Mistral's own mask builders read it. Passing the 0 through makes the SDPA lower bound
    `q_pos - (0 - 1)` sit above the causal upper bound, so every position is masked and the
    layer returns nothing at all."""
    import ast
    from pathlib import Path

    src = Path(attention_dispatch.__file__).resolve().parents[1] / "models" / "mistral.py"
    text = src.read_text(encoding = "utf-8")
    assert "isinstance(sw_cfg, int) and sw_cfg <= 0" in text, (
        "a non-positive configured window must be normalised before it reaches window_size "
        "or the dispatcher"
    )
    ast.parse(text)


def test_run_attention_sdpa_ignores_a_zero_window(monkeypatch):
    # Belt and braces at the dispatcher: even handed a zero, it must not build a mask that
    captured = {}
    monkeypatch.setattr(
        attention_dispatch,
        "scaled_dot_product_attention",
        lambda Q, K, V, **kw: (captured.update(kw), torch.zeros_like(Q))[1],
    )
    config = attention_dispatch.AttentionConfig(
        backend = attention_dispatch.SDPA, n_kv_heads = 1, n_groups = 1
    )
    context = attention_dispatch.AttentionContext(
        bsz = 1,
        q_len = 4,
        kv_seq_len = 4,
        n_heads = 1,
        head_dim = 1,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        sliding_window = 0,
    )
    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(config = config, context = context, Q = Q, K = Q, V = Q)
    mask = captured["attn_mask"]
    assert mask is None or bool(mask.any()), "a zero window must not mask everything"


def test_the_window_mask_is_built_once_per_shape(monkeypatch):
    """Every layer asks for the identical mask, and at 32K that tensor is 1 GiB with two more
    alive while it is built. Rebuilding it per layer is how this SDPA fallback OOMs a run that
    xFormers or flash would have carried."""
    attention_dispatch._WINDOW_MASK_CACHE.clear()
    built = []
    real_arange = torch.arange

    def _counting_arange(*args, **kwargs):
        built.append(1)
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(attention_dispatch.torch, "arange", _counting_arange)

    first = attention_dispatch._windowed_causal_mask(6, 6, 3, torch.device("cpu"))
    calls_after_first = len(built)
    second = attention_dispatch._windowed_causal_mask(6, 6, 3, torch.device("cpu"))

    assert second is first, "the same shape and window must not be rebuilt"
    assert len(built) == calls_after_first, "a cache hit must allocate nothing"
    assert [bool(v) for v in first[0, 0, 5]] == [False, False, False, True, True, True]

    # A different window is a different mask, not a stale hit.
    third = attention_dispatch._windowed_causal_mask(6, 6, 2, torch.device("cpu"))
    assert third is not first
    assert [bool(v) for v in third[0, 0, 5]] == [False, False, False, False, True, True]
    # ...and so is a different shape.
    assert attention_dispatch._windowed_causal_mask(4, 4, 3, torch.device("cpu")) is not third
    attention_dispatch._WINDOW_MASK_CACHE.clear()


def test_the_outgoing_window_mask_is_freed_before_its_replacement(monkeypatch):
    """A shape change must not hold two dense masks at once. Dynamic-length training walks
    through shapes, and at 32K each mask is 1 GiB on top of the construction temporaries."""
    attention_dispatch._WINDOW_MASK_CACHE.clear()
    device = torch.device("cpu")
    first = attention_dispatch._windowed_causal_mask(6, 6, 3, device)
    live = weakref.ref(first)
    del first

    cached_during_build = []
    real_arange = torch.arange

    def _observing_arange(*args, **kwargs):
        cached_during_build.append(live() is not None)
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(attention_dispatch.torch, "arange", _observing_arange)
    attention_dispatch._windowed_causal_mask(8, 8, 3, device)

    assert cached_during_build, "the replacement must actually have been built"
    assert not any(
        cached_during_build
    ), "the previous mask was still alive while its replacement was allocated"
    attention_dispatch._WINDOW_MASK_CACHE.clear()
