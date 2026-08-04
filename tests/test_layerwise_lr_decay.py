"""CPU tests for layer-wise LR decay param grouping.

Loads the stdlib-only module directly so no GPU / unsloth import is needed.
"""

import importlib.util
import os

import pytest

MODULE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "unsloth", "optimizers", "layerwise_lr.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_layerwise_lr_standalone", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()
make_groups = mod.make_layerwise_lr_param_groups
get_layer_index = mod.get_layer_index
check_compat = mod.check_layerwise_lr_compat


class FakeParam:
    def __init__(
        self,
        tag,
        requires_grad = True,
    ):
        self.tag = tag
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return list(self._named)


def build_model(
    num_layers = 4,
    with_head = True,
    with_embedding = True,
    frozen_layer = None,
):
    named = []
    for i in range(num_layers):
        rg = not (frozen_layer is not None and i == frozen_layer)
        name = f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight"
        named.append((name, FakeParam(f"layer{i}", requires_grad = rg)))
    if with_head:
        named.append(("base_model.model.lm_head.weight", FakeParam("lm_head")))
    if with_embedding:
        named.append(
            (
                "base_model.model.model.embed_tokens.modules_to_save.default.weight",
                FakeParam("embed"),
            )
        )
    return FakeModel(named)


def lr_by_tag(groups):
    out = {}
    for group in groups:
        for param in group["params"]:
            out[param.tag] = group["lr"]
    return out


def test_layer_index_parsing():
    assert get_layer_index("a.layers.7.b") == 7
    assert get_layer_index("model.layers.0.mlp") == 0
    assert get_layer_index("layers.3.q_proj.weight") == 3  # top-level stack, no prefix
    assert get_layer_index("model.embed_tokens.weight") is None


def test_decay_math_and_boundaries():
    base, decay = 1e-3, 0.9
    groups = make_groups(
        build_model(4), lr = base, weight_decay = 0.01, layerwise_lr_decay = decay, verbose = False
    )
    lrs = lr_by_tag(groups)
    assert lrs["layer3"] == pytest.approx(base)  # top layer = base
    assert lrs["layer2"] == pytest.approx(base * decay**1)
    assert lrs["layer1"] == pytest.approx(base * decay**2)
    assert lrs["layer0"] == pytest.approx(base * decay**3)  # deepest decay
    assert lrs["lm_head"] == pytest.approx(base)  # non-block = top
    assert lrs["embed"] == pytest.approx(base * decay**3)  # inherits shallowest


def test_each_trainable_param_in_exactly_one_group():
    groups = make_groups(
        build_model(4), lr = 1e-3, weight_decay = 0.0, layerwise_lr_decay = 0.9, verbose = False
    )
    tags = [p.tag for g in groups for p in g["params"]]
    assert sorted(tags) == sorted(["layer0", "layer1", "layer2", "layer3", "lm_head", "embed"])
    assert len(tags) == len(set(tags))


def test_frozen_params_excluded():
    groups = make_groups(
        build_model(4, frozen_layer = 1),
        lr = 1e-3,
        weight_decay = 0.0,
        layerwise_lr_decay = 0.9,
        verbose = False,
    )
    assert "layer1" not in lr_by_tag(groups)


def test_embedding_lr_overrides_decay():
    groups = make_groups(
        build_model(4),
        lr = 1e-3,
        weight_decay = 0.0,
        layerwise_lr_decay = 0.9,
        embedding_lr = 7e-6,
        verbose = False,
    )
    assert lr_by_tag(groups)["embed"] == pytest.approx(7e-6)


def test_decay_one_is_flat():
    base = 1e-3
    groups = make_groups(
        build_model(4), lr = base, weight_decay = 0.0, layerwise_lr_decay = 1.0, verbose = False
    )
    assert all(v == pytest.approx(base) for v in lr_by_tag(groups).values())


def test_groups_sorted_and_weight_decay_propagated():
    groups = make_groups(
        build_model(4), lr = 1e-3, weight_decay = 0.05, layerwise_lr_decay = 0.9, verbose = False
    )
    rates = [g["lr"] for g in groups]
    assert rates == sorted(rates)
    assert all(g["weight_decay"] == 0.05 for g in groups)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_decay_raises(bad):
    with pytest.raises(ValueError):
        make_groups(build_model(2), lr = 1e-3, weight_decay = 0.0, layerwise_lr_decay = bad)


def test_num_layers_uses_full_model_depth():
    base, decay = 1e-3, 0.9
    # Only layer 0 is trainable, but the real model has 32 layers.
    groups = make_groups(
        build_model(1), lr = base, weight_decay = 0.0, layerwise_lr_decay = decay,
        num_layers = 32, verbose = False,
    )
    assert lr_by_tag(groups)["layer0"] == pytest.approx(base * decay ** 31)


def test_num_layers_clamped_to_seen_indices():
    base, decay = 1e-3, 0.9
    # A too-small num_layers must not produce negative exponents (lr > base).
    groups = make_groups(
        build_model(4), lr = base, weight_decay = 0.0, layerwise_lr_decay = decay,
        num_layers = 2, verbose = False,
    )
    lrs = lr_by_tag(groups)
    assert lrs["layer3"] == pytest.approx(base)
    assert lrs["layer0"] == pytest.approx(base * decay ** 3)


def test_compat_check_rejects_q_galore_combo():
    with pytest.raises(ValueError):
        check_compat(0.9, object())


@pytest.mark.parametrize("decay,q_galore", [(None, object()), (1.0, object()), (0.9, None)])
def test_compat_check_allows(decay, q_galore):
    check_compat(decay, q_galore)


def test_separate_stacks_decay_independently():
    base, decay = 1e-3, 0.9
    named = []
    for i in range(4):  # 4-layer decoder
        named.append((f"model.layers.{i}.self_attn.q_proj.weight", FakeParam(f"dec{i}")))
    for i in range(6):  # deeper 6-layer vision tower
        named.append(
            (f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight", FakeParam(f"vis{i}"))
        )
    # config depth (decoder) must not leak into the vision stack.
    groups = make_groups(
        FakeModel(named), lr = base, weight_decay = 0.0, layerwise_lr_decay = decay,
        num_layers = 4, verbose = False,
    )
    lrs = lr_by_tag(groups)
    assert lrs["dec3"] == pytest.approx(base)               # decoder top = base
    assert lrs["dec0"] == pytest.approx(base * decay ** 3)  # decoder depth 4
    assert lrs["vis5"] == pytest.approx(base)               # vision top = base
    assert lrs["vis0"] == pytest.approx(base * decay ** 5)  # vision depth 6, independent
