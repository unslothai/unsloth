"""Unit tests for unsloth.save._normalize_tied_weights_keys.

Regression for the NemotronH (model_type "nemotron_h") save / GGUF-export crash.
transformers >= 5 ``save_pretrained`` reads ``module._tied_weights_keys.keys()``,
which raises ``AttributeError: 'list' object has no attribute 'keys'`` for any module
that still declares the legacy list form (NemotronH uses it on
``backbone.layers.N.mixer.*_proj``). The helper coerces that list/tuple into the dict
form transformers 5.x expects, mapping each key to itself so only-keys-are-read
behaviour is preserved.

These tests exercise the helper directly with a tiny ``torch.nn.Module`` tree, so they
are version-independent: they do not download a model or run a full ``save_pretrained``.
"""

import torch

from unsloth.save import _normalize_tied_weights_keys


def _build_tree():
    """root -> mixer, mirroring NemotronH's backbone.layers.N.mixer nesting."""
    root = torch.nn.Module()
    mixer = torch.nn.Module()
    root.add_module("mixer", mixer)
    return root, mixer


def test_list_tied_weights_keys_become_dict():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ["q_proj.weight", "o_proj.weight"]
    _normalize_tied_weights_keys(root)
    assert mixer._tied_weights_keys == {
        "q_proj.weight": "q_proj.weight",
        "o_proj.weight": "o_proj.weight",
    }
    # The dict must satisfy the transformers >= 5 ``.keys()`` access without raising.
    assert list(mixer._tied_weights_keys.keys()) == ["q_proj.weight", "o_proj.weight"]


def test_tuple_tied_weights_keys_become_dict():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ("lm_head.weight",)
    _normalize_tied_weights_keys(root)
    assert mixer._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}


def test_existing_dict_is_left_unchanged():
    root, mixer = _build_tree()
    original = {"a.weight": "b.weight"}
    mixer._tied_weights_keys = original
    _normalize_tied_weights_keys(root)
    # A dict already satisfies ``.keys()``; it must be left untouched, not rebuilt.
    assert mixer._tied_weights_keys is original


def test_set_keys_become_dict():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = {"q_proj.weight"}
    _normalize_tied_weights_keys(root)
    assert mixer._tied_weights_keys == {"q_proj.weight": "q_proj.weight"}


def test_none_is_left_unchanged_but_empty_becomes_dict():
    root, mixer = _build_tree()
    root._tied_weights_keys = None
    mixer._tied_weights_keys = []
    _normalize_tied_weights_keys(root)
    # None means "no tied weights" to transformers and is left alone; an empty
    # list still lacks ``.keys()`` (transformers only skips on None), so it must
    # be coerced to an empty dict.
    assert root._tied_weights_keys is None
    assert mixer._tied_weights_keys == {}


def test_empty_tuple_and_set_become_dict():
    root, mixer = _build_tree()
    root._tied_weights_keys = ()
    mixer._tied_weights_keys = set()
    _normalize_tied_weights_keys(root)
    assert root._tied_weights_keys == {}
    assert mixer._tied_weights_keys == {}


def test_idempotent():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ["w.weight"]
    _normalize_tied_weights_keys(root)
    first = mixer._tied_weights_keys
    _normalize_tied_weights_keys(root)
    assert mixer._tied_weights_keys == first == {"w.weight": "w.weight"}


def test_modules_without_attribute_are_ignored():
    root, mixer = _build_tree()
    # Neither module declares _tied_weights_keys.
    _normalize_tied_weights_keys(root)  # must not raise
    assert not hasattr(mixer, "_tied_weights_keys")


def test_model_without_modules_method_does_not_raise():
    class NoModules:
        pass

    # Best-effort: a save must never fail over this, even on an odd object.
    _normalize_tied_weights_keys(NoModules())
