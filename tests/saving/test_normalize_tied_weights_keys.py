"""Unit tests for unsloth.save._normalize_tied_weights_keys.

Regression for the NemotronH save / GGUF-export crash: transformers >= 5
``save_pretrained`` reads ``_tied_weights_keys.keys()`` and raises on the legacy list
form. Exercised directly on a tiny module tree (no model download).
"""

import torch

from unsloth.save import _normalize_tied_weights_keys


def _build_tree():
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
    assert mixer._tied_weights_keys is original  # untouched, not rebuilt


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
    # transformers skips only None; an empty list still hits .keys().
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
    _normalize_tied_weights_keys(root)
    assert not hasattr(mixer, "_tied_weights_keys")


def test_model_without_modules_method_does_not_raise():
    class NoModules:
        pass

    _normalize_tied_weights_keys(NoModules())
