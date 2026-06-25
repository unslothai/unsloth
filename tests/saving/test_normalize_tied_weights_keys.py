"""Unit tests for the tied-weights-keys coercion used by unsloth.save.

Regression for the NemotronH save / GGUF-export crash: transformers >= 5
``save_pretrained`` reads ``_tied_weights_keys.keys()`` and raises on the legacy list
form. Exercised on tiny module trees, no model download.
"""

import pytest
import torch

from unsloth.save import (
    _coerce_tied_weights_keys_to_dict,
    _normalize_tied_weights_keys_for_save,
    _restore_tied_weights_keys,
)


def _build_tree():
    root = torch.nn.Module()
    mixer = torch.nn.Module()
    root.add_module("mixer", mixer)
    return root, mixer


def test_list_becomes_dict_and_restores():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ["q_proj.weight", "o_proj.weight"]
    originals = _coerce_tied_weights_keys_to_dict(root)
    assert mixer._tied_weights_keys == {
        "q_proj.weight": "q_proj.weight",
        "o_proj.weight": "o_proj.weight",
    }
    _restore_tied_weights_keys(originals)
    assert mixer._tied_weights_keys == ["q_proj.weight", "o_proj.weight"]


def test_tuple_and_set_become_dict():
    root, mixer = _build_tree()
    root._tied_weights_keys = ("lm_head.weight",)
    mixer._tied_weights_keys = {"q_proj.weight"}
    _coerce_tied_weights_keys_to_dict(root)
    assert root._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}
    assert mixer._tied_weights_keys == {"q_proj.weight": "q_proj.weight"}


def test_empty_containers_become_dict():
    root, mixer = _build_tree()
    root._tied_weights_keys = []
    mixer._tied_weights_keys = ()
    _coerce_tied_weights_keys_to_dict(root)
    # transformers skips only None; an empty list still hits .keys().
    assert root._tied_weights_keys == {} and mixer._tied_weights_keys == {}


def test_none_and_existing_dict_are_left_unchanged():
    root, mixer = _build_tree()
    root._tied_weights_keys = None
    original = {"a.weight": "b.weight"}
    mixer._tied_weights_keys = original
    originals = _coerce_tied_weights_keys_to_dict(root)
    assert root._tied_weights_keys is None
    assert mixer._tied_weights_keys is original  # untouched, not rebuilt
    assert originals == []  # nothing to restore


def test_model_without_modules_method_does_not_raise():
    class NoModules:
        pass

    assert _coerce_tied_weights_keys_to_dict(NoModules()) == []


def test_decorator_coerces_during_save_then_restores():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ["lm_head.weight"]
    seen = {}

    @_normalize_tied_weights_keys_for_save
    def save(model):
        seen["keys"] = dict(model.mixer._tied_weights_keys)
        return "ok"

    assert save(root) == "ok"
    # Dict form was visible to the save, list form restored afterwards.
    assert seen["keys"] == {"lm_head.weight": "lm_head.weight"}
    assert mixer._tied_weights_keys == ["lm_head.weight"]


def test_decorator_restores_on_exception():
    root, mixer = _build_tree()
    mixer._tied_weights_keys = ["lm_head.weight"]

    @_normalize_tied_weights_keys_for_save
    def save(model):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        save(root)
    assert mixer._tied_weights_keys == ["lm_head.weight"]


def test_decorator_finds_model_in_kwargs_and_positional():
    # unsloth_save_model / unsloth_generic_save pass model= as a keyword; the gguf path
    # binds it as the first positional (method ``self``). Both must be coerced.
    for call in (lambda f, r: f(model = r), lambda f, r: f(r)):
        root, mixer = _build_tree()
        mixer._tied_weights_keys = ["w.weight"]
        captured = {}

        @_normalize_tied_weights_keys_for_save
        def save(model):
            captured["dict"] = isinstance(model.mixer._tied_weights_keys, dict)

        call(save, root)
        assert captured["dict"] is True
        assert mixer._tied_weights_keys == ["w.weight"]
