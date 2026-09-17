# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned-snapshot path must not reach PEFT through ``name_or_path``.

``restore_hf_cache_repo_identity`` runs in ``UnslothTrainer.load_model`` *before*
``get_peft_model``, so at that point there is no ``peft_config`` for its adapter branch
to repair. PEFT then derives the adapter's ``base_model_name_or_path`` from
``model.__dict__["name_or_path"]``:

    # peft/mapping_func.py
    new_name = model.__dict__.get("name_or_path", None)
    peft_config.base_model_name_or_path = new_name

``PreTrainedModel.__init__`` copies ``config.name_or_path`` onto the instance, so
restoring only ``config._name_or_path`` leaves that slot holding the machine-local
snapshot path. It then travels into ``adapter_config.json``, every
``checkpoint-*/adapter_config.json``, the run card, ``export_metadata.json`` and the
model card uploaded by ``push_to_hub`` -- none of which are loadable on another machine.

The existing coverage in ``test_model_identity.py`` asserts the *call site* via AST,
which stays green even when the call cannot do anything, so these are behavioural.
"""

from types import SimpleNamespace

from utils.models.model_identity import restore_hf_cache_repo_identity


_REPO = "unsloth/Llama-3.2-1B-Instruct"
_SNAPSHOT = (
    "/home/user/.cache/huggingface/hub/"
    "models--unsloth--Llama-3.2-1B-Instruct/snapshots/0123456789abcdef"
)


def _loaded_model(**overrides):
    """A transformers-shaped model loaded from a pinned snapshot, pre-PEFT."""
    return SimpleNamespace(
        config = SimpleNamespace(_name_or_path = _SNAPSHOT),
        name_or_path = _SNAPSHOT,
        **overrides,
    )


def _peft_derived_base_model_name(model) -> object:
    """Replicate PEFT's own derivation, so this cannot pass vacuously."""
    return model.__dict__.get("name_or_path", None)


def test_restore_rewrites_the_instance_name_that_peft_reads():
    model = _loaded_model()

    assert restore_hf_cache_repo_identity(model, _SNAPSHOT) == _REPO
    assert model.name_or_path == _REPO
    assert (
        _peft_derived_base_model_name(model) == _REPO
    ), "PEFT would stamp a machine-local snapshot path into adapter_config.json"


def test_config_and_instance_identity_agree_after_restore():
    model = _loaded_model()

    restore_hf_cache_repo_identity(model, _SNAPSHOT)

    assert model.config._name_or_path == model.name_or_path == _REPO


def test_restore_is_still_correct_once_the_model_is_wrapped_by_peft():
    adapter = SimpleNamespace(base_model_name_or_path = _SNAPSHOT)
    model = _loaded_model(peft_config = {"default": adapter})

    restore_hf_cache_repo_identity(model, _SNAPSHOT)

    assert adapter.base_model_name_or_path == _REPO
    assert model.name_or_path == _REPO


def test_a_repo_mismatch_leaves_the_instance_name_untouched():
    model = _loaded_model()

    assert restore_hf_cache_repo_identity(model, _SNAPSHOT, expected_repo_id = "someone/else") is None
    assert model.name_or_path == _SNAPSHOT


def test_an_ordinary_local_model_keeps_its_own_name():
    local = "/srv/models/my-finetune"
    model = SimpleNamespace(
        config = SimpleNamespace(_name_or_path = local),
        name_or_path = local,
    )

    assert restore_hf_cache_repo_identity(model, local) is None
    assert model.name_or_path == local


def test_an_existing_hub_id_is_not_rewritten_by_an_unrelated_snapshot():
    model = SimpleNamespace(
        config = SimpleNamespace(_name_or_path = "org/other-model"),
        name_or_path = "org/other-model",
    )

    restore_hf_cache_repo_identity(model, _SNAPSHOT)

    assert model.name_or_path == "org/other-model"
