# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.user_assets_validation import UserAssetValidationError, validate_recipe_payload
from models.user_assets import PortableTrainingConfig


def test_proxy_authorization_is_rejected_and_redacted_by_the_same_policy():
    payload = {"headers": {"Proxy-Authorization": "Basic secret"}}
    with pytest.raises(UserAssetValidationError, match = "secret fields"):
        validate_recipe_payload(payload)
    clean, paths = validate_recipe_payload(payload, legacy = True)
    assert clean == {"headers": {}}
    assert paths == ['$.headers["Proxy-Authorization"]']


@pytest.mark.parametrize("eval_steps", [1, 0.1])
def test_training_contract_accepts_integer_and_fractional_eval_steps(eval_steps):
    training = PortableTrainingConfig.Training(
        max_seq_length = 2048,
        num_epochs = 1,
        learning_rate = 0.0002,
        batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 0,
        max_steps = 10,
        save_steps = 1,
        eval_steps = eval_steps,
        weight_decay = 0,
        random_seed = 1,
        packing = False,
        train_on_completions = False,
        gradient_checkpointing = False,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
    )
    assert training.eval_steps == eval_steps
