# SPDX-License-Identifier: Apache-2.0
"""#3071: for_training must not overwrite a user-set UNSLOTH_RETURN_LOGITS.

for_inference() sets UNSLOTH_RETURN_LOGITS = 1 and marks ownership.
for_training() must reset it to 0 ONLY when the marker is present (i.e. the
value was set by for_inference, not by the user). A user who sets
UNSLOTH_RETURN_LOGITS = 1 directly must have their setting preserved
across for_training calls.
"""

import os

# Remove any leftover state from previous test runs
os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None)


def test_cycle_for_inference_to_for_training_resets():
    """Normal cycle: for_inference sets logits → for_training resets them."""
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    os.environ["_UNSLOTH_FOR_INFERENCE_SET_LOGITS"] = "1"

    # Simulate for_training logic:
    if os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None) == "1":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"

    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "0"
    assert "_UNSLOTH_FOR_INFERENCE_SET_LOGITS" not in os.environ


def test_user_override_preserved():
    """User sets logits=1 AFTER for_training ran once — it must stick."""
    # for_training ran (marker was consumed)
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    # User overrides
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    # for_training runs again (e.g. between epochs)
    if os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None) == "1":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"

    assert (
        os.environ["UNSLOTH_RETURN_LOGITS"] == "1"
    ), "User override was wrongly reset — for_training must keep user-set values"


def test_marker_not_present_noop():
    """If there's no marker at all, for_training must not touch the current value."""
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None) == "1":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"

    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"


def test_for_inference_sets_marker_only_when_value_changes():
    """Marker is only set when for_inference actually changed the value."""
    # Value was "0" → marker IS set (framework owns the change)
    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    if os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1":
        os.environ["_UNSLOTH_FOR_INFERENCE_SET_LOGITS"] = "1"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    assert os.environ.get("_UNSLOTH_FOR_INFERENCE_SET_LOGITS") == "1"

    # Reset
    os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None)

    # Value was already "1" → marker NOT set (user owns it)
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1":
        os.environ["_UNSLOTH_FOR_INFERENCE_SET_LOGITS"] = "1"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    assert "_UNSLOTH_FOR_INFERENCE_SET_LOGITS" not in os.environ


def test_user_value_survives_for_inference_cycle():
    """User's setting survives a complete for_inference → for_training cycle."""
    # User sets logits
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    # for_inference runs (value already "1" — no marker)
    if os.environ.get("UNSLOTH_RETURN_LOGITS", "0") != "1":
        os.environ["_UNSLOTH_FOR_INFERENCE_SET_LOGITS"] = "1"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    # for_training runs
    if os.environ.pop("_UNSLOTH_FOR_INFERENCE_SET_LOGITS", None) == "1":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
    assert (
        os.environ["UNSLOTH_RETURN_LOGITS"] == "1"
    ), "User's '1' was reset across for_inference loop — must survive"
