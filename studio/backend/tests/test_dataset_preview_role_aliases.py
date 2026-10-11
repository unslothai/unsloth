# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The preview's copy of the alias map matched the RAW role, while the zoo matches
`role.strip().lower()` (unslothai/unsloth-zoo#1225), so "Human"/"GPT" trained differently."""

import json
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils.dataset_format import (  # noqa: E402
    _ROLE_MAP,
    _normalize_role_alias,
    _standardize_sharegpt_row,
)

# Left previews as right, matching what standardize_data_formats writes for the trainer.
_SPELLINGS = [
    ("human", "user"),
    ("Human", "user"),
    ("HUMAN", "user"),
    ("hUmAn", "user"),
    (" human", "user"),
    ("human ", "user"),
    ("\thuman\n", "user"),
    ("  Human  ", "user"),
    ("user", "user"),
    ("User", "user"),
    ("USER", "user"),
    (" user ", "user"),
    ("input", "user"),
    ("Input", "user"),
    ("INPUT", "user"),
    ("gpt", "assistant"),
    ("GPT", "assistant"),
    ("Gpt", "assistant"),
    (" gpt\t", "assistant"),
    ("assistant", "assistant"),
    ("Assistant", "assistant"),
    ("ASSISTANT", "assistant"),
    ("output", "assistant"),
    ("Output", "assistant"),
    ("OUTPUT", "assistant"),
    ("system", "system"),
    ("System", "system"),
    ("SYSTEM", "system"),
    ("\tSystem\n", "system"),
]


@pytest.mark.parametrize("spelling, expected", _SPELLINGS)
def test_every_spelling_of_a_known_alias_normalises(spelling, expected):
    assert _ROLE_MAP.get(_normalize_role_alias(spelling)) == expected


@pytest.mark.parametrize("spelling, expected", _SPELLINGS)
def test_the_preview_row_standardiser_maps_every_spelling(spelling, expected):
    row = {"conversations": [{"from": spelling, "value": "hello"}]}
    result = _standardize_sharegpt_row(row, "conversations")
    assert result["conversations"][0]["role"] == expected
    assert result["conversations"][0]["content"] == "hello"


def test_the_role_key_is_read_before_the_from_key():
    row = {"conversations": [{"role": "GPT", "from": "Human", "value": "x"}]}
    assert (
        _standardize_sharegpt_row(row, "conversations")["conversations"][0]["role"] == "assistant"
    )


def test_the_content_key_is_read_before_the_value_key():
    row = {"conversations": [{"from": "Human", "content": "c", "value": "v"}]}
    message = _standardize_sharegpt_row(row, "conversations")["conversations"][0]
    assert message["content"] == "c"


def test_the_map_keys_are_all_already_normalised():
    for key in _ROLE_MAP:
        assert key == _normalize_role_alias(key), f"{key!r} is not in normalised form"


def test_the_alias_set_is_the_one_the_trainer_accepts():
    """Cross-repo pin: an alias the zoo grows has to reach the preview too."""
    # Not importorskip: unsloth_zoo raises NotImplementedError, not ImportError, on a host
    # with no recognised accelerator, which is every GitHub macOS and Windows runner.
    try:
        import unsloth_zoo.dataset_utils as dataset_utils
    except Exception as error:
        pytest.skip(f"unsloth_zoo is not importable here: {type(error).__name__}: {error}")
    import inspect

    defaults = inspect.signature(dataset_utils.standardize_data_formats).parameters
    expected = {}
    for parameter, role in (
        ("aliases_for_system", "system"),
        ("aliases_for_user", "user"),
        ("aliases_for_assistant", "assistant"),
    ):
        for alias in defaults[parameter].default:
            expected[_normalize_role_alias(alias)] = role
    assert _ROLE_MAP == expected


def test_an_unknown_role_is_still_shown_as_written():
    row = {"conversations": [{"from": "Narrator", "value": "x"}]}
    assert _standardize_sharegpt_row(row, "conversations")["conversations"][0]["role"] == "Narrator"


@pytest.mark.parametrize("role", [None, "", "   ", "\t"])
def test_a_missing_or_blank_role_still_defaults_to_user(role):
    row = {"conversations": [{"from": role, "value": "x"}]}
    assert _standardize_sharegpt_row(row, "conversations")["conversations"][0]["role"] == "user"


def test_a_non_string_role_does_not_raise():
    row = {"conversations": [{"from": 7, "value": "x"}]}
    assert _standardize_sharegpt_row(row, "conversations")["conversations"][0]["role"] == "7"


def test_a_none_content_becomes_an_empty_string():
    row = {"conversations": [{"from": "Human", "value": None}]}
    assert _standardize_sharegpt_row(row, "conversations")["conversations"][0]["content"] == ""


def test_a_non_list_chat_column_is_returned_untouched():
    row = {"conversations": "not a list", "other": 1}
    assert _standardize_sharegpt_row(row, "conversations") == row


def test_non_dict_messages_are_skipped():
    row = {"conversations": ["plain", {"from": "GPT", "value": "y"}]}
    messages = _standardize_sharegpt_row(row, "conversations")["conversations"]
    assert messages == [{"role": "assistant", "content": "y"}]


@pytest.fixture
def isolated_studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    return tmp_path


def test_the_check_format_route_previews_a_mixed_case_sharegpt_upload(isolated_studio_home):
    pytest.importorskip("datasets")
    from routes import datasets as datasets_route
    from utils.paths import dataset_uploads_root

    conversation = [
        {"from": "  System ", "value": "be brief"},
        {"from": "Human", "value": "hello"},
        {"from": "GPT", "value": "hi"},
        {"from": "Input", "value": "again"},
        {"from": "OUTPUT", "value": "sure"},
    ]
    uploads = dataset_uploads_root()
    uploads.mkdir(parents = True, exist_ok = True)
    path = uploads / "mixed_case_sharegpt.jsonl"
    path.write_text(
        "\n".join(json.dumps({"conversations": conversation}) for _ in range(3)) + "\n",
        encoding = "utf-8",
    )

    request = datasets_route.CheckFormatRequest(dataset_name = str(path))
    response = datasets_route.check_format(request, hf_token = None, current_subject = "test")

    assert response.detected_format == "sharegpt"
    assert response.preview_samples, "the route returned no preview rows"
    for sample in response.preview_samples:
        messages = sample["conversations"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]


def test_the_training_standardiser_normalises_the_same_way():
    """Studio training reaches the local `standardize_chat_format`, not the zoo helper."""
    from utils.datasets.format_conversion import _normalize_role_alias as training_normalize
    for spelling, expected in _SPELLINGS:
        assert _ROLE_MAP[training_normalize(spelling)] == expected


@pytest.mark.parametrize("spelling, expected", _SPELLINGS)
def test_standardize_chat_format_maps_every_spelling(spelling, expected):
    datasets = pytest.importorskip("datasets")
    from utils.datasets.format_conversion import standardize_chat_format

    dataset = datasets.Dataset.from_list(
        [{"conversations": [{"from": spelling, "value": "hello"}]}] * 4
    )
    result = standardize_chat_format(dataset, num_proc = 1)
    message = result[0]["conversations"][0]
    assert message["role"] == expected
    assert message["content"] == "hello"


def test_standardize_chat_format_leaves_an_unknown_role_as_written():
    datasets = pytest.importorskip("datasets")
    from utils.datasets.format_conversion import standardize_chat_format

    dataset = datasets.Dataset.from_list(
        [{"conversations": [{"from": "Narrator", "value": "x"}]}] * 4
    )
    assert standardize_chat_format(dataset, num_proc = 1)[0]["conversations"][0]["role"] == "Narrator"


def test_the_preview_and_the_training_path_agree_on_every_spelling():
    datasets = pytest.importorskip("datasets")
    from utils.datasets.format_conversion import standardize_chat_format

    conversation = [{"from": spelling, "value": "x"} for spelling, _ in _SPELLINGS]
    trained = standardize_chat_format(
        datasets.Dataset.from_list([{"conversations": conversation}] * 4), num_proc = 1
    )[0]["conversations"]
    previewed = _standardize_sharegpt_row({"conversations": conversation}, "conversations")[
        "conversations"
    ]
    assert [m["role"] for m in trained] == [m["role"] for m in previewed]


@pytest.mark.parametrize("role", [None, "", "   ", "\t"])
def test_the_training_path_defaults_a_blank_role_like_the_preview(role):
    """The training path preserved a blank role and most chat templates reject one."""
    datasets = pytest.importorskip("datasets")
    from utils.datasets.format_conversion import standardize_chat_format

    dataset = datasets.Dataset.from_list(
        [{"conversations": [{"from": role, "value": "x"}, {"from": "gpt", "value": "y"}]}] * 4
    )
    trained = standardize_chat_format(dataset, num_proc = 1)[0]["conversations"]
    previewed = _standardize_sharegpt_row(
        {"conversations": [{"from": role, "value": "x"}, {"from": "gpt", "value": "y"}]},
        "conversations",
    )["conversations"]

    assert trained[0]["role"] == "user"
    assert [m["role"] for m in trained] == [m["role"] for m in previewed]


@pytest.mark.parametrize(
    "message, expected",
    [
        # Blank inferred role, usable fallback in the same message: training's `is None`
        # check skipped it and labelled an assistant turn as user.
        ({"role": "", "from": "gpt", "content": "answer"}, "assistant"),
        ({"role": "   ", "from": "Human", "content": "q"}, "user"),
        ({"role": None, "from": "gpt", "content": "answer"}, "assistant"),
        # NEGATIVE CONTROLS: no fallback stays "user"; a present role wins over one.
        ({"role": "", "content": "answer"}, "user"),
        ({"role": "gpt", "from": "human", "content": "answer"}, "assistant"),
    ],
)
def test_a_blank_inferred_role_consults_the_same_fallback_the_preview_does(message, expected):
    datasets = pytest.importorskip("datasets")
    from utils.datasets.format_conversion import standardize_chat_format

    rows = [{"conversations": [dict(message), {"role": "user", "content": "x"}]}] * 4
    trained = standardize_chat_format(datasets.Dataset.from_list(rows), num_proc = 1)
    assert trained[0]["conversations"][0]["role"] == expected

    previewed = _standardize_sharegpt_row({"conversations": [dict(message)]}, "conversations")
    assert (
        previewed["conversations"][0]["role"] == expected
    ), "the preview and the training path disagree on this message"
