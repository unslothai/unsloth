import importlib.util
from pathlib import Path

import pytest


BACKEND = Path(__file__).resolve().parents[2] / "studio/backend"
IMPLEMENTATIONS = (
    BACKEND / "utils/datasets/format_detection.py",
    BACKEND / "hub/utils/dataset_format.py",
)


@pytest.fixture(params=IMPLEMENTATIONS, ids=("datasets", "hub"))
def detect_vlm_dataset_structure(request):
    spec = importlib.util.spec_from_file_location(
        f"_vlm_format_detection_{request.param.parent.name}", request.param
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.detect_vlm_dataset_structure


@pytest.mark.parametrize(
    "system_content",
    (
        [{"type": "text", "text": "Answer the user's question."}],
        "Answer the user's question.",
    ),
    ids=("structured-system-message", "string-system-prompt"),
)
def test_detects_image_after_system_message(detect_vlm_dataset_structure, system_content):
    dataset = [
        {
            "messages": [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown?"},
                        {"type": "image", "image": "image.jpg"},
                    ],
                },
            ]
        }
    ]

    assert detect_vlm_dataset_structure(dataset)["format"] == "vlm_messages"


def test_detects_image_in_first_message(detect_vlm_dataset_structure):
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "image.jpg"}],
                }
            ]
        }
    ]

    assert detect_vlm_dataset_structure(dataset)["format"] == "vlm_messages"


def test_detects_llava_index_after_system_message(detect_vlm_dataset_structure):
    dataset = [
        {
            "messages": [
                {"role": "system", "content": "Answer the user's question."},
                {
                    "role": "user",
                    "content": [{"type": "image", "index": 0}],
                },
            ],
            "images": ["image.jpg"],
        }
    ]

    result = detect_vlm_dataset_structure(dataset)

    assert result["format"] == "vlm_messages_llava"
    assert result["image_column"] == "images"


def test_text_only_conversation_is_not_vlm(detect_vlm_dataset_structure):
    dataset = [
        {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        }
    ]

    assert detect_vlm_dataset_structure(dataset)["format"] == "unknown"
