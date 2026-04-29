"""Verify dpo_trainer_vision_process_row forwards prompt and images verbatim."""
import ast
import os

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _load_helpers():
    src = open(RL_PATH).read()
    tree = ast.parse(src)
    import torch as _torch

    ns = {"torch": _torch}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_DPO_VISION_KEYS"
            for t in node.targets
        ):
            exec(ast.get_source_segment(src, node), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            ("dpo_trainer_", "_dpo_trainer_")
        ):
            exec(ast.get_source_segment(src, node), ns)
    return ns


class _Tok:
    eos_token_id = 99
    bos_token_id = None

    def __call__(self, t, add_special_tokens=False):
        return {"input_ids": [10]}


class _Capture:
    image_token = "<img>"
    boi_token = "<boi>"

    def __init__(self):
        self.tokenizer = _Tok()
        self.last_text = None
        self.last_images = "__sentinel__"

    def __call__(self, images=None, text=None, add_special_tokens=False):
        self.last_text = text
        self.last_images = images
        out = {"input_ids": [[1, 2]]}
        if images is not None:
            out["pixel_values"] = [object()]
        return out


def test_prompt_passes_through_without_image_token_synthesis():
    ns = _load_helpers()
    proc = _Capture()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "describe", "chosen": "c", "rejected": "r", "images": ["i"]},
        proc,
    )
    assert proc.last_text == "describe"


def test_prompt_with_existing_image_token_unchanged():
    ns = _load_helpers()
    proc = _Capture()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "<img> describe", "chosen": "c", "rejected": "r", "images": ["i"]},
        proc,
    )
    assert proc.last_text == "<img> describe"


def test_gemma3_style_boi_token_prompt_not_corrupted():
    ns = _load_helpers()
    proc = _Capture()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "<boi> describe", "chosen": "c", "rejected": "r", "images": ["i"]},
        proc,
    )
    assert proc.last_text == "<boi> describe"
    assert "<img>" not in proc.last_text


def test_multi_image_prompt_unchanged_no_extra_placeholders():
    ns = _load_helpers()
    proc = _Capture()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "compare", "chosen": "c", "rejected": "r", "images": ["a", "b", "c"]},
        proc,
    )
    assert proc.last_text == "compare"


def test_list_images_forwarded_verbatim():
    ns = _load_helpers()
    proc = _Capture()
    payload = ["a", "b"]
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "p", "chosen": "c", "rejected": "r", "images": payload},
        proc,
    )
    assert proc.last_images is payload


def test_single_pil_like_image_forwarded_verbatim():
    ns = _load_helpers()

    class PIL:
        def __bool__(self):
            return True

    proc = _Capture()
    pil = PIL()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "p", "chosen": "c", "rejected": "r", "images": pil},
        proc,
    )
    assert proc.last_images is pil


def test_numpy_ndarray_image_forwarded_verbatim():
    ns = _load_helpers()
    proc = _Capture()
    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "p", "chosen": "c", "rejected": "r", "images": arr},
        proc,
    )
    assert proc.last_images is arr


def test_missing_images_key_passes_none_to_processor():
    ns = _load_helpers()
    proc = _Capture()
    ns["dpo_trainer_vision_process_row"](
        {"prompt": "p", "chosen": "c", "rejected": "r"},
        proc,
    )
    assert proc.last_images is None
