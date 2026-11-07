# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A video column must reach the vision collator, not train as text with the clip dropped."""

import pytest
from datasets import Dataset

from hub.utils.dataset_format import check_dataset_format
from utils.datasets import format_and_template_dataset

QUESTION = "What colour is the ball and which way does it move?"
ANSWERS = ["red moves left", "blue moves right"]
CLIPS = ["clips/clip00.mp4", "clips/clip01.mp4"]


def _messages_rows():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip},
                            {"type": "text", "text": QUESTION},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
            }
            for clip, answer in zip(CLIPS, ANSWERS)
        ]
    )


def _flat_rows():
    return Dataset.from_dict({"video": CLIPS, "question": [QUESTION] * 2, "answer": ANSWERS})


def _format(ds):
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
    )
    assert info["success"], info["errors"]
    assert info["final_format"] == "vlm_messages"
    return list(info["dataset"])


def _video_parts(row):
    return [
        part
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "video"
    ]


@pytest.mark.parametrize("rows", [_messages_rows, _flat_rows], ids = ["messages", "flat"])
def test_check_format_routes_video_to_the_vision_path(rows):
    result = check_dataset_format(rows(), is_vlm = True)
    # is_image is what the UI sends back as is_dataset_image, which selects the VLM trainer.
    assert result["is_image"] is True
    assert result["requires_manual_mapping"] is False
    assert result["detected_format"] in ("vlm_messages", "video_text")


def test_messages_rows_keep_their_video_parts():
    out = _format(_messages_rows())
    assert [_video_parts(row) for row in out] == [
        [{"type": "video", "video": clip}] for clip in CLIPS
    ]


def test_flat_video_rows_become_video_messages():
    out = _format(_flat_rows())
    for row, clip, answer in zip(out, CLIPS, ANSWERS):
        user, assistant = row["messages"]
        assert user["content"] == [
            {"type": "video", "video": clip},
            {"type": "text", "text": QUESTION},
        ]
        assert assistant["content"] == [{"type": "text", "text": answer}]


@pytest.mark.parametrize(
    "clip", ["clips/a.webm", {"bytes": None, "path": "clips/a.webm"}], ids = ["str", "dict"]
)
def test_webm_in_a_video_column_is_video_not_audio(clip):
    ds = Dataset.from_list([{"video": clip, "answer": "red moves left"}] * 2)
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_image"] is True
    assert result["is_audio"] is False
    assert result["detected_format"] == "video_text"


def test_audio_in_a_column_named_video_stays_audio():
    ds = Dataset.from_list([{"video": {"bytes": None, "path": "a.wav"}, "text": "hello"}] * 2)
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_audio"] is True
    assert "video" not in (result["multimodal_columns"] or [])


def test_text_columns_named_like_video_stay_text():
    ds = Dataset.from_dict(
        {"video_id": ["abc", "def"], "video title": ["Cooking", "Hiking"], "text": ["a", "b"]}
    )
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_image"] is False


def _mlx_adapter():
    # worker.py pulls the whole CUDA/MLX import chain; the adapter itself is self-contained.
    import ast
    from pathlib import Path

    source = (Path(__file__).parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_adapt_for_mlx_vlm"
    )
    namespace = {"_resize_mlx_vlm_image": lambda image, *a, **k: image}
    exec(compile(ast.Module([node], []), "worker.py", "exec"), namespace)
    return namespace["_adapt_for_mlx_vlm"]


def test_mlx_refuses_video_rather_than_dropping_it():
    with pytest.raises(ValueError, match = "video"):
        _mlx_adapter()(_format(_messages_rows()))


@pytest.mark.parametrize("url", ["https://example.com/clip.mp4", "HTTP://example.com/clip.mov?x=1"])
@pytest.mark.parametrize("shape", ["flat", "messages"])
def test_remote_video_urls_are_refused_with_a_clear_message(url, shape):
    # Never handed to the collator to fetch unguarded, nor read as a local path for a managed account.
    if shape == "flat":
        ds = Dataset.from_dict({"video": [url] * 2, "answer": ANSWERS})
    else:
        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "video", "video": url}]},
                        {"role": "assistant", "content": [{"type": "text", "text": a}]},
                    ]
                }
                for a in ANSWERS
            ]
        )
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
    )
    assert not info["success"]
    assert "Remote video URLs are not fetched" in " ".join(map(str, info["errors"]))


def test_completion_is_a_video_answer_column():
    ds = Dataset.from_dict({"video": CLIPS, "completion": ANSWERS})
    assert check_dataset_format(ds, is_vlm = True)["detected_format"] == "video_text"
    for row, answer in zip(_format(ds), ANSWERS):
        assert row["messages"][1]["content"] == [{"type": "text", "text": answer}]


def test_a_video_column_with_no_answer_column_is_not_reported_as_formatted():
    ds = Dataset.from_dict({"video": CLIPS, "notes_id": ["1", "2"]})
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
    )
    assert not info["success"]
    assert info["requires_manual_mapping"] is True
    assert "video" in " ".join(map(str, info["errors"]))


def test_a_null_first_video_row_still_reads_as_video():
    ds = Dataset.from_dict({"video": [None, *CLIPS], "answer": [None, *ANSWERS]})
    result = check_dataset_format(ds, is_vlm = True)
    assert result["is_image"] is True
    assert result["detected_format"] == "video_text"
    out = _format(ds)
    assert [_video_parts(row) for row in out] == [[{"type": "video", "video": c}] for c in CLIPS]


def test_filling_a_blank_first_row_does_not_decode_other_columns():
    import io

    from datasets import Image as HFImage
    from PIL import Image

    from utils.datasets.format_detection import detect_video_columns, video_sample

    png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(png, "PNG")
    ds = Dataset.from_dict(
        {
            "image": [
                {"bytes": png.getvalue(), "path": None},
                {"bytes": b"not an image", "path": None},
            ],
            "clip": [None, CLIPS[0]],
        }
    ).cast_column("image", HFImage())
    assert detect_video_columns(video_sample(ds, ds[0])) == ["clip"]


def test_a_blank_first_answer_still_finds_the_answer_column():
    ds = Dataset.from_dict({"video": CLIPS, "answer": ["  ", ANSWERS[1]]})
    assert check_dataset_format(ds, is_vlm = True)["detected_format"] == "video_text"
    out = _format(ds)
    assert len(out) == 1 and out[0]["messages"][1]["content"] == [
        {"type": "text", "text": ANSWERS[1]}
    ]


def test_every_video_column_reaches_the_message():
    ds = Dataset.from_dict(
        {"video_front": CLIPS, "video_side": ["clips/side0.mp4", None], "answer": ANSWERS}
    )
    out = _format(ds)
    assert _video_parts(out[0]) == [
        {"type": "video", "video": CLIPS[0]},
        {"type": "video", "video": "clips/side0.mp4"},
    ]
    assert _video_parts(out[1]) == [{"type": "video", "video": CLIPS[1]}]


class _DecodedVideo:
    """What a datasets `Video` feature yields with decoding on: a decoder carrying its source."""

    def __init__(self, path):
        self._hf_encoded = {"path": path, "bytes": None}


def test_decoded_hf_video_cells_are_video():
    from utils.datasets.format_conversion import convert_video_to_vlm_format
    from utils.datasets.format_detection import detect_video_structure

    rows = [{"clip": _DecodedVideo(c), "answer": a} for c, a in zip(CLIPS, ANSWERS)]
    structure = detect_video_structure(rows[0])
    assert structure["format"] == "video_text" and structure["video_column"] == "clip"
    out = convert_video_to_vlm_format(
        rows, "clip", "answer", video_columns = structure["video_columns"]
    )
    assert [_video_parts(row) for row in out] == [[{"type": "video", "video": c}] for c in CLIPS]


def test_a_numeric_answer_column_is_an_answer():
    ds = Dataset.from_dict({"video": CLIPS, "label": [0, 1]})
    assert check_dataset_format(ds, is_vlm = True)["detected_format"] == "video_text"
    out = _format(ds)
    assert [row["messages"][1]["content"][0]["text"] for row in out] == ["0", "1"]


def test_rows_mixing_video_and_image_columns_are_refused():
    from PIL import Image

    image = Image.new("RGB", (4, 4))
    ds = Dataset.from_dict({"video": CLIPS, "image": [image, image], "answer": ANSWERS})
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
    )
    assert not info["success"]
    assert "image" in " ".join(map(str, info["errors"]))


def test_a_text_column_named_image_is_not_an_image():
    ds = Dataset.from_dict({"video": CLIPS, "image_caption": ["a", "b"], "answer": ANSWERS})
    assert len(_format(ds)) == 2


def test_byte_backed_hf_video_is_refused_with_a_clear_message():
    from utils.datasets.format_conversion import convert_video_to_vlm_format
    from utils.datasets.format_detection import detect_video_structure

    clip = _DecodedVideo(None)
    clip._hf_encoded["bytes"] = b"\x00\x00\x00\x18ftypmp42"
    row = {"clip": clip, "answer": "red"}
    structure = detect_video_structure(row)
    assert structure["video_column"] == "clip"
    with pytest.raises(ValueError, match = "embedded bytes"):
        convert_video_to_vlm_format([row], "clip", "answer")


def test_an_image_past_the_detection_probe_is_still_refused():
    from PIL import Image

    from utils.datasets.format_conversion import convert_video_to_vlm_format

    rows = [{"video": CLIPS[0], "image": None, "answer": "a"}] * 40
    rows = rows + [{"video": CLIPS[1], "image": Image.new("RGB", (4, 4)), "answer": "b"}]
    with pytest.raises(ValueError, match = "image"):
        convert_video_to_vlm_format(rows, "video", "answer")


def test_hub_relative_clips_resolve_from_the_dataset_repo(monkeypatch, tmp_path):
    import huggingface_hub

    from utils.datasets.format_conversion import convert_video_to_vlm_format, own_message_videos

    calls, revisions = [], []

    class Api:
        def list_repo_files(
            self,
            repo_id,
            repo_type = None,
            revision = None,
        ):
            assert (repo_id, repo_type) == ("someone/clips", "dataset")
            revisions.append(("list", revision))
            return ["videos/clip00.mp4", "videos/clip01.mp4", "README.md"]

    def download(
        repo_id,
        filename,
        repo_type = None,
        revision = None,
        cache_dir = None,
        **kw,
    ):
        calls.append(filename)
        revisions.append(("download", revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    rows = [{"video": f"videos/clip0{i}.mp4", "answer": a} for i, a in enumerate(ANSWERS)]
    out = convert_video_to_vlm_format(rows, "video", "answer", dataset_name = "someone/clips")
    assert [_video_parts(r)[0]["video"] for r in out] == [
        str(tmp_path / f"videos/clip0{i}.mp4") for i in range(2)
    ]
    assert {revision for _, revision in revisions} == {None}
    messages = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": "clip01.mp4"}]}]}
    ]
    out = own_message_videos(messages, dataset_name = "someone/clips")
    assert _video_parts(out[0])[0]["video"] == str(tmp_path / "videos/clip01.mp4")

    # A pinned dataset revision lists and downloads clips from that revision, not main.
    revisions.clear()
    convert_video_to_vlm_format(
        rows, "video", "answer", dataset_name = "someone/clips", dataset_revision = "abc123"
    )
    own_message_videos(messages, dataset_name = "someone/clips", dataset_revision = "abc123")
    assert revisions and {revision for _, revision in revisions} == {"abc123"}
    assert {kind for kind, _ in revisions} == {"list", "download"}


def test_the_dataset_revision_reaches_the_clip_resolver_from_every_trainer():
    import ast
    from pathlib import Path

    backend = Path(__file__).parents[1]
    for path in ("core/training/trainer.py", "core/training/worker.py"):
        tree = ast.parse((backend / path).read_text(encoding = "utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "format_and_template_dataset"
            # Clips are resolved only on the VLM path; a text-only call never reaches them.
            and any(
                kw.arg == "is_vlm"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
                for kw in node.keywords
            )
        ]
        assert calls, path
        missing = [
            c.lineno for c in calls if not any(kw.arg == "dataset_revision" for kw in c.keywords)
        ]
        assert (
            not missing
        ), f"{path}: format_and_template_dataset without dataset_revision at {missing}"


def test_a_lone_column_of_clip_paths_is_not_raw_text():
    from utils.datasets.format_detection import is_plain_text_dataset
    assert not is_plain_text_dataset(Dataset.from_dict({"clip": CLIPS}))
    assert is_plain_text_dataset(Dataset.from_dict({"text": ["a story", "another"]}))


def test_an_ambiguous_bare_clip_name_is_refused(monkeypatch, tmp_path):
    import huggingface_hub

    from utils.datasets.format_conversion import convert_video_to_vlm_format

    class Api:
        def list_repo_files(
            self,
            repo_id,
            repo_type = None,
            revision = None,
        ):
            return ["train/clip.mp4", "test/clip.mp4"]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(tmp_path / "x.mp4"))
    with pytest.raises(ValueError, match = "train/clip.mp4"):
        convert_video_to_vlm_format(
            [{"video": "clip.mp4", "answer": "a"}], "video", "answer", dataset_name = "someone/clips"
        )
    out = convert_video_to_vlm_format(
        [{"video": "test/clip.mp4", "answer": "a"}], "video", "answer", dataset_name = "someone/clips"
    )
    assert _video_parts(out[0])[0]["video"] == str(tmp_path / "x.mp4")


@pytest.mark.parametrize("repo_files", [["clip.mp4", "sub/clip.mp4"], ["sub/clip.mp4", "clip.mp4"]])
def test_a_root_level_clip_is_its_own_repo_path(monkeypatch, tmp_path, repo_files):
    import huggingface_hub

    from utils.datasets.format_conversion import convert_video_to_vlm_format

    class Api:
        def list_repo_files(
            self,
            repo_id,
            repo_type = None,
            revision = None,
        ):
            return repo_files

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, filename, **k: str(tmp_path / filename)
    )
    # "clip.mp4" is the root file's repo-relative path, so it is exact whatever the listing order.
    for name in ("clip.mp4", "sub/clip.mp4"):
        out = convert_video_to_vlm_format(
            [{"video": name, "answer": "a"}], "video", "answer", dataset_name = "someone/clips"
        )
        assert _video_parts(out[0])[0]["video"] == str(tmp_path / name)


def test_a_classlabel_answer_trains_on_the_class_name():
    from datasets import ClassLabel, Features, Value

    features = Features({"video": Value("string"), "label": ClassLabel(names = ["cat", "dog"])})
    ds = Dataset.from_dict({"video": CLIPS, "label": [1, 0]}, features = features)
    out = _format(ds)
    assert [row["messages"][1]["content"][0]["text"] for row in out] == ["dog", "cat"]


def test_a_manual_video_mapping_uses_the_video_converter():
    ds = Dataset.from_dict({"video": CLIPS, "ask": [QUESTION] * 2, "target": ANSWERS})
    info = format_and_template_dataset(
        ds,
        model_name = "unsloth/Qwen2.5-VL-3B-Instruct",
        tokenizer = None,
        is_vlm = True,
        dataset_name = "local",
        custom_format_mapping = {"video": "image", "target": "assistant", "ask": "user"},
    )
    assert info["success"], info["errors"]
    for row, clip, answer in zip(info["dataset"], CLIPS, ANSWERS):
        user, assistant = row["messages"]
        assert user["content"] == [
            {"type": "video", "video": clip},
            {"type": "text", "text": QUESTION},
        ]
        assert assistant["content"] == [{"type": "text", "text": answer}]
