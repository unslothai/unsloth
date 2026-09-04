# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from PIL import Image

from core.inference import mcp_images
from core.inference.mcp_images import (
    IMAGE_TURN_TEXT,
    MAX_IMAGE_EDGE,
    MAX_MODEL_IMAGES,
    append_image_turn,
    content_parts,
    promote_history,
    split_images,
)


def _png(size = (8, 8), fmt = "PNG") -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buffer, format = fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image(data = None, mime = "image/png") -> dict:
    return {"data": data if data is not None else _png(), "mimeType": mime}


def _envelope(text: str, *images: dict) -> str:
    return text + "\n" + mcp_images.SENTINEL + json.dumps(list(images))


def _decode(part: dict):
    from PIL import Image

    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_split_returns_text_and_images():
    text, images = split_images(_envelope("a screenshot", _image()))

    assert text == "a screenshot"
    assert len(images) == 1


def test_split_leaves_a_result_that_only_mentions_the_marker():
    result = "docs say the marker is\n__MCP_IMAGES__: and nothing follows"

    assert split_images(result) == (result, [])


def test_split_leaves_an_envelope_that_is_not_an_image_array():
    result = 'log\n__MCP_IMAGES__:["not", "image", "dicts"]'

    assert split_images(result) == (result, [])


def test_content_parts_reencode_to_png():
    parts = content_parts([_image(data = _png(fmt = "WEBP"), mime = "image/webp")])

    assert len(parts) == 1
    assert _decode(parts[0]).format == "PNG"


def test_content_parts_downscale_a_large_image():
    parts = content_parts([_image(data = _png(size = (MAX_IMAGE_EDGE * 2, MAX_IMAGE_EDGE)))])

    assert max(_decode(parts[0]).size) == MAX_IMAGE_EDGE


def test_content_parts_cap_how_many_reach_the_model():
    parts = content_parts([_image() for _ in range(MAX_MODEL_IMAGES + 3)])

    assert len(parts) == MAX_MODEL_IMAGES


def test_content_parts_drop_an_undecodable_payload():
    assert content_parts([_image(data = "not base64 at all")]) == []
    assert content_parts([_image(data = base64.b64encode(b"nope").decode())]) == []


def test_image_turn_is_its_own_user_message():
    conversation = [{"role": "tool", "name": "mcp__fs__read", "content": "[1 image returned]"}]

    append_image_turn(conversation, [_image()])

    assert conversation[-1]["role"] == "user"
    assert conversation[-1]["content"][0] == {"type": "text", "text": IMAGE_TURN_TEXT}
    assert conversation[-1]["content"][1]["type"] == "image_url"
    assert conversation[1] is conversation[-1]


def test_image_turn_merges_into_a_trailing_user_turn():
    conversation = [{"role": "user", "content": "a nudge"}]

    append_image_turn(conversation, [_image()])

    assert len(conversation) == 1
    assert conversation[0]["content"][0] == {"type": "text", "text": "a nudge"}
    assert conversation[0]["content"][1]["type"] == "image_url"


def test_image_turn_is_skipped_when_nothing_decodes():
    conversation = [{"role": "tool", "content": "[1 image returned]"}]

    append_image_turn(conversation, [_image(data = "///")])

    assert len(conversation) == 1


def test_history_promotes_a_replayed_envelope():
    messages = promote_history(
        [
            {"role": "user", "content": "what is in the file"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0"}]},
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": _envelope("[1 image returned]", _image()),
            },
            {"role": "assistant", "content": "a blue square"},
        ],
        vision = True,
    )

    assert messages[2]["content"] == "[1 image returned]"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"][1]["type"] == "image_url"
    assert messages[4]["content"] == "a blue square"


def test_history_flushes_after_the_whole_batch_of_tool_results():
    messages = promote_history(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": _envelope("first", _image())},
            {"role": "tool", "tool_call_id": "b", "content": _envelope("second", _image())},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "tool", "user"]
    assert len(messages[3]["content"]) == 3


def test_history_strips_the_envelope_for_a_model_that_cannot_see_it():
    messages = promote_history(
        [{"role": "tool", "content": _envelope("[1 image returned]", _image())}],
        vision = False,
    )

    assert messages == [{"role": "tool", "content": "[1 image returned]"}]


def test_history_keeps_an_image_only_result_from_emptying_its_tool_message():
    messages = promote_history(
        [{"role": "tool", "content": _envelope("", _image())}],
        vision = False,
    )

    assert messages[0]["content"] == "[image returned]"


def test_history_leaves_other_messages_untouched():
    original = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "plain result"},
    ]

    assert promote_history(original, vision = True) == original


def test_history_merges_into_a_following_user_turn():
    messages = promote_history(
        [
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "user", "content": "what colour was it"},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["tool", "user"]
    assert messages[1]["content"][0]["type"] == "image_url"
    assert messages[1]["content"][1] == {"type": "text", "text": "what colour was it"}


def test_local_history_carries_markers_and_payloads():
    messages, payloads = mcp_images.promote_history_local(
        [
            {"role": "user", "content": "what is in the file"},
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "assistant", "content": "a blue square"},
        ],
        vision = True,
    )

    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0] == {"type": "image"}
    assert len(payloads) == 1
    assert base64.b64decode(payloads[0])[:8] == b"\x89PNG\r\n\x1a\n"


def test_local_history_merges_markers_into_a_following_user_turn():
    messages, payloads = mcp_images.promote_history_local(
        [
            {"role": "tool", "content": _envelope("[1 image returned]", _image())},
            {"role": "user", "content": "what colour was it"},
        ],
        vision = True,
    )

    assert [message["role"] for message in messages] == ["tool", "user"]
    assert messages[1]["content"][0] == {"type": "image"}
    assert len(payloads) == 1


def test_local_history_strips_without_vision():
    messages, payloads = mcp_images.promote_history_local(
        [{"role": "tool", "content": _envelope("[1 image returned]", _image())}],
        vision = False,
    )

    assert messages == [{"role": "tool", "content": "[1 image returned]"}]
    assert payloads == []


def test_placeholder_turn_marks_one_image_per_payload():
    turn = mcp_images.placeholder_turn(2)

    assert [part["type"] for part in turn["content"]] == ["image", "image", "text"]


def test_an_oversized_raster_is_rejected_off_the_header():
    from PIL import Image

    buffer = io.BytesIO()
    # Uniform colour, so a 9000x9000 raster still encodes to a few kilobytes and
    # clears every payload-size gate ahead of the decode.
    Image.new("RGB", (9000, 9000), (0, 0, 0)).save(buffer, format = "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    from core.inference.mcp_client import MAX_IMAGE_PAYLOAD_CHARS

    assert len(encoded) < MAX_IMAGE_PAYLOAD_CHARS

    assert content_parts([{"data": encoded, "mimeType": "image/png"}]) == []


def test_an_image_inside_the_pixel_budget_still_downscales():
    parts = content_parts([_image(data = _png(size = (2000, 1000)))])

    assert max(_decode(parts[0]).size) == MAX_IMAGE_EDGE


def test_local_history_caps_the_images_a_replay_resends():
    turns = []
    for _ in range(mcp_images.MAX_TOTAL_MODEL_IMAGES + 3):
        turns.append({"role": "tool", "content": _envelope("[1 image returned]", _image())})
        turns.append({"role": "assistant", "content": "noted"})

    messages, payloads = mcp_images.promote_history_local(turns, vision = True)
    markers = sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    )

    assert len(payloads) == mcp_images.MAX_TOTAL_MODEL_IMAGES
    # A marker with no pixels behind it makes the processor count image tokens it
    # was given nothing for, so the two have to come down together.
    assert markers == len(payloads)


def _svg_image() -> dict:
    # _flatten_result accepts image/svg+xml, and Pillow cannot decode it.
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="8" height="8"/></svg>'
    return {"data": base64.b64encode(svg).decode(), "mimeType": "image/svg+xml"}


def test_undecodable_images_do_not_spend_the_quota():
    images = [_svg_image() for _ in range(MAX_MODEL_IMAGES)] + [_image(), _image()]

    parts = content_parts(images)

    assert len(parts) == 2
    assert len(mcp_images.png_payloads(images)) == 2


def test_the_data_url_history_is_capped_too():
    turns = []
    for _ in range(mcp_images.MAX_TOTAL_MODEL_IMAGES + 3):
        turns.append({"role": "tool", "content": _envelope("[1 image returned]", _image())})
        turns.append({"role": "assistant", "content": "noted"})

    messages = promote_history(turns, vision = True)
    parts = mcp_images.count_image_parts(messages, "image_url")

    assert parts == mcp_images.MAX_TOTAL_MODEL_IMAGES


def test_a_live_image_turn_merges_into_a_trailing_nudge():
    """append_deferred_nudges leaves a role=user turn; a second one in a row is
    what a strict VLM template rejects."""
    conversation = [
        {"role": "user", "content": "take a screenshot"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "[1 image returned]"},
        {"role": "user", "content": "One earlier request was not executed."},
    ]

    append_image_turn(conversation, [_image()])

    assert [m["role"] for m in conversation] == ["user", "assistant", "tool", "user"]
    assert conversation[-1]["content"][0] == {
        "type": "text",
        "text": "One earlier request was not executed.",
    }
    assert conversation[-1]["content"][1]["type"] == "image_url"


def test_a_marker_turn_merges_into_a_trailing_nudge():
    conversation = [
        {"role": "tool", "tool_call_id": "c", "content": "[1 image returned]"},
        {"role": "user", "content": "One earlier request was not executed."},
    ]

    mcp_images.append_placeholder_turn(conversation, 2, 2)

    assert [m["role"] for m in conversation] == ["tool", "user"]
    assert [p["type"] for p in conversation[-1]["content"]] == ["text", "image", "image"]


def test_a_marker_turn_still_opens_its_own_turn_after_a_tool_result():
    conversation = [{"role": "tool", "tool_call_id": "c", "content": "[1 image returned]"}]

    mcp_images.append_placeholder_turn(conversation, 1, 1)

    assert [m["role"] for m in conversation] == ["tool", "user"]
    assert conversation[-1]["content"][0] == {"type": "image"}


def test_a_non_mcp_tool_never_has_its_output_read_as_images():
    """The envelope is a plain text suffix, so terminal output or a fetched page
    can end in a syntactically valid one. Only an mcp__ call is trusted."""
    from core.inference.tool_loop_controller import ToolCallCompletion, ToolCallDecision

    payload = _envelope("$ cat notes.txt", _image())

    def completion(tool_name):
        decision = ToolCallDecision(
            action = "execute",
            tool_name = tool_name,
            arguments = {},
            tool_call_id = "call_0",
            card_call_id = "",
            key = tool_name,
            provenance = {},
            noop_result = "",
        )
        return ToolCallCompletion(decision = decision, result = payload, executed = True)

    assert completion("mcp__fs__read_media_file").mcp_images()
    assert completion("bash").mcp_images() == []
    assert completion("web_fetch").mcp_images() == []


def test_image_parts_are_dropped_for_a_text_only_fallback():
    from core.inference.inference import _without_image_parts

    messages = [
        {"role": "user", "content": "what is in the file"},
        {"role": "tool", "content": "[1 image returned]"},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": IMAGE_TURN_TEXT}],
        },
        {"role": "user", "content": [{"type": "image"}]},
    ]

    out = _without_image_parts(messages)

    # And the fallback branch really calls it, not just defines it.
    import inspect

    from core.inference import inference as inference_module

    body = inspect.getsource(inference_module.InferenceBackend._generate_chat_response_inner)
    assert "messages = _without_image_parts(messages)" in body

    assert out[0] == messages[0]
    assert out[1] == messages[1]
    # Collapsed back to a plain string, which is what a text template takes.
    assert out[2]["content"] == IMAGE_TURN_TEXT
    assert out[3]["content"] == ""


def test_a_replay_only_turn_takes_the_vision_render():
    """The reasoning-channel markers follow the vision render, and #10092 made that
    unconditional on the vision path -- so what this PR has to keep true is that a
    replay-only turn reaches that path at all, with no attachment to trigger it."""
    import inspect

    from core.inference import inference as inference_module

    inner = inspect.getsource(inference_module.InferenceBackend._generate_chat_response_inner)
    assert (
        "if is_vision and (image or images):" in inner
    ), "a conversation whose only pictures are replayed still has to render as vision"
    vision = inspect.getsource(inference_module.InferenceBackend._generate_vision_response)
    assert "if attached:" in vision, "and the branch inside keys off every attached image"


def test_a_named_non_mcp_tool_result_is_not_promoted_on_replay():
    """The live loop checks the call's tool name; the replay path reads the
    message's own `name`. Terminal output ending in a valid envelope must not
    become image input on the next request either."""
    history = [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "bash",
            "content": _envelope("$ cat notes.txt", _image()),
        }
    ]

    out = promote_history(history, vision = True)

    # The suffix still comes off -- it is base64 and the model must not read it as
    # text -- but it never becomes image input.
    assert out[0]["content"] == "$ cat notes.txt"
    assert len(out) == 1, "a bash result was promoted into image input"


def test_an_mcp_named_result_is_still_promoted_on_replay():
    history = [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "mcp__fs__read_media_file",
            "content": _envelope("[1 image returned]", _image()),
        }
    ]

    out = promote_history(history, vision = True)

    assert out[0]["content"] == "[1 image returned]"
    assert out[1]["content"][1]["type"] == "image_url"


def test_an_unnamed_tool_result_keeps_working():
    """Older stored turns carry no name; the envelope only ever came from an MCP
    server, so an absent name is not evidence against it."""
    history = [{"role": "tool", "content": _envelope("[1 image returned]", _image())}]

    out = promote_history(history, vision = True)

    assert len(out) == 2
    assert out[1]["content"][1]["type"] == "image_url"


def test_stripping_is_unconditional_and_only_promotion_is_gated():
    """Two different rules, and conflating them breaks one or the other.

    The suffix runs to megabytes of base64, so it comes off every model-facing
    text path whoever produced it -- that is a context-window property, and
    test_tool_result_fits_window depends on it. Provenance answers a separate
    question: whether those bytes are trusted enough to become IMAGE input.
    """
    from core.inference.mcp_images import promote_history
    from core.inference.tool_loop_controller import (
        ToolCallCompletion,
        ToolCallDecision,
        strip_result_for_model,
    )

    payload = _envelope("$ cat notes.txt", _image())

    def completion(tool_name):
        decision = ToolCallDecision(
            action = "execute",
            tool_name = tool_name,
            arguments = {},
            tool_call_id = "call_0",
            card_call_id = "",
            key = tool_name,
            provenance = {},
            noop_result = "",
        )
        return ToolCallCompletion(decision = decision, result = payload, executed = True)

    for name in ("mcp__fs__read", "bash", "web_search", "mcp"):
        # 1. never as text, on the live path or the replay
        assert strip_result_for_model(payload, name) == "$ cat notes.txt", name
        replayed = promote_history(
            [{"role": "tool", "name": name, "content": payload}], vision = True
        )
        assert replayed[0]["content"] == "$ cat notes.txt", name

    # 2. as image input only from a tool an MCP server served
    assert completion("mcp__fs__read").mcp_images()
    assert completion("bash").mcp_images() == []
    promoted = promote_history(
        [{"role": "tool", "name": "mcp__fs__read", "content": payload}], vision = True
    )
    assert any(isinstance(m.get("content"), list) for m in promoted)
    not_promoted = promote_history(
        [{"role": "tool", "name": "bash", "content": payload}], vision = True
    )
    assert not any(isinstance(m.get("content"), list) for m in not_promoted)


def test_a_parallel_batch_keeps_each_result_its_own_quota():
    """Two calls returning four images each deliver eight, not four: flattening
    first hands the whole per-result allowance to the first call."""
    from core.inference.mcp_images import content_parts_per_result

    one = [_image() for _ in range(MAX_MODEL_IMAGES)]
    two = [_image() for _ in range(MAX_MODEL_IMAGES)]

    assert len(content_parts(one + two)) == MAX_MODEL_IMAGES
    assert len(content_parts_per_result([one, two])) == 2 * MAX_MODEL_IMAGES


def test_a_parallel_batch_still_stops_at_the_conversation_cap():
    from core.inference.mcp_images import content_parts_per_result
    results = [[_image() for _ in range(MAX_MODEL_IMAGES)] for _ in range(4)]

    assert len(content_parts_per_result(results)) == mcp_images.MAX_TOTAL_MODEL_IMAGES


def test_an_emptied_image_turn_is_dropped_not_left_blank():
    """content: [] is an empty user message, which strict provider APIs and chat
    templates reject rather than serving the newer images."""
    conversation = [
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "user", "content": [{"type": "image"}]},
    ]
    payloads = ["a", "b"]

    mcp_images.trim_image_turns(conversation, payloads, limit = 1)

    assert len(conversation) == 1
    assert all(message.get("content") != [] for message in conversation)


def test_replayed_markers_sit_ahead_of_the_attachment_marker():
    """Pixels go history-first with the attachment last, and a positional VLM
    processor binds them in document order."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "this one"}]},
    ]

    mcp_images.insert_placeholder_turn(messages, 0, 2, 2)

    replayed = [
        index
        for index, message in enumerate(messages)
        for part in message["content"]
        if part.get("type") == "image"
    ]
    # Both replayed markers come before the attachment's turn.
    assert replayed[:2] == [0, 0]
    assert replayed[-1] == 1


def test_the_attachment_is_marked_on_the_turn_that_supplied_it():
    """_extract_content_parts takes the newest user image from anywhere in the
    thread, so marking the newest TURN tells a text-only latest question that it
    carried an older picture."""
    conversation = [
        {"role": "user", "content": "here is the screenshot"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "now a text-only follow-up"},
    ]

    marked = mcp_images.mark_last_user_turn(conversation, 1, ordinal = 0)

    assert isinstance(marked[0]["content"], list)
    assert marked[2]["content"] == "now a text-only follow-up"


def test_without_an_ordinal_the_newest_user_turn_still_takes_it():
    conversation = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]

    marked = mcp_images.mark_last_user_turn(conversation, 1)

    assert marked[0]["content"] == "first"
    assert isinstance(marked[1]["content"], list)


def test_the_provenance_name_survives_local_extraction():
    """promote_history reads message['name'] to decide whether an envelope came
    from an MCP server; an extraction that drops it bypasses the check."""
    import inspect

    import routes.inference as inference_route

    body = inspect.getsource(inference_route._extract_content_parts)
    assert (
        'chat_message["name"] = msg.name' in body
    ), "the tool message's name has to reach promote_history"


def test_a_parallel_batch_survives_the_replay_too():
    """The live loops kept result boundaries; the replay still flattened them, so
    a conversation resumed after two parallel calls lost the second call's images."""
    history = [
        {
            "role": "tool",
            "name": "mcp__a__shot",
            "content": _envelope("[4 images returned]", *[_image() for _ in range(4)]),
        },
        {
            "role": "tool",
            "name": "mcp__b__shot",
            "content": _envelope("[4 images returned]", *[_image() for _ in range(4)]),
        },
    ]

    promoted = promote_history(history, vision = True)

    assert mcp_images.count_image_parts(promoted, "image_url") == 2 * MAX_MODEL_IMAGES


def test_the_local_replay_keeps_markers_and_payloads_in_step_across_results():
    history = [
        {
            "role": "tool",
            "name": f"mcp__{server}__shot",
            "content": _envelope("[4 images returned]", *[_image() for _ in range(4)]),
        }
        for server in ("a", "b")
    ]

    messages, payloads = mcp_images.promote_history_local(history, vision = True)

    assert len(payloads) == 2 * MAX_MODEL_IMAGES
    assert mcp_images.count_image_parts(messages, "image") == len(payloads)


def test_the_ordinal_skips_the_turns_promotion_inserted():
    """The ordinal is counted on the ORIGINAL history, but applied to the promoted
    one, which has synthetic image turns in it. Counting those would put the
    attachment's marker on a historical picture's turn."""
    promoted = [
        {"role": "user", "content": "here is the shot"},
        mcp_images.placeholder_turn(1, 1),
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and this new one"},
    ]

    out = mcp_images.mark_last_user_turn(promoted, 1, ordinal = 1)

    # The second REAL user turn, not the second user turn in the promoted list.
    assert isinstance(out[3]["content"], list)
    assert out[3]["content"][-1] == {"type": "text", "text": "and this new one"}


def test_a_synthetic_image_turn_is_recognised():
    assert mcp_images.is_synthetic_image_turn(mcp_images.placeholder_turn(2, 2))
    assert not mcp_images.is_synthetic_image_turn({"role": "user", "content": "hello"})
    assert not mcp_images.is_synthetic_image_turn(
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    )


def test_undecodable_entries_are_not_counted_as_images():
    """_flatten_result accepts formats Pillow cannot read, and promotion drops
    them, so charging their KV reserves cache for images never sent."""
    svg = {
        "data": base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode(),
        "mimeType": "image/svg+xml",
    }

    assert mcp_images.count_probably_decodable([svg] * 4) == 0
    assert mcp_images.count_probably_decodable([_image()] * 3) == 3
    assert mcp_images.count_probably_decodable([svg, _image(), svg, _image()]) == 2
    # And the sniff never claims something it cannot open.
    assert not mcp_images.probably_decodable({"data": "not base64 at all!!", "mimeType": "x"})
    assert not mcp_images.probably_decodable({})


def test_the_replay_cap_never_deletes_a_callers_own_attachments():
    """promote_history runs on every request now, so an unscoped cap silently
    dropped the oldest attachments of a caller that simply sent many images."""
    attachments = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png()}"}}
        for _ in range(mcp_images.MAX_TOTAL_MODEL_IMAGES + 4)
    ]
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "compare these"}, *attachments]}
    ]

    out = promote_history(conversation, vision = True)

    assert mcp_images.count_image_parts(out, "image_url") == len(attachments)


def test_the_replay_cap_still_bounds_what_promotion_adds():
    turns = []
    for _ in range(mcp_images.MAX_TOTAL_MODEL_IMAGES + 3):
        turns.append({"role": "tool", "name": "mcp__a__s", "content": _envelope("[1]", _image())})
        turns.append({"role": "assistant", "content": "noted"})

    out = promote_history(turns, vision = True)

    assert mcp_images.count_image_parts(out, "image_url") == mcp_images.MAX_TOTAL_MODEL_IMAGES


def test_the_sniff_charges_every_format_promotion_can_decode():
    """A deny list, not an allow list: a format Pillow gains that an allow list
    missed would be promoted and reserved nothing for, which overcommits the KV
    cache. Guessing the other way only over-reserves."""
    for fmt in ("PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP", "ICO"):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format = fmt)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        assert mcp_images.probably_decodable({"data": encoded}), fmt

    svg = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode()
    assert not mcp_images.probably_decodable({"data": svg})
    assert not mcp_images.probably_decodable({"data": base64.b64encode(b'{"a": 1}').decode()})


def test_the_live_cap_also_leaves_callers_attachments_alone():
    """The replay cap was scoped first; the live loops kept the unscoped one, so a
    tool loop that starts from caller attachments still deleted them."""
    attachments = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png()}"}}
        for _ in range(6)
    ]
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "compare"}, *attachments]}
    ]
    owned: list = []
    for _ in range(3):
        conversation.append({"role": "tool", "content": "[4 images returned]"})
        append_image_turn(
            conversation, [[_image() for _ in range(4)]], per_result = True, owned = owned
        )

    survived = sum(
        1
        for part in attachments
        for message in conversation
        if isinstance(message.get("content"), list)
        for other in message["content"]
        if other is part
    )
    assert survived == len(attachments), "a caller attachment was deleted by the loop's cap"
    assert len(owned) == mcp_images.MAX_TOTAL_MODEL_IMAGES


def test_pixels_follow_the_markers_when_the_attachment_came_first():
    """ "History first, attachment last" only holds when the attachment is on the
    newest turn. It is not when an earlier question carried it and a tool returned
    pictures afterwards, and a positional VLM binds the Nth pixel to the Nth marker."""
    conversation = [
        {"role": "user", "content": "look at this"},
        {"role": "tool", "content": "[1 image returned]"},
        mcp_images.placeholder_turn(1, 1),
    ]
    prior = mcp_images.image_marker_parts(conversation)
    conversation = mcp_images.mark_last_user_turn(conversation, 1, ordinal = 0)

    ordered = mcp_images.pixels_in_marker_order(conversation, prior, ["MCP"], "ATTACH")

    assert ordered == ["ATTACH", "MCP"]


def test_pixels_stay_history_first_when_the_attachment_is_newest():
    conversation = [
        {"role": "tool", "content": "[1 image returned]"},
        mcp_images.placeholder_turn(1, 1),
        {"role": "user", "content": "and this one"},
    ]
    prior = mcp_images.image_marker_parts(conversation)
    conversation = mcp_images.mark_last_user_turn(conversation, 1, ordinal = 1)

    ordered = mcp_images.pixels_in_marker_order(conversation, prior, ["MCP"], "ATTACH")

    assert ordered == ["MCP", "ATTACH"]


def test_the_vision_streamer_resolves_markers_for_a_replay_only_turn():
    """The streamer marks its markers resolved either way, so gating them on the
    singular attachment resolves a replay-only turn to none AND suppresses the
    fallback detection -- native reasoning output then reaches the user as answer
    text. Route classification keys off every image; generation has to as well."""
    import inspect

    from core.inference import inference as inference_module

    body = inspect.getsource(inference_module.InferenceBackend._generate_vision_response)
    call = body.index("detect_reasoning_channel_markers(processor")
    window = body[call : call + 400]
    assert "if attached" in window, window[:200]
    assert "if image\n" not in window


def test_the_gguf_loop_caps_images_across_iterations_not_per_batch():
    """A per-iteration ownership list only ever sees the current batch, so three
    sequential four-image results leave twelve images in the conversation against
    a cap of eight."""
    import inspect

    from core.inference import llama_cpp

    body = inspect.getsource(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    declared = body.index("loop_mcp_image_parts: list = list(replayed_image_parts)")
    loop_start = body.index("while True:")
    assert declared < loop_start, (
        "the ownership list has to outlive the iteration, or the cap only ever "
        "counts the newest batch"
    )


def test_promotion_reports_the_parts_it_created():
    """A resumed chat already carries promoted images; a loop starting its cap at
    zero lets the whole history through and then adds a batch on top."""
    history = [
        {"role": "tool", "name": "mcp__a__s", "content": _envelope("[1]", _image())},
        {"role": "assistant", "content": "noted"},
    ]
    promoted: list = []

    out = promote_history(history, vision = True, promoted_out = promoted)

    assert len(promoted) == 1
    live = [
        part
        for message in out
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert all(any(part is other for other in live) for part in promoted)


def test_the_plain_vision_path_marks_the_turn_that_supplied_the_attachment():
    """messages_with_attached_image returns early when markers exist, so the top-up
    placed the attachment's marker on the newest turn -- showing a later text-only
    question as carrying an older image."""
    conversation = [
        {"role": "user", "content": "here it is"},
        {"role": "assistant", "content": "ok"},
        mcp_images.placeholder_turn(1, 1),
        {"role": "user", "content": "a later text question"},
    ]
    prior = mcp_images.image_marker_parts(conversation)

    out = mcp_images.top_up_image_markers(conversation, 2, ordinal = 0)

    assert isinstance(out[0]["content"], list)
    assert out[3]["content"] == "a later text question"
    assert mcp_images.pixels_in_marker_order(out, prior, ["MCP"], "ATTACH") == ["ATTACH", "MCP"]


def test_the_attachment_is_not_the_one_the_cap_drops():
    """Trimming the combined list removes whatever is first, and the attachment is
    first whenever its turn precedes the replayed pictures."""
    conversation = [mcp_images.placeholder_turn(1, 1) for _ in range(8)]
    payloads = [f"mcp{i}" for i in range(8)]

    # what the route now does: trim the replay to leave room, THEN interleave
    mcp_images.trim_image_turns(conversation, payloads, limit = mcp_images.MAX_TOTAL_MODEL_IMAGES - 1)

    assert len(payloads) == mcp_images.MAX_TOTAL_MODEL_IMAGES - 1
    assert payloads[-1] == "mcp7", "the newest replayed image survived"
    assert "mcp0" not in payloads, "the oldest replayed image went, not the attachment"


def test_a_client_marked_attachment_still_gets_its_pixel():
    """A marker that predates the top-up is history's only while history has
    pixels left. When the client marked the attachment itself, the top-up adds
    nothing and that pre-existing marker is the attachment's."""
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
    prior = mcp_images.image_marker_parts(conversation)

    ordered = mcp_images.pixels_in_marker_order(conversation, prior, [], "ATTACH")

    assert ordered == ["ATTACH"], "the attachment's pixel was dropped"


def test_history_still_takes_its_own_markers_first():
    conversation = [
        mcp_images.placeholder_turn(2, 2),
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "and this"}]},
    ]
    prior = mcp_images.image_marker_parts(conversation)

    ordered = mcp_images.pixels_in_marker_order(conversation, prior, ["A", "B"], "ATTACH")

    assert ordered == ["A", "B", "ATTACH"]


def test_the_gguf_loop_starts_its_cap_from_the_replayed_parts():
    """The list is seeded from the conversation, not from zero: a resumed chat whose
    history already carries the allowance would otherwise get a second one for this
    run and send both."""
    import inspect

    from core.inference import llama_cpp

    signature = inspect.signature(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    assert "replayed_image_parts" in signature.parameters

    body = inspect.getsource(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    assert "loop_mcp_image_parts: list = list(replayed_image_parts)" in body


def test_the_gguf_route_hands_the_loop_what_promotion_created():
    import inspect

    from routes import inference

    builder = inspect.signature(inference._openai_messages_for_gguf_chat)
    assert "promoted_out" in builder.parameters

    route = inspect.getsource(inference.produce_openai_chat_completions)
    assert "_gguf_replayed_image_parts: list = []" in route
    assert "replayed_image_parts = tuple(_gguf_replayed_image_parts)" in route
