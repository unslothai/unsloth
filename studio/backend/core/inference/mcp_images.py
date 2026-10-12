# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
import io
import json
import re
from typing import Any, Sequence

from loggers import get_logger

logger = get_logger(__name__)

SENTINEL = "__MCP_IMAGES__:"
# Provenance the envelope is trusted on (live and replay).
MCP_TOOL_PREFIX = "mcp__"
IMAGE_TURN_TEXT = "Images returned by the tool call above:"
DETACHED_IMAGE_TURN_TEXT = "Images returned by earlier tool calls in this conversation:"

MAX_MODEL_IMAGES = 4
MAX_TOTAL_MODEL_IMAGES = 8
# Safetensors and MLX allow one image per message; GGUF is exempt.
LOCAL_MAX_IMAGES_PER_TURN = 1
# Mirrors DECODE_FAILURE_ALLOWANCE in studio/frontend/src/features/chat/api/mcp-images.ts.
DECODE_FAILURE_ALLOWANCE = 4
# Mirrors MAX_MCP_IMAGE_MIME_CHARS in studio/frontend/src/features/chat/api/mcp-images.ts.
MAX_MCP_IMAGE_MIME_CHARS = 256
MAX_IMAGE_EDGE = 1024
# Bounded off the header: a small PNG can hold tens of gigapixels.
MAX_IMAGE_PIXELS = 40_000_000


def is_image_tool(name: str) -> bool:
    return name in ("view_image", "browser_screenshot") or name.startswith(MCP_TOOL_PREFIX)


# The tool loop's own notes, which results saved before #11367 carry after the array (#11358).
_LOOP_NOTE = re.compile(
    r"\[(?:No room in the window for this result, so "
    r"|[^\n\]]+ returned exactly this \d+ times; it will not change)[^\n]*\]"
)


def split_images(result: str) -> tuple[str, list[dict]]:
    """Validated, so tool text that merely mentions the marker is not truncated."""
    end = len(result)
    # At most the starved note and then the repeat note; peeled first so the array parses once.
    for _ in range(2):
        if not result.endswith("]", 0, end):
            break
        cut = result.rfind("\n\n[", max(0, end - 512), end)
        if cut == -1 or not _LOOP_NOTE.fullmatch(result, cut + 2, end):
            break
        end = cut
    if end == len(result):
        return _split_images(result)
    text, images = _split_images(result[:end])
    return (text + result[end:], images) if images else (result, [])


def _split_images(result: str) -> tuple[str, list[dict]]:
    head, sep, payload = result.rpartition("\n" + SENTINEL)
    if not sep:
        return result, []
    try:
        images = json.loads(payload)
    except (ValueError, RecursionError):
        return result, []
    if not isinstance(images, list) or not images:
        return result, []
    if not all(_is_image(image) for image in images):
        return result, []
    return head.rstrip(), images


def _is_image(image: Any) -> bool:
    return (
        isinstance(image, dict)
        and isinstance(image.get("data"), str)
        and isinstance(image.get("mimeType"), str)
    )


# Denylist, not allowlist: false positives only over-reserve KV.
_UNDECODABLE_PREFIXES = (
    b"<svg",
    b"<?xml",
    b"<!DOCTYPE",
    b"<html",
    b"{",
    b"[",
)


def _mime_is_bounded(image: Any) -> bool:
    """Metadata is not where megabytes may hide from the byte budgets on either side;
    an entry whose mimeType runs past the bound is not a picture this path sends."""
    mime = image.get("mimeType") if isinstance(image, dict) else None
    return not (isinstance(mime, str) and len(mime) > MAX_MCP_IMAGE_MIME_CHARS)


def probably_decodable(image: Any) -> bool:
    """Header sniff only, to avoid decoding rasters per request; True unless plainly not an image."""
    data = image.get("data") if isinstance(image, dict) else None
    if not isinstance(data, str) or not data:
        return False
    if not _mime_is_bounded(image):
        return False
    try:
        head = base64.b64decode(data[:32], validate = False)
    except (binascii.Error, ValueError, TypeError):
        return False
    if not head:
        return False
    return not head.lstrip()[:16].startswith(_UNDECODABLE_PREFIXES)


def count_probably_decodable(images: Sequence[dict]) -> int:
    return sum(1 for image in images if probably_decodable(image))


def text_before_envelope(result: str) -> str:
    """Display only: no json parse, which a 12 MB envelope cannot afford on the event loop."""
    index = result.rfind("\n" + SENTINEL)
    return result if index == -1 else result[:index]


def has_images(result: str) -> bool:
    return bool(split_images(result)[1])


def mentions_images(result: str) -> bool:
    """Substring test only: a false positive costs a thread hop, not an event-loop parse of the array."""
    return ("\n" + SENTINEL) in result


def _decoded_urls(
    images: Sequence[dict],
    limit: int = MAX_MODEL_IMAGES,
    attempts: "list | None" = None,
    cache: "dict | None" = None,
) -> list[str]:
    """Successful decodes are capped, but attempts are capped too and shared across a parallel batch."""
    urls = []
    own = [limit + DECODE_FAILURE_ALLOWANCE] if attempts is None else attempts
    for image in images:
        if len(urls) >= limit or own[0] <= 0:
            break
        own[0] -= 1
        if not _mime_is_bounded(image):
            continue
        data = image.get("data", "")
        if cache is not None and data in cache:
            url = cache[data]
        else:
            url = _png_data_url(data)
            if cache is not None:
                cache[data] = url
        if url:
            urls.append(url)
    return urls


def _decoded_urls_per_result(results: Sequence[Sequence[dict]]) -> list[str]:
    """Quota per result, not per concatenated batch, so one call cannot take it all; newest filled first."""
    chosen: list[list[str]] = []
    room = MAX_TOTAL_MODEL_IMAGES
    # One attempt budget per batch, so failing decodes cannot multiply the allowance.
    attempts = [MAX_TOTAL_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE]
    for images in reversed(results):
        if room <= 0 or attempts[0] <= 0:
            break
        urls = _decoded_urls(images, min(MAX_MODEL_IMAGES, room), attempts = attempts)
        room -= len(urls)
        chosen.append(urls)
    return [url for urls in reversed(chosen) for url in urls]


def eligible_replay_images(
    messages: Sequence[dict],
    *,
    local: bool = False,
    budget: int = MAX_TOTAL_MODEL_IMAGES,
) -> dict:
    """Chooses survivors before any decode, so Pillow work follows the cap, not the history."""
    eligible: dict = {}
    spare = DECODE_FAILURE_ALLOWANCE if budget > 0 else 0
    per_result = LOCAL_MAX_IMAGES_PER_TURN if local else MAX_MODEL_IMAGES
    call_names = resolve_tool_names(messages)

    def _is_tool(position: int) -> bool:
        message = messages[position]
        return isinstance(message, dict) and message.get("role") == "tool"

    def _mcp_images_at(position: int) -> "list | None":
        content = messages[position].get("content")
        if not isinstance(content, str):
            return None
        name = messages[position].get("name") or call_names.get(position)
        if isinstance(name, str) and name and not is_image_tool(name):
            return None
        _text, images = split_images(content)
        return images or None

    index = len(messages) - 1
    while index >= 0:
        if not _is_tool(index):
            index -= 1
            continue
        start = index
        if local:
            while start > 0 and _is_tool(start - 1):
                start -= 1
        batch = [position for position in range(index, start - 1, -1)]
        index = start - 1
        room = min(budget, per_result)
        allowance = room + spare
        taken = 0
        for position in batch:
            images = _mcp_images_at(position)
            if images is None:
                continue
            take = min(len(images), allowance)
            if take <= 0:
                eligible[position] = 0
                continue
            eligible[position] = take
            allowance -= take
            taken += take
        charged = min(taken, room)
        budget -= charged
        spare -= taken - charged
    return eligible


def content_parts(images: Sequence[dict]) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": url}} for url in _decoded_urls(images)]


def content_parts_per_result(results: Sequence[Sequence[dict]]) -> list[dict]:
    """content_parts for a batch, keeping each result's own quota."""
    return [
        {"type": "image_url", "image_url": {"url": url}}
        for url in _decoded_urls_per_result(results)
    ]


def png_payloads_per_result(
    results: Sequence[Sequence[dict]], cache: "dict | None" = None
) -> list[str]:
    """For the local marker paths: at most LOCAL_MAX_IMAGES_PER_TURN pictures, taken
    from the NEWEST result that decodes, since a batch lands as one turn and a
    non-GGUF message takes one image."""
    attempts = [LOCAL_MAX_IMAGES_PER_TURN + DECODE_FAILURE_ALLOWANCE]
    for images in reversed(list(results)):
        if attempts[0] <= 0:
            break
        urls = _decoded_urls(images, LOCAL_MAX_IMAGES_PER_TURN, attempts = attempts, cache = cache)
        if urls:
            return [url.split(",", 1)[1] for url in urls]
    return []


def flattened_rgb(image, background = None):
    """Composites alpha onto white (black for light ink), since convert('RGB') keeps the colour under it."""
    from PIL import Image, ImageChops, ImageStat

    # Match routes/inference.py _image_bytes_to_png_b64; I;16B/L reject point(), go through 'I'.
    if image.mode.startswith("I;16"):
        if image.mode != "I;16":
            image = image.convert("I")
        image = image.point(lambda v: v * (1.0 / 257), mode = "L")
        # A 16-bit tRNS key would match the wrong 8-bit samples; drop it, as convert("RGB") did.
        image.info.pop("transparency", None)
    has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] == 255:
        return rgba.convert("RGB")
    if background is None:
        # Alpha-weighted: light ink (dark-mode logos, white text) goes onto black, not white.
        ink = ImageStat.Stat(ImageChops.multiply(rgba.convert("L"), alpha)).sum[0]
        light = 255 * ink > 128 * ImageStat.Stat(alpha).sum[0] > 0
        background = (0, 0, 0) if light else (255, 255, 255)
    canvas = Image.new("RGB", rgba.size, background)
    canvas.paste(rgba, mask = alpha)
    return canvas


def _png_data_url(data: str) -> str | None:
    # PNG always: llama-server's stb_image reads few formats (MCP servers often send WebP).
    try:
        raw = base64.b64decode(data, validate = True)
    except (binascii.Error, ValueError, TypeError):
        logger.debug("MCP image payload is not base64")
        return None
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            logger.debug("MCP image is %dx%d, past the pixel budget", width, height)
            return None
        image.draft("RGB", (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        image.load()
        # Apply EXIF orientation, or the model sees camera JPEGs sideways.
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image) or image
        if max(image.size) > MAX_IMAGE_EDGE:
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        flattened_rgb(image).save(buffer, format = "PNG")
    except Exception:
        logger.debug("MCP image could not be decoded", exc_info = True)
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def png_payloads(images: Sequence[dict]) -> list[str]:
    """Normalized PNG base64, for backends that take images as objects rather
    than as data URLs inside the prompt."""
    return [url.split(",", 1)[1] for url in _decoded_urls(images)]


def _turn_text(
    shown: int,
    total: int,
    lead: str = IMAGE_TURN_TEXT,
) -> str:
    if total > shown:
        # Report counts: batches keep newest results and trims remove oldest parts,
        # so a positional label could identify images the model never saw.
        return f"{lead} ({shown} of {total})"
    return lead


def placeholder_turn(
    count: int,
    total: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
) -> dict:
    """The user turn a local processor renders: ``{"type": "image"}`` markers the
    template turns into image tokens, with the pixels passed alongside."""
    return {
        "role": "user",
        "content": [
            *({"type": "image"} for _ in range(count)),
            {
                "type": "text",
                "text": _turn_text(count, count if total is None else total, lead),
            },
        ],
    }


def _relabelled(
    kept: list,
    part_type: str,
    original: int,
    owned: "set | None" = None,
) -> list:
    """Rewrites the note for the pictures left after a partial trim, not counting the caller's own."""
    remaining = sum(
        1
        for part in kept
        if isinstance(part, dict)
        and part.get("type") == part_type
        and (owned is None or id(part) in owned)
    )
    out = []
    for part in kept:
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and _is_image_turn_note(part.get("text"))
        ):
            lead = (
                DETACHED_IMAGE_TURN_TEXT
                if str(part.get("text", "")).startswith(DETACHED_IMAGE_TURN_TEXT)
                else IMAGE_TURN_TEXT
            )
            total = _note_total(part.get("text"), original)
            out.append({**part, "text": _turn_text(remaining, total, lead)})
            continue
        out.append(part)
    return out


def _note_total(text, original: int) -> int:
    """Keeps the note's own 'of N', else falls back to the pre-trim count, not the post-trim one."""
    match = re.search(r"\((?:first )?\d+ of (\d+)\)\s*$", str(text or ""))
    if match:
        return int(match.group(1))
    return original


def _is_image_turn_note(text) -> bool:
    """Either lead the placeholder turn can carry, so a turn whose last picture went
    is recognised as having nothing left to say on both paths."""
    value = str(text or "")
    return value.startswith(IMAGE_TURN_TEXT) or value.startswith(DETACHED_IMAGE_TURN_TEXT)


def _image_parts(conversation: Sequence[dict], part_type: str):
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            yield from (
                part for part in content if isinstance(part, dict) and part.get("type") == part_type
            )


def _all_image_url_parts(conversation: Sequence[dict]) -> list:
    return list(_image_parts(conversation, "image_url"))


def count_image_parts(conversation: Sequence[dict], part_type: str) -> int:
    return sum(1 for _ in _image_parts(conversation, part_type))


def _drop_oldest_image_parts(
    conversation: list,
    excess: int,
    part_type: str,
    only: "list | None" = None,
) -> None:
    """With only, touches just the promoted parts, so a caller's own attachments survive the cap."""
    owned = {id(part) for part in only} if only is not None else None
    ordinals = set()
    for ordinal, part in enumerate(_image_parts(conversation, part_type)):
        if len(ordinals) >= excess:
            break
        if owned is None or id(part) in owned:
            ordinals.add(ordinal)
    _drop_image_parts_at(conversation, ordinals, part_type, owned = owned)


def _drop_image_parts_at(
    conversation: list,
    ordinals: set,
    part_type: str,
    *,
    owned: "set | None" = None,
) -> None:
    """Drops by position, not oldest-first: a count would bind later pixels to the wrong markers."""
    if not ordinals:
        return
    seen = 0
    drained = []
    for index, message in enumerate(conversation):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        original = sum(
            1
            for part in content
            if isinstance(part, dict)
            and part.get("type") == part_type
            and (owned is None or id(part) in owned)
        )
        kept = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == part_type:
                ordinal = seen
                seen += 1
                if ordinal in ordinals:
                    continue
            kept.append(part)
        if len(kept) == len(content):
            continue
        if not kept or (
            len(kept) == 1
            and kept[0].get("type") == "text"
            and _is_image_turn_note(kept[0].get("text"))
        ):
            drained.append(index)
        else:
            conversation[index] = {
                **message,
                "content": _relabelled(kept, part_type, original, owned = owned),
            }
    for index in reversed(drained):
        del conversation[index]


def trim_image_turns(
    conversation: list,
    payloads: list,
    limit: int = MAX_TOTAL_MODEL_IMAGES,
    keep: "Sequence[int] | None" = None,
) -> tuple:
    """Keeps the newest limit pictures; protected indexes survive, and all markers count against limit."""
    protected = sorted(index for index in (keep or ()) if 0 <= index < len(payloads))
    excess = len(payloads) - limit
    if excess <= 0:
        return tuple(protected)
    drop = [index for index in range(len(payloads)) if index not in set(protected)][:excess]
    if not drop:
        return tuple(protected)
    _drop_image_parts_at(conversation, set(drop), "image")
    for index in reversed(drop):
        del payloads[index]
    # Rebase protected positions; callers reuse them on later trims.
    dropped = set(drop)
    return tuple(index - sum(1 for gone in dropped if gone < index) for index in protected)


def trim_image_url_turns(
    conversation: list,
    limit: int = MAX_TOTAL_MODEL_IMAGES,
    only: "list | None" = None,
) -> None:
    """Same cap for data-URL pixels; only promoted parts count, never the caller's own attachments."""
    if only is not None:
        # Prune first: the context fitter may have evicted turns.
        present = {id(part) for part in _all_image_url_parts(conversation)}
        only[:] = [part for part in only if id(part) in present]
    counted = len(only) if only is not None else count_image_parts(conversation, "image_url")
    excess = counted - limit
    if excess <= 0:
        return
    _drop_oldest_image_parts(conversation, excess, "image_url", only = only)
    if only is not None:
        still_present = {id(part) for part in _all_image_url_parts(conversation)}
        only[:] = [part for part in only if id(part) in still_present]


def _merge_into_trailing_user_turn(conversation: list, parts: list[dict]) -> bool:
    """Folds parts into a trailing user turn, since two user messages in a row break strict VLM
    templates."""
    last = conversation[-1] if conversation else None
    if not isinstance(last, dict) or last.get("role") != "user":
        return False
    content = last.get("content")
    own = list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    conversation[-1] = {**last, "content": [*own, *parts]}
    return True


def append_image_turn(
    conversation: list,
    images: Sequence,
    *,
    limit: "int | None" = MAX_TOTAL_MODEL_IMAGES,
    per_result: bool = False,
    owned: "list | None" = None,
    reserve_caller_images: bool = False,
    returned: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
) -> None:
    """Adds a user turn, since tool messages take no image parts; per_result keeps each result's quota."""
    parts = content_parts_per_result(images) if per_result else content_parts(images)
    if not parts:
        return
    # Count what the TOOL returned, not the admission-sliced candidates.
    total = (
        returned
        if returned is not None
        else (sum(len(result) for result in images) if per_result else len(images))
    )
    if owned is not None:
        owned.extend(parts)
    note = {"type": "text", "text": _turn_text(len(parts), total, lead)}
    if not _merge_into_trailing_user_turn(conversation, [*parts, note]):
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _turn_text(len(parts), total, lead)},
                    *parts,
                ],
            }
        )
    if limit is not None:
        if reserve_caller_images:
            # Providers cap images in document order; reserve attachment slots so the newest survive.
            limit = max(0, limit - (len(_all_image_url_parts(conversation)) - len(owned or ())))
        trim_image_url_turns(conversation, limit, only = owned)


def insert_placeholder_turn(
    conversation: list,
    index: int,
    count: int,
    total: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
) -> None:
    """Marker-only turn placed before index, as a positional processor binds pixels to markers in order."""
    if count > 0:
        conversation.insert(index, placeholder_turn(count, total, lead))


def append_placeholder_turn(
    conversation: list,
    count: int,
    total: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
) -> None:
    """The note rides on the merge, so markers are not misattributed to a nudge or the user's question."""
    markers = [{"type": "image"} for _ in range(count)]
    if not markers:
        return
    note = {"type": "text", "text": _turn_text(count, count if total is None else total, lead)}
    if not _merge_into_trailing_user_turn(conversation, [*markers, note]):
        conversation.append(placeholder_turn(count, total, lead))


def _without_replay_parts(content: list) -> list:
    """The turn without a replay's marker AND its note. A non-GGUF message takes one
    image, so the attachment displaces a replay marker merged into its turn; the
    note that came with it would then say the caller's own picture was the tool's."""
    return [
        part
        for part in content
        if not (
            isinstance(part, dict)
            and (
                part.get("type") == "image"
                or (part.get("type") == "text" and _is_image_turn_note(part.get("text")))
            )
        )
    ]


def top_up_image_markers(
    messages: Sequence[dict],
    total: int,
    *,
    ordinal: "int | None" = None,
) -> list[dict]:
    """Adds the shortfall to the newest user turn, since replayed markers belong to earlier turns."""
    out = list(messages)
    have = sum(
        1
        for message in out
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_image")
    )
    missing = total - have
    if missing <= 0:
        return out
    markers = [{"type": "image"} for _ in range(missing)]
    if ordinal is not None:
        seen = 0
        for index, message in enumerate(out):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if is_synthetic_image_turn(message):
                continue
            if seen == ordinal:
                out[index] = _with_attachment_markers(message, markers, after_text = True)
                return out
            seen += 1
    candidates = [
        index
        for index in range(len(out) - 1, -1, -1)
        if isinstance(out[index], dict) and out[index].get("role") == "user"
    ]
    real = [index for index in candidates if not is_synthetic_image_turn(out[index])]
    for index in real or candidates:
        out[index] = _with_attachment_markers(out[index], markers, after_text = True)
        break
    return out


def image_marker_parts(conversation: Sequence[dict]) -> list:
    """Every ``{"type": "image"}`` marker part, in document order."""
    return list(_image_parts(conversation, "image"))


def pixels_in_marker_order(
    conversation: Sequence[dict],
    prior_markers: Sequence[dict],
    prior_payloads: Sequence,
    new_payload,
    placed_at: "list | None" = None,
) -> list:
    """Orders pixels by where their markers sit, since a positional VLM binds pixels to markers in order."""
    # Pair payloads with markers by identity; popping from the front caused an off-by-one.
    by_marker = {id(part): payload for part, payload in zip(prior_markers, prior_payloads)}
    ordered = []
    placed_new = False
    for part in image_marker_parts(conversation):
        if id(part) in by_marker:
            ordered.append(by_marker[id(part)])
        elif not placed_new:
            if placed_at is not None:
                placed_at.append(len(ordered))
            ordered.append(new_payload)
            placed_new = True
    return ordered


def is_synthetic_image_turn(message) -> bool:
    """Ordinals count real user turns only, so skip promotion's inserted turns or markers hit history."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    texts = [
        part.get("text")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return bool(texts) and all(_is_image_turn_note(text) for text in texts)


def prepare_image_turn_boundaries(messages: list, template: str | None) -> list:
    """Balance inserted image turns for Ministral's tool-skipping alternation check."""
    if not isinstance(template, str) or (
        "conversation roles must alternate user and assistant roles except for tool calls and results."
        not in template
    ):
        return messages
    out = []
    expects_user = True
    changed = False
    for message in messages:
        role = message.get("role")
        if role == "user":
            if not expects_user and is_synthetic_image_turn(message):
                out.append({"role": "assistant", "content": "The tool returned image content."})
                changed = True
            expects_user = False
        elif role == "assistant" and not message.get("tool_calls"):
            expects_user = True
        out.append(message)
    return out if changed else messages


def _with_attachment_markers(
    message: dict,
    markers: list[dict],
    *,
    after_text: bool = False,
) -> dict:
    """Replace a replay's marker and attribution with the caller's attachment."""
    content = message.get("content")
    own = (
        _without_replay_parts(content)
        if isinstance(content, list)
        else [{"type": "text", "text": content or ""}]
    )
    parts = [*own, *markers] if after_text and isinstance(content, list) else [*markers, *own]
    return {**message, "content": parts}


def mark_last_user_turn(
    messages: Sequence[dict],
    count: int,
    *,
    ordinal: "int | None" = None,
) -> list[dict]:
    """Marks the turn at ordinal, since an attachment can belong to an older question than the newest."""
    out = list(messages)
    markers = [{"type": "image"} for _ in range(count)]
    if ordinal is not None:
        seen = 0
        for index, message in enumerate(out):
            if message.get("role") != "user" or is_synthetic_image_turn(message):
                continue
            if seen == ordinal:
                out[index] = _with_attachment_markers(message, markers)
                return out
            seen += 1
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            out[index] = _with_attachment_markers(out[index], markers)
            break
    return out


def promote_history(
    messages: Sequence[dict],
    *,
    vision: bool,
    promoted_out: "list | None" = None,
    reserve_for_caller: bool = False,
) -> list[dict]:
    """Strips the envelope from tool text either way, so text-only models never see base64."""
    out, _payloads, promoted = _promote(
        messages, vision, local = False, reserve_for_caller = reserve_for_caller
    )
    if promoted_out is not None:
        promoted_out.extend(promoted)
    return out


def promote_history_local(
    messages: Sequence[dict],
    *,
    vision: bool,
    decode_cache: "dict | None" = None,
    caller_images: Sequence = (),
) -> tuple[list[dict], list[str]]:
    """The same, for backends that take the pixels beside the prompt: the turns
    carry markers and the payloads come back with them. ``caller_images`` supplies
    the pixels for existing markers, which retain their positions and survive trimming."""
    out, payloads, _promoted = _promote(
        messages, vision, local = True, decode_cache = decode_cache, caller_images = caller_images
    )
    return out, payloads


def _field(message, key):
    return message.get(key) if isinstance(message, dict) else getattr(message, key, None)


def resolve_tool_names(messages: Sequence) -> dict:
    """Pairs each result with the nearest unmatched earlier call of its id; ids like call_0 repeat."""
    open_calls: dict = {}
    names: dict = {}
    for index, message in enumerate(messages or ()):
        for call in _field(message, "tool_calls") or ():
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            call_id = call.get("id")
            if isinstance(function, dict) and isinstance(call_id, str):
                name = function.get("name")
                if isinstance(name, str) and name:
                    open_calls.setdefault(call_id, []).append(name)
        if _field(message, "role") == "tool":
            call_id = _field(message, "tool_call_id")
            stack = open_calls.get(call_id) if isinstance(call_id, str) else None
            if stack:
                names[index] = stack.pop()
    return names


def _returned_count(images: Sequence[dict]) -> int:
    """How many the tool returned. An upstream bound may already have shortened the
    array; it leaves the original length on the first entry when it does."""
    first = images[0] if images else None
    stated = first.get("returned") if isinstance(first, dict) else None
    if isinstance(stated, int) and stated >= len(images):
        return stated
    return len(images)


def _promote(
    messages,
    vision: bool,
    *,
    local: bool,
    reserve_for_caller: bool = False,
    decode_cache: "dict | None" = None,
    caller_images: Sequence = (),
) -> tuple[list[dict], list[str], list[dict]]:
    out: list[dict] = []
    caller_payloads = (
        dict(zip(map(id, image_marker_parts(messages)), caller_images)) if caller_images else {}
    )
    call_names = resolve_tool_names(messages)
    interrupted = [False]
    # Reserve the caller's room before decoding so replay candidates are not decoded just to be dropped.
    _reserved = (
        len(caller_payloads)
        if local
        else (len(_all_image_url_parts(messages)) if vision and reserve_for_caller else 0)
    )
    eligible = (
        eligible_replay_images(
            messages, local = local, budget = max(0, MAX_TOTAL_MODEL_IMAGES - _reserved)
        )
        if vision
        else {}
    )
    # One entry per tool result so parallel calls do not share one quota.
    pending: list[list[dict]] = []
    returned_totals: list[int] = []
    payloads: list[str] = []
    promoted: list[dict] = []

    def flush(into: "dict | None" = None) -> "dict | None":
        if not pending or not vision:
            pending.clear()
            returned_totals.clear()
            return into
        returned = sum(returned_totals) or sum(len(result) for result in pending)
        # Detached wording for multi-result batches too: 'the tool call above' would be wrong.
        lead = DETACHED_IMAGE_TURN_TEXT if interrupted[0] or len(pending) > 1 else IMAGE_TURN_TEXT
        interrupted[0] = False
        if local:
            if into is not None and any(
                id(part) in caller_payloads for part in image_marker_parts([into])
            ):
                pending.clear()
                returned_totals.clear()
                return into
            encoded = png_payloads_per_result(pending, cache = decode_cache)
            pending.clear()
            returned_totals.clear()
            if not encoded:
                return into
            payloads.extend(encoded)
            markers = [{"type": "image"} for _ in encoded]
            if into is None:
                out.append(placeholder_turn(len(encoded), returned, lead))
                return None
            note = {"type": "text", "text": _turn_text(len(encoded), returned, lead)}
            return _with_parts(into, [*markers, note])
        results = list(pending)
        pending.clear()
        returned_totals.clear()
        if into is None:
            before = {id(part) for part in _all_image_url_parts(out)}
            append_image_turn(
                out, results, per_result = True, limit = None, returned = returned, lead = lead
            )
            promoted.extend(part for part in _all_image_url_parts(out) if id(part) not in before)
            return None
        parts = content_parts_per_result(results)
        promoted.extend(parts)
        if not parts:
            return into
        note = {"type": "text", "text": _turn_text(len(parts), returned, lead)}
        return _with_parts(into, [*parts, note])

    for position, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            text, images = split_images(content)
            # The suffix always comes off; provenance decides only whether it becomes image input.
            name = message.get("name") or call_names.get(position)
            if isinstance(name, str) and name and not is_image_tool(name):
                if pending:
                    interrupted[0] = True
                out.append(
                    {**message, "content": text or "[image returned]"} if images else message
                )
                continue
            if images:
                admitted = images[: eligible.get(position, len(images))]
                if admitted:
                    pending.append(admitted)
                returned_totals.append(_returned_count(images))
            elif pending:
                interrupted[0] = True
            out.append({**message, "content": text or "[image returned]"} if images else message)
            continue
        if pending and vision and message.get("role") == "user":
            # Merged: two consecutive user turns break strict templates.
            out.append(flush(message))
            continue
        flush()
        out.append(message)
    flush()
    if local:
        protected = ()
        if caller_payloads:
            markers = image_marker_parts(out)
            replay_payloads = dict(
                zip((id(part) for part in markers if id(part) not in caller_payloads), payloads)
            )
            by_marker = {**replay_payloads, **caller_payloads}
            payloads = [by_marker[id(part)] for part in markers]
            protected = tuple(
                index for index, part in enumerate(markers) if id(part) in caller_payloads
            )
        trim_image_turns(out, payloads, keep = protected)
    else:
        # Providers cap images in document order (Gemini: 8, rest dropped), so reserve the attachment's room.
        _caller_parts = len(_all_image_url_parts(out)) - len(promoted) if reserve_for_caller else 0
        trim_image_url_turns(
            out,
            limit = max(0, MAX_TOTAL_MODEL_IMAGES - _caller_parts),
            only = promoted,
        )
    return out, payloads, promoted


def _with_parts(message: dict, parts: list[dict]) -> dict:
    content = message.get("content")
    own = list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    return {**message, "content": [*parts, *own]}
