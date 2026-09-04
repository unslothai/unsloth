# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import binascii
import io
import json
from typing import Any, Sequence

from loggers import get_logger

logger = get_logger(__name__)

SENTINEL = "__MCP_IMAGES__:"
# Stamped by mcp_client on every tool it registers; the provenance the envelope
# is trusted on, on the replay side as well as the live one.
MCP_TOOL_PREFIX = "mcp__"
IMAGE_TURN_TEXT = "Images returned by the tool call above:"
# The same pictures where the turn cannot sit beside the result that produced them.
# The local client-tool passthrough flattens every content part to text before the
# markers are rebuilt, so they come back as one block rather than at the positions
# they were taken from, and "above" would name whatever turn happens to precede it.
DETACHED_IMAGE_TURN_TEXT = "Images returned by earlier tool calls in this conversation:"

MAX_MODEL_IMAGES = 4
MAX_TOTAL_MODEL_IMAGES = 8
MAX_IMAGE_EDGE = 1024
# A PNG stays small while its raster does not: 12 MB of encoded payload can hold
# tens of gigapixels. Bounded off the header, before a pixel is allocated.
MAX_IMAGE_PIXELS = 40_000_000


def split_images(result: str) -> tuple[str, list[dict]]:
    """Validated, so tool text that merely mentions the marker is not truncated."""
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


# Markup Pillow will never open, sniffed by prefix. Deliberately a DENY list:
# an allow list of magic bytes drifts behind whatever Pillow can actually decode,
# and every format it gains that this misses is an image promotion sends and
# admission reserved nothing for -- which overcommits the KV cache. Guessing the
# other way merely over-reserves.
_UNDECODABLE_PREFIXES = (
    b"<svg",
    b"<?xml",
    b"<!DOCTYPE",
    b"<html",
    b"{",
    b"[",
)


def probably_decodable(image: Any) -> bool:
    """Whether this entry could become a picture.

    Header sniff only: admission runs per request and must not decode rasters to
    price them. Answers True unless the payload is plainly not an image, so a
    format Pillow can open is never charged nothing.
    """
    data = image.get("data") if isinstance(image, dict) else None
    if not isinstance(data, str) or not data:
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


def has_images(result: str) -> bool:
    return bool(split_images(result)[1])


def _decoded_urls(images: Sequence[dict], limit: int = MAX_MODEL_IMAGES) -> list[str]:
    """Up to *limit* data URLs, counting only what decoded.

    Slicing first would spend the quota on formats Pillow cannot read -- an SVG
    _flatten_result accepted, say -- and drop the real PNGs behind them.
    """
    urls = []
    for image in images:
        if len(urls) >= limit:
            break
        url = _png_data_url(image.get("data", ""))
        if url:
            urls.append(url)
    return urls


def _decoded_urls_per_result(results: Sequence[Sequence[dict]]) -> list[str]:
    """MAX_MODEL_IMAGES from EACH result, then the conversation cap over the whole.

    A parallel batch arrives as several results concatenated. Applying the
    per-result quota to the concatenation gives the first result the whole
    allowance and delivers none of the second, though the conversation cap has
    room for both.
    """
    urls: list[str] = []
    for images in results:
        if len(urls) >= MAX_TOTAL_MODEL_IMAGES:
            break
        room = MAX_TOTAL_MODEL_IMAGES - len(urls)
        urls.extend(_decoded_urls(images, min(MAX_MODEL_IMAGES, room)))
    return urls


def content_parts(images: Sequence[dict]) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": url}} for url in _decoded_urls(images)]


def content_parts_per_result(results: Sequence[Sequence[dict]]) -> list[dict]:
    """content_parts for a batch, keeping each result's own quota."""
    return [
        {"type": "image_url", "image_url": {"url": url}}
        for url in _decoded_urls_per_result(results)
    ]


def png_payloads_per_result(results: Sequence[Sequence[dict]]) -> list[str]:
    return [url.split(",", 1)[1] for url in _decoded_urls_per_result(results)]


def _png_data_url(data: str) -> str | None:
    # PNG regardless of what the server sent: llama-server's stb_image reads only
    # a few formats, and MCP servers commonly answer with WebP.
    try:
        raw = base64.b64decode(data, validate = True)
    except (binascii.Error, ValueError, TypeError):
        logger.debug("MCP image payload is not base64")
        return None
    try:
        from PIL import Image

        # open() parses the header only, so the size is known before the raster
        # exists. load() below is what allocates it.
        image = Image.open(io.BytesIO(raw))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            logger.debug("MCP image is %dx%d, past the pixel budget", width, height)
            return None
        # JPEG decodes straight to a smaller raster; a no-op for every other format.
        image.draft("RGB", (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        image.load()
        if max(image.size) > MAX_IMAGE_EDGE:
            image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format = "PNG")
    except Exception:
        logger.debug("MCP image could not be decoded", exc_info = True)
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def png_payloads(images: Sequence[dict]) -> list[str]:
    """Normalized PNG base64, for backends that take images as objects rather
    than as data URLs inside the prompt."""
    return [url.split(",", 1)[1] for url in _decoded_urls(images)]


def _turn_text(shown: int, total: int, lead: str = IMAGE_TURN_TEXT) -> str:
    # The tool result's own note counts every image it returned, so a turn that
    # carries fewer has to say so rather than let the model wait for the rest.
    if total > shown:
        return f"{lead} (first {shown} of {total})"
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


def _all_image_url_parts(conversation: Sequence[dict]) -> list:
    return [
        part
        for message in conversation
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def count_image_parts(conversation: Sequence[dict], part_type: str) -> int:
    return sum(
        1
        for message in conversation
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == part_type
    )


def _drop_oldest_image_parts(
    conversation: list,
    excess: int,
    part_type: str,
    only: "list | None" = None,
    skip: int = 0,
) -> None:
    """Drop the *excess* oldest image parts, and any turn they emptied.

    With *only*, a list of the exact part objects promotion created, nothing else
    is touched: a caller that attaches more than the cap keeps every one of its own
    images, which admission has already charged for.

    *skip* is the same protection where the parts are bare markers that carry no
    identity: the first *skip* of them are the caller's and are passed over.
    """
    owned = {id(part) for part in only} if only is not None else None
    drained = []
    for index, message in enumerate(conversation):
        if excess <= 0:
            break
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == part_type
                and (owned is None or id(part) in owned)
            ):
                if skip > 0:
                    skip -= 1
                elif excess > 0:
                    excess -= 1
                    continue
            kept.append(part)
        if len(kept) == len(content):
            continue
        if not kept or (
            len(kept) == 1
            and kept[0].get("type") == "text"
            and str(kept[0].get("text", "")).startswith(IMAGE_TURN_TEXT)
        ):
            # An image-only turn whose last picture just went has nothing left to
            # say. Written back as content: [] it becomes an empty user message,
            # which strict provider APIs and chat templates reject.
            drained.append(index)
        else:
            conversation[index] = {**message, "content": kept}
    for index in reversed(drained):
        del conversation[index]


def trim_image_turns(
    conversation: list,
    payloads: list,
    limit: int = MAX_TOTAL_MODEL_IMAGES,
    protected: int = 0,
) -> None:
    """Keep the newest *limit* pictures: a loop that keeps calling an image tool
    otherwise re-sends every one it has seen. Markers and their own pixels go
    together, or the processor counts image tokens it was given none for.

    *protected* is the count of leading payloads the caller attached. Bare markers
    carry no identity the way promoted ``image_url`` parts do, so the sink's own
    order is what says which are the caller's: they seed it before the loop runs.
    Like the ``only`` scoping on the ``image_url`` cap, they are neither counted
    against the limit nor deleted by it -- this cap is about what a tool loop
    re-sends, and evicting the picture the question was asked about answers a
    different question."""
    excess = len(payloads) - protected - limit
    if excess <= 0:
        return
    del payloads[protected : protected + excess]
    _drop_oldest_image_parts(conversation, excess, "image", skip = protected)


def trim_image_url_turns(
    conversation: list,
    limit: int = MAX_TOTAL_MODEL_IMAGES,
    only: "list | None" = None,
) -> None:
    """The same cap where the pixels ride in the prompt as data URLs.

    GGUF and the external providers carry no separate payload list, so the parts
    themselves are the budget: without this a screenshot loop resends every
    picture it has ever taken on every later turn.

    *only* scopes the cap to the parts promotion created. This cap is about what
    REPLAY re-sends; a caller's own attachments are not counted against it and are
    never deleted by it.
    """
    counted = len(only) if only is not None else count_image_parts(conversation, "image_url")
    excess = counted - limit
    if excess <= 0:
        return
    _drop_oldest_image_parts(conversation, excess, "image_url", only = only)
    if only is not None:
        # Drop what the trim removed, or the next call counts parts that are no
        # longer in the conversation and cuts far more than the cap asks for.
        still_present = {id(part) for part in _all_image_url_parts(conversation)}
        only[:] = [part for part in only if id(part) in still_present]


def _merge_into_trailing_user_turn(conversation: list, parts: list[dict]) -> bool:
    """Fold *parts* into a trailing ``role=user`` turn, if there is one.

    A deferred no-op nudge lands as exactly such a turn, and appending after it
    would put two user messages in a row -- which a strict VLM template rejects.
    """
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
) -> None:
    """A user turn, not the ``role=tool`` result they came with: tool messages take
    no image parts, and local templates render tool content as a string.

    With *per_result*, *images* is a list of per-result lists and each keeps its own
    MAX_MODEL_IMAGES quota; flattened first, a parallel batch would spend the whole
    allowance on its first call.
    """
    parts = content_parts_per_result(images) if per_result else content_parts(images)
    if not parts:
        return
    total = sum(len(result) for result in images) if per_result else len(images)
    if owned is not None:
        # Everything this loop has appended, across turns. The cap counts what the
        # tools returned and never a caller's own attachments, which admission has
        # already charged for and which are not this cap's business.
        owned.extend(parts)
    if not _merge_into_trailing_user_turn(conversation, parts):
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _turn_text(len(parts), total)},
                    *parts,
                ],
            }
        )
    if limit is not None:
        trim_image_url_turns(conversation, limit, only = owned)


def insert_placeholder_turn(
    conversation: list,
    index: int,
    count: int,
    total: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
) -> None:
    """A marker-only user turn placed *before* ``index``.

    The pixels arrive history-first with the current attachment last, and a
    positional VLM processor binds them to markers in document order. So the
    replayed markers have to sit ahead of the turn that owns the attachment,
    not after it.
    """
    if count > 0:
        conversation.insert(index, placeholder_turn(count, total, lead))


def append_placeholder_turn(
    conversation: list,
    count: int,
    total: "int | None" = None,
    lead: str = IMAGE_TURN_TEXT,
    annotate_merge: bool = False,
) -> None:
    """The marker-only form of the above, for backends taking pixels alongside.

    *annotate_merge* carries the note into the merge as well. The live loop merges
    into a deferred no-op nudge, which is synthetic and needs none. The passthrough
    merges into the user's real latest question, where bare markers read as pictures
    the USER attached to it -- so the model answers about ones nobody sent.
    """
    markers = [{"type": "image"} for _ in range(count)]
    if not markers:
        return
    merged = list(markers)
    if annotate_merge:
        merged.append(
            {"type": "text", "text": _turn_text(count, count if total is None else total, lead)}
        )
    if not _merge_into_trailing_user_turn(conversation, merged):
        conversation.append(placeholder_turn(count, total, lead))


def top_up_image_markers(
    messages: Sequence[dict],
    total: int,
    *,
    ordinal: "int | None" = None,
) -> list[dict]:
    """Give the conversation exactly *total* image markers, adding any shortfall
    to the newest user turn.

    ``messages_with_attached_image`` leaves a conversation that already carries
    markers alone, which is right for a nudge retry but wrong for replayed MCP
    pictures: those markers belong to earlier turns, not to the attachment. The
    top-up goes last, matching where the attachment sits in the pixel list.
    """
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
    if ordinal is not None:
        # The turn that supplied the attachment, which need not be the newest: a
        # later text-only question would otherwise be shown as carrying it.
        seen = 0
        for index, message in enumerate(out):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            if is_synthetic_image_turn(message):
                continue
            if seen == ordinal:
                content = message.get("content", "")
                markers = [{"type": "image"} for _ in range(missing)]
                if isinstance(content, list):
                    out[index] = {**message, "content": [*content, *markers]}
                else:
                    out[index] = {
                        **message,
                        "content": [*markers, {"type": "text", "text": content or ""}],
                    }
                return out
            seen += 1
    for index in range(len(out) - 1, -1, -1):
        message = out[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        markers = [{"type": "image"} for _ in range(missing)]
        if isinstance(content, list):
            out[index] = {**message, "content": [*content, *markers]}
        else:
            out[index] = {
                **message,
                "content": [*markers, {"type": "text", "text": content or ""}],
            }
        break
    return out


def image_marker_parts(conversation: Sequence[dict]) -> list:
    """Every ``{"type": "image"}`` marker part, in document order."""
    return [
        part
        for message in conversation
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ]


def pixels_in_marker_order(
    conversation: Sequence[dict],
    prior_markers: Sequence[dict],
    prior_payloads: Sequence,
    new_payload,
) -> list:
    """The pixel list ordered the way its markers actually appear.

    A positional VLM binds the Nth pixel to the Nth marker, so "history first,
    attachment last" is only right when the attachment's turn is last. It is not
    when the attachment came with an earlier question and a tool returned pictures
    after it, so the order is read off the conversation rather than assumed.
    """
    prior_ids = {id(part) for part in prior_markers}
    remaining = list(prior_payloads)
    ordered = []
    placed_new = False
    for part in image_marker_parts(conversation):
        # A marker that predates the top-up belongs to history for as long as
        # history has pixels left to give it. Once those run out, a pre-existing
        # marker is the attachment's own -- the client may have marked it before
        # the request ever reached this path.
        if id(part) in prior_ids and remaining:
            ordered.append(remaining.pop(0))
        elif not placed_new:
            ordered.append(new_payload)
            placed_new = True
        elif remaining:
            ordered.append(remaining.pop(0))
    return ordered


def is_synthetic_image_turn(message) -> bool:
    """Whether promotion inserted this user turn, rather than the caller sending it.

    An ordinal computed against the ORIGINAL history counts only real user turns,
    so resolving it against the promoted list has to skip the turns promotion added
    or the attachment's marker lands on a historical picture's turn.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict)
        and part.get("type") == "text"
        and str(part.get("text", "")).startswith(IMAGE_TURN_TEXT)
        for part in content
    )


def mark_last_user_turn(
    messages: Sequence[dict],
    count: int,
    *,
    ordinal: "int | None" = None,
) -> list[dict]:
    """Mark the user turn carrying ``count`` images, where an attachment belongs.

    *ordinal* names that turn counted among user turns. The newest user turn is
    only the right guess when the attachment came with the newest question: the
    extractor takes the newest user image from anywhere in the thread, so a
    text-only latest turn would otherwise be told it supplied an older picture.
    """
    out = list(messages)
    markers = [{"type": "image"} for _ in range(count)]
    if ordinal is not None:
        seen = 0
        for index, message in enumerate(out):
            # Promotion inserts its own user turns ahead of this, and the ordinal was
            # counted before they existed.
            if message.get("role") != "user" or is_synthetic_image_turn(message):
                continue
            if seen == ordinal:
                out[index] = _with_parts(message, markers)
                return out
            seen += 1
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            out[index] = _with_parts(out[index], markers)
            break
    return out


def promote_history(
    messages: Sequence[dict],
    *,
    vision: bool,
    promoted_out: "list | None" = None,
) -> list[dict]:
    """Rebuild image turns from replayed envelopes. The envelope leaves the tool
    text either way: a text-only model must not be shown its base64.

    *promoted_out* collects the exact image parts this call created, so a tool loop
    resuming the conversation can seed its own cap with them instead of starting
    from zero and letting the history's images through uncounted.
    """
    out, _payloads, promoted = _promote(messages, vision, local = False)
    if promoted_out is not None:
        promoted_out.extend(promoted)
    return out


def promote_history_local(
    messages: Sequence[dict], *, vision: bool
) -> tuple[list[dict], list[str]]:
    """The same, for backends that take the pixels beside the prompt: the turns
    carry markers and the payloads come back with them."""
    out, payloads, _promoted = _promote(messages, vision, local = True)
    return out, payloads


def _promote(messages, vision: bool, *, local: bool) -> tuple[list[dict], list[str], list[dict]]:
    out: list[dict] = []
    # One entry per tool result, not flattened: two parallel calls each returning
    # four images would otherwise share a single result's quota and replay only the
    # first call's four.
    pending: list[list[dict]] = []
    payloads: list[str] = []
    # The exact part objects promotion creates, so the cap below can leave a
    # caller's own attachments alone.
    promoted: list[dict] = []

    def flush(into: "dict | None" = None) -> "dict | None":
        if not pending or not vision:
            pending.clear()
            return into
        returned = sum(len(result) for result in pending)
        if local:
            encoded = png_payloads_per_result(pending)
            pending.clear()
            if not encoded:
                return into
            payloads.extend(encoded)
            markers = [{"type": "image"} for _ in encoded]
            if into is None:
                out.append(placeholder_turn(len(encoded), returned))
                return None
            return _with_parts(into, markers)
        results = list(pending)
        pending.clear()
        if into is None:
            before = {id(part) for part in _all_image_url_parts(out)}
            append_image_turn(out, results, per_result = True, limit = None)
            promoted.extend(part for part in _all_image_url_parts(out) if id(part) not in before)
            return None
        parts = content_parts_per_result(results)
        promoted.extend(parts)
        return _with_parts(into, parts) if parts else into

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            text, images = split_images(content)
            # The suffix always comes off -- it is megabytes of base64 and the model
            # must never read it as text. Provenance decides only whether it becomes
            # IMAGE input: a named non-MCP tool that happens to end in a valid
            # envelope is not one an MCP server served.
            name = message.get("name")
            if isinstance(name, str) and name and not name.startswith(MCP_TOOL_PREFIX):
                out.append(
                    {**message, "content": text or "[image returned]"} if images else message
                )
                continue
            if images:
                pending.append(images)
            out.append({**message, "content": text or "[image returned]"} if images else message)
            continue
        if pending and vision and message.get("role") == "user":
            # Merged, not inserted ahead of it: two user turns in a row is what
            # a strict template rejects.
            out.append(flush(message))
            continue
        flush()
        out.append(message)
    flush()
    # A replay carries every image turn the conversation ever had; the cap has to
    # hold here too or the whole history is re-sent on every later turn.
    if local:
        trim_image_turns(out, payloads)
    else:
        trim_image_url_turns(out, only = promoted)
    return out, payloads, promoted


def _with_parts(message: dict, parts: list[dict]) -> dict:
    content = message.get("content")
    own = list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    return {**message, "content": [*parts, *own]}
