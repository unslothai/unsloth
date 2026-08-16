# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turning a frame log into "did a code block flicker", separated so it can be tested.

`playwright_code_block_flicker.py` records, on every frame, the rendered height and the
document-space top of every `[data-streamdown="code-block"]` in the thread. Everything that
decides whether a run passes is computed here, from those numbers alone, so the decision can be
exercised against hand-written frame logs without a browser -- including the logs that must NOT
be read as a flicker.

Kept out of the harness module because the harness imports playwright, and the contract test
that exercises these rules has to run wherever the repo's CPU suite runs.
"""

from __future__ import annotations

# A block has to be this tall before a drop counts, so a genuinely short fence is never read as a
# collapsed tall one.
TALL_PX = 400
# Streamdown's inline `contain-intrinsic-size` fallback is 200px; the wrapper adds padding and a
# header row on top of it. A drop that lands in this band is the fallback specifically rather
# than some other height, which is what lets a collapse be attributed instead of merely counted.
PLACEHOLDER_LO, PLACEHOLDER_HI = 150, 300
# Frames a drop is given to come back before it is recorded as a drop that never recovered. A
# block that goes short and STAYS short is a different bug from a block that blinks.
RECOVERY_FRAMES = 240
# A block's document-space top moving further than this between two frames of a scroll gesture is
# content above it having been relaid out under the user.
SHIFT_PX = 8


def analyse_stream(frames: list[dict]) -> dict:
    """Collapse-and-recover events over a frame log, per code block.

    Block indices are stable across frames because blocks are only ever APPENDED while a reply
    streams: a fence that has started keeps its position while the ones after it appear.

    A collapse is a block that was at least TALL_PX tall rendering at half that or less and then
    coming back. Both halves are required. Requiring the recovery is what keeps a block that is
    legitimately replaced by something shorter, or a thread that is being torn down, from being
    reported as a flicker.
    """
    collapses = 0
    placeholder_frames = 0
    detail: list[dict] = []
    worst_drop_px = 0.0
    block_count = max((len(f["heights"]) for f in frames), default = 0)

    for index in range(block_count):
        series = [
            (i, f["heights"][index]) for i, f in enumerate(frames) if index < len(f["heights"])
        ]
        open_drop: tuple[int, float] | None = None
        for position in range(1, len(series)):
            frame_index, height = series[position]
            _, previous = series[position - 1]
            if open_drop is None:
                if previous >= TALL_PX and height <= previous * 0.5:
                    open_drop = (frame_index, previous)
                    worst_drop_px = max(worst_drop_px, previous - height)
                    if PLACEHOLDER_LO <= height <= PLACEHOLDER_HI:
                        placeholder_frames += 1
                continue
            start_frame, before = open_drop
            if PLACEHOLDER_LO <= height <= PLACEHOLDER_HI:
                placeholder_frames += 1
            if height >= before * 0.9:
                collapses += 1
                detail.append(
                    {
                        "block": index,
                        "fromFrame": start_frame,
                        "toFrame": frame_index,
                        "heightBefore": before,
                        "heightAtFloor": min(
                            h for j, h in series if start_frame <= j <= frame_index
                        ),
                        "frames": frame_index - start_frame,
                    }
                )
                open_drop = None
            elif frame_index - start_frame > RECOVERY_FRAMES:
                detail.append(
                    {
                        "block": index,
                        "fromFrame": start_frame,
                        "toFrame": None,
                        "heightBefore": before,
                        "heightAtFloor": height,
                        "frames": None,
                    }
                )
                open_drop = None

    dips = 0
    for i in range(1, len(frames)):
        drop = frames[i - 1]["scrollHeight"] - frames[i]["scrollHeight"]
        if drop < 300:
            continue
        for j in range(i + 1, min(i + RECOVERY_FRAMES, len(frames))):
            if frames[j]["scrollHeight"] >= frames[i - 1]["scrollHeight"] - 50:
                dips += 1
                break

    anchor_shift = 0.0
    for i in range(1, len(frames)):
        previous, current = frames[i - 1], frames[i]
        if previous["anchorTop"] is None or current["anchorTop"] is None:
            continue
        # Document space. The viewport scrolling under the anchor is not the anchor moving.
        moved = abs(
            (current["anchorTop"] + current["scrollTop"])
            - (previous["anchorTop"] + previous["scrollTop"])
        )
        anchor_shift = max(anchor_shift, moved)

    return {
        "frames": len(frames),
        "blocks": block_count,
        "collapses": collapses,
        "placeholderFrames": placeholder_frames,
        "scrollHeightDips": dips,
        "anchorShiftPx": round(anchor_shift, 1),
        "worstDropPx": round(worst_drop_px, 1),
        "detail": detail[:12],
    }


def analyse_sweep(frames: list[dict]) -> dict:
    """Layout shift under a scroll gesture, from the same frame log.

    A block's `tops` entry is its distance from the top of the THREAD'S CONTENT, not from the
    viewport, so scrolling does not move it. Anything that does move it is a block above it
    changing size, which is what the user sees as the page moving under their finger.
    """
    shift_frames = 0
    worst_shift = 0.0
    for i in range(1, len(frames)):
        previous, current = frames[i - 1], frames[i]
        moved = 0.0
        for index in range(min(len(previous["tops"]), len(current["tops"]))):
            moved = max(moved, abs(current["tops"][index] - previous["tops"][index]))
        if moved > SHIFT_PX:
            shift_frames += 1
        worst_shift = max(worst_shift, moved)
    heights = [f["scrollHeight"] for f in frames]
    return {
        "sweepFrames": len(frames),
        "shiftFrames": shift_frames,
        "worstShiftPx": round(worst_shift, 1),
        "scrollHeightMin": min(heights) if heights else -1,
        "scrollHeightMax": max(heights) if heights else -1,
        "scrollHeightGrowthPx": (max(heights) - min(heights)) if heights else -1,
    }
