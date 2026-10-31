# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Computes flicker verdicts from frame logs without playwright, so the CPU contract test can run."""

from __future__ import annotations

# Only blocks this tall count, so a genuinely short fence is never read as collapsed.
TALL_PX = 400
# Streamdown's inline fallback is 200px plus the wrapper's padding and header row.
PLACEHOLDER_LO, PLACEHOLDER_HI = 150, 300
# Frames a drop may take to recover; staying short is a different bug.
RECOVERY_FRAMES = 240
SHIFT_PX = 8


def analyse_stream(frames: list[dict]) -> dict:
    """A flicker is a block at least TALL_PX tall that drops to half TALL_PX or less, then recovers."""
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
            # A collapse can deepen after it opens, so track the worst drop while it stays open.
            worst_drop_px = max(worst_drop_px, before - height)
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

        # The log tail is shorter than RECOVERY_FRAMES, so record a drop still open at the end.
        if open_drop is not None:
            start_frame, before = open_drop
            detail.append(
                {
                    "block": index,
                    "fromFrame": start_frame,
                    "toFrame": None,
                    "heightBefore": before,
                    "heightAtFloor": min(h for j, h in series if j >= start_frame),
                    "frames": None,
                }
            )

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
    """Tops are measured from thread content, so a scroll cannot move them; only a resize above can."""
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
