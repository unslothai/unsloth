# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Real-browser regression gate for the opt-in, bounded Agenten-Chatverlauf.

Run ONLY against an isolated disposable Studio instance, not a production installation:
    python -m tests.studio.playwright_agent_history_scroll \
        --url http://127.0.0.1:5401 --username unsloth --password '...' --messages 400

Requires the repository's Playwright/Chromium test prerequisites and a running backend.
This test writes and then deletes ONE uniquely named fixture thread on that instance.
Missing prerequisites, auth failures, missing virtualizer and any incomplete traversal
are hard failures, not skipped or counted as passes. It checks bounded find/reveal
navigation, but not clipboard/export parity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Executed as a standalone Playwright driver from the checkout root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.studio.studiobench.runtime.lifecycle import (
    authenticate,
    auth_request_json,
    seed_init_script,
)

ROW = "[data-agent-history-row]"
LIST = '[data-agent-history-scroll-list="true"]'
VIEWPORT = ".aui-thread-viewport"
MAX_ROWS = 96


def create_fixture(base: str, auth, count: int) -> str:
    thread_id = str(uuid.uuid4())
    now = int(time.time() * 1000) - count * 1000
    auth_request_json(
        auth,
        f"{base}/api/chat/threads",
        method = "POST",
        body = {
            "id": thread_id,
            "title": "agent-history-e2e-" + thread_id,
            "modelType": "base",
            "modelId": "agent-history-fixture",
            "createdAt": now,
        },
    )
    messages = []
    parent = None
    for index in range(count):
        message_id = str(uuid.uuid4())
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": parent,
                "role": "user" if index % 2 == 0 else "assistant",
                "content": [
                    {"type": "text", "text": f"agent-history-e2e-row-{index:05d} unique-fixture"}
                ],
                "attachments": None,
                "metadata": None,
                "createdAt": now + index * 1000,
            }
        )
        parent = message_id
    auth_request_json(
        auth,
        f"{base}/api/chat/threads/{thread_id}/messages",
        method = "PUT",
        timeout = 120,
        body = {"messages": messages, "pruneMissing": True},
    )
    persisted = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
    if len(persisted.get("messages", [])) != count:
        raise AssertionError("The backend did not persist the complete fixture")
    return thread_id


def census(page, expected: int, where: str) -> dict:
    state = page.evaluate("""() => {
      const rows = Array.from(document.querySelectorAll('[data-agent-history-row]'));
      const list = document.querySelector('[data-agent-history-scroll-list="true"]');
      const viewport = document.querySelector('.aui-thread-viewport');
      return {
        hasList: !!list, mounted: rows.length,
        ordinal: rows.map(row => Number(row.getAttribute('aria-posinset'))),
        sizes: rows.map(row => Number(row.getAttribute('aria-setsize'))),
        viewport: !!viewport,
        scrollTop: viewport?.scrollTop ?? null,
        scrollHeight: viewport?.scrollHeight ?? null
      };
    }""")
    if not state["hasList"] or not state["viewport"]:
        raise AssertionError(f"{where}: active agent renderer/viewport absent: {state}")
    if not (1 <= state["mounted"] <= MAX_ROWS):
        raise AssertionError(f"{where}: mounted rows not bounded: {state}")
    ordinals = state["ordinal"]
    if (
        len(set(ordinals)) != len(ordinals)
        or ordinals != list(range(ordinals[0], ordinals[-1] + 1))
        or any(size != expected for size in state["sizes"])
        or not (1 <= ordinals[0] <= ordinals[-1] <= expected)
    ):
        raise AssertionError(f"{where}: invalid ARIA positions/total: {state}")
    return state


def scroll_to(page, fraction: float, expected: int, label: str) -> dict:
    # Like the shipped Studio readiness probe, scroll + user wheel intent together.
    # A one-shot programmatic jump alone can be cancelled by intent-aware autoscroll.
    viewport = page.locator(VIEWPORT)
    viewport.hover()
    for _ in range(12):
        page.mouse.wheel(0, -1800 if fraction == 0 else 1800)
        page.evaluate(
            """fraction => {
          const el = document.querySelector('.aui-thread-viewport');
          el.scrollTop = (el.scrollHeight - el.clientHeight) * fraction;
          el.dispatchEvent(new Event('scroll'));
        }""",
            fraction,
        )
        page.wait_for_timeout(100)
    state = census(page, expected, label)
    first, last = state["ordinal"][0], state["ordinal"][-1]
    if fraction == 0 and first != 1:
        raise AssertionError(f"{label}: first row not reachable: {state}")
    if fraction == 1 and last != expected:
        raise AssertionError(f"{label}: final row not reachable: {state}")
    if fraction == 0.5 and not (expected // 4 <= first <= 3 * expected // 4):
        raise AssertionError(f"{label}: middle of the transcript not reached: {state}")
    return state


def search_snapshot(page) -> dict:
    """Read-only diagnostics. A missing search result is always a hard failure."""
    return page.evaluate("""() => {
      const bar = document.querySelector('[data-find-bar-layer]');
      const input = bar?.querySelector('input');
      const counter = bar?.querySelector('[aria-live="polite"]');
      const viewport = document.querySelector('.aui-thread-viewport');
      const rows = [...document.querySelectorAll('[data-agent-history-row]')];
      const marked = [...document.querySelectorAll('[data-agent-history-find-active="true"]')];
      return {
        query: input?.value ?? null,
        counter: counter?.textContent?.trim() ?? null,
        focused: document.activeElement === input,
        barVisible: !!bar && !bar.hidden,
        currentRows: [rows[0]?.getAttribute('data-agent-history-row') ?? null,
                      rows.at(-1)?.getAttribute('data-agent-history-row') ?? null],
        mounted: rows.length,
        searchHitRows: marked.map(row => row.getAttribute('data-agent-history-row')),
        scrollTop: viewport?.scrollTop ?? null,
        scrollHeight: viewport?.scrollHeight ?? null,
        pageErrors: window.__agentSearchErrors ?? null
      };
    }""")


def require_search_state(
    page,
    script: str,
    label: str,
    timeout: int = 15_000,
    arg = None,
) -> None:
    try:
        page.wait_for_function(script, arg = arg, timeout = timeout)
    except Exception as exc:
        raise AssertionError(f"{label}: {search_snapshot(page)}") from exc


def require_visible_search_row(page, row_index: int, label: str) -> None:
    """Indexing + DOM mounting is not enough: the hit must intersect the viewport."""
    require_search_state(
        page,
        """index => {
          const viewport = document.querySelector('.aui-thread-viewport');
          const row = document.querySelector(
            '[data-agent-history-row="' + index +
            '"][data-agent-history-find-active="true"]'
          );
          if (!viewport || !row || !viewport.contains(row)) return false;
          const view = viewport.getBoundingClientRect();
          const hit = row.getBoundingClientRect();
          return hit.width > 0 && hit.height > 0 &&
            hit.bottom > view.top + 8 && hit.top < view.bottom - 8 &&
            document.querySelectorAll('[data-agent-history-row]').length <= 96;
        }""",
        label,
        arg = row_index,
    )


def check_adaptive_thumb_drag(page, level: str) -> None:
    """Actual pointer drag; no wheel or programmatic scroll recovery allowed."""
    page.add_style_tag(
        content = (
            ".aui-thread-viewport{overflow-y:scroll!important;scrollbar-gutter:stable!important}"
            ".aui-thread-viewport::-webkit-scrollbar{width:16px}"
            ".aui-thread-viewport::-webkit-scrollbar-track{background:#ddd}"
            ".aui-thread-viewport::-webkit-scrollbar-thumb{background:#888}"
        )
    )
    # Set up the scrollbar thumb at the bottom before the real pointer drag.
    # No synthetic scroll or wheel is used AFTER the drag to repair the view.
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(100)
    box = page.locator(VIEWPORT).bounding_box()
    if box is None:
        raise AssertionError("Agent viewport missing")
    metrics = page.locator(VIEWPORT).evaluate(
        """el => ({
          scrollTop: el.scrollTop, scrollHeight: el.scrollHeight,
          clientHeight: el.clientHeight, clientWidth: el.clientWidth,
          offsetWidth: el.offsetWidth, offsetHeight: el.offsetHeight,
          scrollbarGutter: getComputedStyle(el).scrollbarGutter
        })"""
    )
    track_width = metrics["offsetWidth"] - metrics["clientWidth"]
    if track_width < 8 or metrics["scrollHeight"] <= metrics["clientHeight"]:
        raise AssertionError(f"{level}: native scrollbar not testable: {metrics}")
    # Chromium honours the app's scrollbar-width:thin over the injected
    # ::-webkit-scrollbar width. With a 10px gutter its bottommost ~12px
    # are NOT a draggable thumb (a bare Chromium pointer reproduction confirmed
    # y=height-6 does nothing while y=height-18 drags normally). On a 16px
    # normal scrollbar the short thumb IS at height-6; height-18 can hit the
    # track instead. Select the inset using the measured *native* gutter.
    x = box["x"] + box["width"] - track_width / 2
    horizontal_gutter = max(0, metrics["offsetHeight"] - metrics["clientHeight"])
    thumb_inset = 18 if track_width <= 12 else 6
    bottom = box["y"] + box["height"] - horizontal_gutter - thumb_inset
    destination = box["y"] + box["height"] * 0.35
    hit = page.evaluate(
        """([x, y]) => {
          const el = document.elementFromPoint(x, y);
          const view = document.querySelector('.aui-thread-viewport');
          return {
            hitTag: el?.tagName, hitClass: String(el?.className ?? ''),
            viewportHit: el === view || !!(view && el && view.contains(el)),
            nativeScrollbarWidth: view ? getComputedStyle(view, '::-webkit-scrollbar').width : null,
            devicePixelRatio: devicePixelRatio
          };
        }""",
        [x, bottom],
    )
    if not hit["viewportHit"]:
        raise AssertionError(f"{level}: scrollbar pointer occluded: {hit}; at ({x}, {bottom})")
    before = metrics["scrollTop"]
    page.mouse.move(x, bottom)
    page.mouse.down()
    page.mouse.move(x, destination, steps = 10)
    page.mouse.up()
    after = page.locator(VIEWPORT).evaluate("el => el.scrollTop")
    if abs(after - before) < box["height"]:
        detail = (
            f"{level}: physical scrollbar drag did not move the viewport: "
            f"{before} -> {after}; scrollbar: {metrics}; hit: {hit}; at ({x}, {bottom})"
        )
        raise AssertionError(detail)
    try:
        page.wait_for_function(
            """() => {
              const viewport = document.querySelector('.aui-thread-viewport');
              if (!viewport) return false;
              const view = viewport.getBoundingClientRect();
              return [...viewport.querySelectorAll('[data-agent-history-row]')]
                .some(row => {
                  const r = row.getBoundingClientRect();
                  return r.bottom > view.top + 8 && r.top < view.bottom - 8;
                });
            }""",
            timeout = 3_000,
        )
    except Exception as exc:
        geometry = page.evaluate(
            """() => {
              const viewport = document.querySelector('.aui-thread-viewport');
              const list = viewport?.querySelector('[data-agent-history-scroll-list]');
              const top = list?.querySelector('[data-agent-history-spacer="top"]');
              const bottom = list?.querySelector('[data-agent-history-spacer="bottom"]');
              const rows = [...(list?.querySelectorAll('[data-agent-history-row]') ?? [])];
              const rect = el => {
                if (!el) return null;
                const r = el.getBoundingClientRect();
                return { top: r.top, bottom: r.bottom, height: r.height };
              };
              return {
                scrollTop: viewport?.scrollTop ?? null,
                scrollHeight: viewport?.scrollHeight ?? null,
                clientHeight: viewport?.clientHeight ?? null,
                viewport: rect(viewport),
                list: rect(list),
                topSpacer: rect(top),
                bottomSpacer: rect(bottom),
                rows: rows.map(row => ({
                  index: Number(row.dataset.agentHistoryRow),
                  rect: rect(row)
                }))
              };
            }"""
        )
        raise AssertionError(
            f"{level}: physical drag moved {before} -> {after}, but no row "
            f"intersects the viewport after 3000ms: "
            f"{json.dumps(geometry, sort_keys = True)}"
        ) from exc

    first = page.evaluate(
        """() => {
          const viewport = document.querySelector('.aui-thread-viewport');
          const v = viewport.getBoundingClientRect();
          const row = [...viewport.querySelectorAll('[data-agent-history-row]')]
            .find(el => el.getBoundingClientRect().bottom > v.top + 8
              && el.getBoundingClientRect().top < v.bottom - 8);
          return row && { index: row.dataset.agentHistoryRow,
            y: row.getBoundingClientRect().top };
        }"""
    )
    page.wait_for_timeout(650)
    second = page.evaluate(
        """index => {
          const row = document.querySelector(
            '[data-agent-history-row="' + index + '"]');
          return row ? row.getBoundingClientRect().top : null;
        }""",
        first["index"],
    )
    if second is None or abs(second - first["y"]) > 5:
        raise AssertionError(f"{level}: text shifted after drag without input: {first} -> {second}")

    # A native thumb dragged ALL the way to the track origin must reach the
    # first logical message in ONE gesture, even when measured heights change.
    # Earlier tests only targeted 35% of the track and could not detect this.
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(120)
    page.mouse.move(x, bottom)
    page.mouse.down()
    page.mouse.move(x, box["y"] + 2, steps = 18)
    page.mouse.up()
    page.wait_for_function(
        """() => {
          const viewport = document.querySelector('.aui-thread-viewport');
          const first = viewport?.querySelector('[data-agent-history-row="0"]');
          if (!viewport || !first) return false;
          const view = viewport.getBoundingClientRect();
          const bounds = first.getBoundingClientRect();
          return viewport.scrollTop <= 2 &&
            bounds.bottom > view.top + 1 && bounds.top < view.bottom - 1;
        }""",
        timeout = 5_000,
    )
    at_top = page.locator(VIEWPORT).evaluate(
        """el => ({
          scrollTop: el.scrollTop,
          windowStart: Number(el.querySelector('[data-agent-history-scroll-list]')?.dataset.agentHistoryQaWindowStart ?? -1),
          firstVisible: !!el.querySelector('[data-agent-history-row="0"]')
        })"""
    )
    page.wait_for_timeout(650)
    settled_top = page.locator(VIEWPORT).evaluate(
        """el => ({
          scrollTop: el.scrollTop,
          windowStart: Number(el.querySelector('[data-agent-history-scroll-list]')?.dataset.agentHistoryQaWindowStart ?? -1),
          firstVisible: !!el.querySelector('[data-agent-history-row="0"]')
        })"""
    )
    if (
        at_top["windowStart"] != 0
        or settled_top["windowStart"] != 0
        or not settled_top["firstVisible"]
        or abs(settled_top["scrollTop"] - at_top["scrollTop"]) > 2
    ):
        raise AssertionError(
            f"{level}: full native thumb drag failed to settle at transcript origin: "
            f"{at_top} -> {settled_top}"
        )


def check_native_thumb_track_stability(page, level: str) -> None:
    """One continuous native drag should keep the thumb under the mouse.

    Deliberately sample after every pointer step while the button is STILL
    down. A previous endpoint-only assertion missed geometry drift.
    """
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(150)
    box = page.locator(VIEWPORT).bounding_box()
    if box is None:
        raise AssertionError("Native thumb viewport missing")
    initial = page.locator(VIEWPORT).evaluate(
        """el => ({
          scrollHeight: el.scrollHeight, scrollTop: el.scrollTop,
          height: el.clientHeight, width: el.offsetWidth - el.clientWidth,
          horizontalGutter: el.offsetHeight - el.clientHeight
        })"""
    )
    if initial["width"] < 8 or initial["scrollHeight"] <= initial["height"]:
        raise AssertionError(f"{level}: no physical scrollbar: {initial}")
    x = box["x"] + box["width"] - initial["width"] / 2
    inset = 18 if initial["width"] <= 12 else 6
    start_y = box["y"] + box["height"] - initial["horizontalGutter"] - inset

    def thumb_top(scroll_top: float, scroll_height: float) -> float:
        vh = initial["height"]
        thumb_height = max(18, vh * vh / scroll_height)
        travel = max(0, vh - thumb_height)
        return scroll_top / max(1, scroll_height - vh) * travel

    initial_thumb_top = thumb_top(initial["scrollTop"], initial["scrollHeight"])
    samples = []
    page.mouse.move(x, start_y)
    page.mouse.down()
    try:
        for fraction in (0.83, 0.70, 0.57, 0.44, 0.31, 0.18):
            target_y = box["y"] + box["height"] * fraction
            page.mouse.move(x, target_y, steps = 2)
            page.wait_for_timeout(35)
            point = page.locator(VIEWPORT).evaluate(
                """el => ({scrollHeight: el.scrollHeight, scrollTop: el.scrollTop})"""
            )
            # Delta-only comparison cancels the unknown native grab offset.
            thumb_delta = initial_thumb_top - thumb_top(point["scrollTop"], point["scrollHeight"])
            pointer_delta = start_y - target_y
            samples.append(
                {
                    "finger_y": round(target_y, 2),
                    "scroll_height": point["scrollHeight"],
                    "thumb_gap_px": round(abs(pointer_delta - thumb_delta), 2),
                }
            )
    finally:
        page.mouse.up()
    worst_height_change = max(
        abs(sample["scroll_height"] - initial["scrollHeight"]) for sample in samples
    )
    worst_gap = max(sample["thumb_gap_px"] for sample in samples)
    # A 400-message history is about 280k virtual pixels. Permit only a
    # small intrinsic layout/scrollbar-gutter variance during native dragging.
    # Native Chromium can briefly lag pointer steps even for an ordinary,
    # NON-VIRTUALIZED 260k-px element. The virtual document length is our
    # deterministic invariant; native thumb/pointer lag is QA telemetry.
    print(
        json.dumps(
            {
                "case": "native-thumb-track-stability",
                "level": level,
                "virtual_height_start": initial["scrollHeight"],
                "virtual_height_delta_px": worst_height_change,
                "native_thumb_gap_observed_px": worst_gap,
                "samples": samples,
            },
            sort_keys = True,
        ),
        flush = True,
    )
    if worst_height_change > max(144, initial["scrollHeight"] * 0.006):
        raise AssertionError(
            f"{level}: scrollbar virtual document changed during drag: "
            f"height_delta={worst_height_change}, gap_observed={worst_gap}, samples={samples}"
        )


def prepare_native_thumb_from_latest(page, level: str) -> None:
    """Restore the real latest ROW WINDOW, not only the physical scrollTop.

    Directly assigning scrollTop at the end of a long run does not authorize
    React to mount the tail after a consumed reader gesture. The toolbar's
    genuine back-to-latest command is the correct way to reset the virtualizer
    before a new independent scrollbar test.
    """
    latest = page.locator('[data-agent-history-scroll-list="true"] > button')
    if latest.count() > 0:
        latest.click(timeout = 5_000)
    page.wait_for_function(
        """() => {
          const l = document.querySelector('[data-agent-history-scroll-list="true"]');
          return l && Number(l.dataset.agentHistoryQaWindowEnd) === 400;
        }""",
        timeout = 6_000,
    )
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(150)


def check_native_thumb_live_follow(page, level: str) -> None:
    """A SINGLE held native scrollbar gesture must update mounted rows live.

    Chromium captures the thumb outside the DOM pointermove stream. Verify
    the actual row intersects the viewport at two distant positions BEFORE
    pointerup; an eventual release-only seek cannot make this test pass.
    """
    prepare_native_thumb_from_latest(page, level)
    box = page.locator(VIEWPORT).bounding_box()
    if box is None:
        raise AssertionError(f"{level}: live thumb viewport missing")
    metrics = page.locator(VIEWPORT).evaluate(
        """el => ({
          gutter: el.offsetWidth - el.clientWidth,
          horizontalGutter: el.offsetHeight - el.clientHeight,
          start: Number(el.querySelector('[data-agent-history-scroll-list]')
            ?.dataset.agentHistoryQaWindowStart ?? -1)
        })"""
    )
    if metrics["gutter"] < 8 or metrics["start"] < 200:
        raise AssertionError(f"{level}: missing native thumb/start window: {metrics}")
    x = box["x"] + box["width"] - metrics["gutter"] / 2
    inset = 18 if metrics["gutter"] <= 12 else 6
    y = box["y"] + box["height"] - metrics["horizontalGutter"] - inset
    samples = []
    page.mouse.move(x, y)
    page.mouse.down()
    try:
        for fraction in (0.55, 0.18):
            target_y = box["y"] + box["height"] * fraction
            page.mouse.move(x, target_y, steps = 18)
            # The button must remain down throughout this assertion.
            page.wait_for_function(
                """limit => {
                  const v = document.querySelector('.aui-thread-viewport');
                  const list = v?.querySelector('[data-agent-history-scroll-list]');
                  if (!v || !list) return false;
                  const rangeStart = Number(list.dataset.agentHistoryQaWindowStart);
                  const box = v.getBoundingClientRect();
                  const rows = [...list.querySelectorAll('[data-agent-history-row]')];
                  return rangeStart < limit && rows.some(row => {
                    const r = row.getBoundingClientRect();
                    return r.bottom > box.top + 8 && r.top < box.bottom - 8;
                  });
                }""",
                arg = 280 if fraction > 0.5 else 150,
                timeout = 4_000,
            )
            sample = page.locator(VIEWPORT).evaluate(
                """el => {
                  const list = el.querySelector('[data-agent-history-scroll-list]');
                  const v = el.getBoundingClientRect();
                  return {
                    top: el.scrollTop,
                    windowStart: Number(list.dataset.agentHistoryQaWindowStart),
                    windowEnd: Number(list.dataset.agentHistoryQaWindowEnd),
                    mounted: list.querySelectorAll('[data-agent-history-row]').length,
                    visible: [...list.querySelectorAll('[data-agent-history-row]')]
                      .filter(row => {
                        const r = row.getBoundingClientRect();
                        return r.bottom > v.top + 8 && r.top < v.bottom - 8;
                      }).map(row => Number(row.dataset.agentHistoryRow)),
                  };
                }"""
            )
            if not sample["visible"] or sample["mounted"] > 32:
                raise AssertionError(f"{level}: bad live thumb coverage while held: {sample}")
            samples.append(sample)
    finally:
        page.mouse.up()
    if not samples[1]["windowStart"] < samples[0]["windowStart"] - 20:
        raise AssertionError(f"{level}: live windows failed to chase held thumb: {samples}")
    print(
        json.dumps(
            {"case": "native-thumb-live-follow", "level": level, "samples": samples},
            sort_keys = True,
        ),
        flush = True,
    )


def check_native_thumb_drift_release(page, level: str) -> None:
    """Keep the painted native row when pointerup reports a divergent Y.

    First drag the real Chromium scrollbar and wait for the physical viewport
    and mounted rows to settle WHILE held. Then deliver an intentionally
    displaced pointerup coordinate to the real JS event listener. Chromium's
    captured scrollbar can otherwise undo a synthetic scrollTop change during
    a held gesture, making such a test timing-dependent. Finish with a real
    mouse.up() to release Chromium's native pointer capture.
    """
    prepare_native_thumb_from_latest(page, level)
    box = page.locator(VIEWPORT).bounding_box()
    if box is None:
        raise AssertionError(f"{level}: release-drift viewport missing")
    metrics = page.locator(VIEWPORT).evaluate(
        """el => ({
          gutter: el.offsetWidth - el.clientWidth,
          horizontalGutter: el.offsetHeight - el.clientHeight
        })"""
    )
    if metrics["gutter"] < 8:
        raise AssertionError(f"{level}: release-drift native scrollbar missing")
    x = box["x"] + box["width"] - metrics["gutter"] / 2
    inset = 18 if metrics["gutter"] <= 12 else 6
    start_y = box["y"] + box["height"] - metrics["horizontalGutter"] - inset
    native_y = box["y"] + box["height"] * 0.24
    divergent_pointer_y = box["y"] + box["height"] * 0.65

    def displayed_state():
        return page.locator(VIEWPORT).evaluate(
            """el => {
              const view = el.getBoundingClientRect();
              const rows = [...el.querySelectorAll('[data-agent-history-row]')];
              const visible = rows.filter(row => {
                const rect = row.getBoundingClientRect();
                return rect.bottom > view.top + 8 && rect.top < view.bottom - 8;
              });
              return {
                top: el.scrollTop, height: el.scrollHeight,
                visibleRows: visible.map(row => Number(row.dataset.agentHistoryRow))
              };
            }"""
        )

    page.evaluate(
        """() => {
          document.querySelector('.aui-thread-viewport').addEventListener(
            'pointerdown', event => { window.__qaThumbPointerId = event.pointerId; },
            { once: true }
          );
        }"""
    )
    page.mouse.move(x, start_y)
    page.mouse.down()
    before = None
    try:
        page.mouse.move(x, native_y, steps = 18)
        page.wait_for_function(
            """() => {
              const v = document.querySelector('.aui-thread-viewport');
              if (!v) return false;
              const box = v.getBoundingClientRect();
              const visible = [...v.querySelectorAll('[data-agent-history-row]')]
                .some(row => {
                  const r = row.getBoundingClientRect();
                  return r.bottom > box.top + 8 && r.top < box.bottom - 8;
                });
              const fraction = v.scrollTop / Math.max(1, v.scrollHeight - v.clientHeight);
              return visible && fraction > 0.07 && fraction < 0.43;
            }""",
            timeout = 4_000,
        )
        before = displayed_state()
        if not before["visibleRows"]:
            raise AssertionError(f"{level}: no mounted content during native drag: {before}")
        # Only the release event's clientY is displaced. No programmatic
        # scrollTop write or mount operation intervenes here.
        page.evaluate(
            """releaseY => {
              const pointerId = window.__qaThumbPointerId;
              if (!Number.isInteger(pointerId)) throw Error('Native thumb pointerdown absent');
              window.dispatchEvent(new PointerEvent('pointerup', {
                pointerId, pointerType: 'mouse', button: 0,
                clientY: releaseY, bubbles: true
              }));
              delete window.__qaThumbPointerId;
            }""",
            divergent_pointer_y,
        )
    finally:
        page.mouse.up()
    page.wait_for_timeout(650)
    after = displayed_state()
    displacement = (
        abs(after["visibleRows"][0] - before["visibleRows"][0]) if after["visibleRows"] else 400
    )
    print(
        json.dumps(
            {
                "case": "native-thumb-divergent-pointer-release",
                "level": level,
                "before": before,
                "after": after,
                "row_displacement": displacement,
            },
            sort_keys = True,
        ),
        flush = True,
    )
    if displacement > 24:
        raise AssertionError(
            f"{level}: pointer-up snapped away from native thumb position: {before} -> {after}"
        )


def check_native_thumb_release_telemetry(page, level: str) -> None:
    """Record native position immediately before/after release on real Chromium.

    No synthetic scroll or wheel recovery is issued. Measurements are
    observational, not hard timing gates: thumb capture and scheduling are
    browser-specific. A visible message after release is mandatory.
    """
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(150)
    box = page.locator(VIEWPORT).bounding_box()
    if box is None:
        raise AssertionError(f"{level}: viewport disappeared during release QA")
    metrics = page.locator(VIEWPORT).evaluate(
        """el => ({
          gutter: el.offsetWidth - el.clientWidth,
          horizontalGutter: el.offsetHeight - el.clientHeight
        })"""
    )
    if metrics["gutter"] < 8:
        raise AssertionError(f"{level}: native scrollbar gutter disappeared")
    x = box["x"] + box["width"] - metrics["gutter"] / 2
    inset = 18 if metrics["gutter"] <= 12 else 6
    start_y = box["y"] + box["height"] - metrics["horizontalGutter"] - inset
    target_y = box["y"] + box["height"] * 0.5
    page.mouse.move(x, start_y)
    page.mouse.down()
    page.mouse.move(x, target_y, steps = 18)
    page.wait_for_timeout(250)
    before = page.locator(VIEWPORT).evaluate(
        """el => ({ top: el.scrollTop, height: el.scrollHeight })"""
    )
    page.mouse.up()
    page.wait_for_timeout(750)
    after = page.locator(VIEWPORT).evaluate(
        """el => {
          const view = el.getBoundingClientRect();
          const rows = [...el.querySelectorAll('[data-agent-history-row]')];
          const visible = rows.filter(row => {
            const rect = row.getBoundingClientRect();
            return rect.bottom > view.top + 8 && rect.top < view.bottom - 8;
          });
          return {
            top: el.scrollTop, height: el.scrollHeight,
            visibleRows: visible.map(row => Number(row.dataset.agentHistoryRow)),
          };
        }"""
    )
    print(
        json.dumps(
            {
                "case": "native-thumb-release-telemetry",
                "level": level,
                "before_release": before,
                "after_settle": after,
                "delta_scroll_top": round(after["top"] - before["top"], 2),
            },
            sort_keys = True,
        ),
        flush = True,
    )
    if not after["visibleRows"]:
        raise AssertionError(f"{level}: release left a blank viewport: {before} -> {after}")


def check_dynamic_thumb_under_heavy_dom(page, level: str) -> None:
    """Stress the real renderer with a 15k-descendant tool message before dragging."""
    # Two closely spaced synthetic tool-output resizes exercise the
    # trailing one-second recount; neither resize may lose the target.
    target = page.evaluate(
        """() => {
          const row = [...document.querySelectorAll('[data-agent-history-row]')].at(-1);
          const owner = row?.querySelector('[data-role]');
          if (!row || !owner) throw Error('Adaptive target missing');
          const heavy = document.createElement('div');
          heavy.dataset.qaAdaptiveHeavy = 'true';
          heavy.style.cssText = 'height:100px;overflow:hidden;contain:layout';
          heavy.innerHTML = '<span>qa</span>'.repeat(200);
          owner.appendChild(heavy);
          return Number(row.dataset.agentHistoryRow);
        }"""
    )
    page.wait_for_timeout(100)
    page.evaluate(
        """() => {
          const heavy = document.querySelector('[data-qa-adaptive-heavy]');
          if (!heavy) throw Error('Synthetic row disappeared before expansion');
          heavy.style.height = '1600px';
          heavy.innerHTML = '<span>qa</span>'.repeat(15000);
        }"""
    )
    page.wait_for_function(
        """target => {
          const rows = [...document.querySelectorAll('[data-agent-history-row]')];
          return rows.length === 1 &&
            Number(rows[0].dataset.agentHistoryRow) === target &&
            rows[0].querySelector('[data-qa-adaptive-heavy]') !== null &&
            rows[0].querySelectorAll('*').length > 15000;
        }""",
        arg = target,
        timeout = 15_000,
    )
    page.evaluate("() => document.querySelector('[data-qa-adaptive-heavy]')?.remove()")
    check_adaptive_thumb_drag(page, level)


def check_dynamic_streaming_tail_visibility(page, level: str, messages: int) -> None:
    """During tail-follow, repeated streaming-style DOM mutations must not blank the viewport.

    Synthetic tool markup is added only to the currently mounted row in an
    isolated browser. The persisted fixture and assistant-ui message state are
    unchanged. Read the painted layout on successive frames, not just the
    mounted count, and do not use a wheel/scroll event to rescue a blank view.
    """
    page.locator(VIEWPORT).evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(120)
    last_row = page.locator(f'[data-agent-history-row="{messages - 1}"]')
    last_row.wait_for(state = "attached", timeout = 15_000)
    # Clicking a visible message must not turn subsequent auto-follow layout
    # changes into reader navigation; no wheel or scrollbar drag follows.
    last_row.click(position = {"x": 20, "y": 20})
    observations = page.evaluate(
        """async expected => {
          const viewport = document.querySelector('.aui-thread-viewport');
          const last = viewport?.querySelector(
            '[data-agent-history-row="' + (expected - 1) + '"]'
          );
          const owner = last?.querySelector('[data-role]');
          if (!viewport || !last || !owner) {
            throw Error('stream-tail: last row or viewport missing');
          }
          const fixture = document.createElement('div');
          fixture.dataset.qaStreamingTail = 'true';
          fixture.style.cssText = 'overflow:hidden;contain:layout;height:120px';
          owner.appendChild(fixture);
          const twoFrames = () => new Promise(resolve =>
            requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const samples = [];
          try {
            for (let i = 0; i < 44; i++) {
              fixture.style.height = (120 + (i % 7) * 185) + 'px';
              fixture.innerHTML = '<span>synthetic tool output</span>'.repeat(
                i % 2 ? 700 : 100
              );
              await twoFrames();
              const view = viewport.getBoundingClientRect();
              const rows = [...viewport.querySelectorAll('[data-agent-history-row]')];
              const tail = rows.find(row =>
                Number(row.dataset.agentHistoryRow) === expected - 1);
              const rect = tail?.getBoundingClientRect();
              const visible = !!rect && rect.width > 0 && rect.height > 0 &&
                rect.bottom > view.top + 8 && rect.top < view.bottom - 8;
              samples.push({
                step: i, mounted: rows.length,
                first: Number(rows[0]?.dataset.agentHistoryRow ?? -1),
                last: Number(rows.at(-1)?.dataset.agentHistoryRow ?? -1),
                visible, scrollTop: Math.round(viewport.scrollTop),
                scrollHeight: Math.round(viewport.scrollHeight)
              });
              if (!visible) break;
            }
          } finally {
            fixture.remove();
          }
          return samples;
        }""",
        messages,
    )
    bad = next((sample for sample in observations if not sample["visible"]), None)
    if bad is not None or len(observations) != 44:
        raise AssertionError(
            f"{level}: streaming tail became invisible without reader scroll: "
            f"{json.dumps({'failure': bad, 'samples': len(observations)}, sort_keys = True)}"
        )


def check_isolated_dynamic_thumb_modes(page, thread_id: str, messages: int) -> None:
    """Test the native drag independently of Radix menu focus and mode selection.

    Persisted per-thread preferences use the production Zustand hydration path.
    The complete menu-selection E2E remains in check_chat_header_menu.
    """
    for level, policy, cap in (
        ("dynamic", "medium", 16),
        ("dynamic-high", "medium", 16),
        ("dynamic-low", "medium", 16),
    ):
        page.evaluate(
            """([threadId, selected]) => {
              const key = 'unsloth_agent_chat_history_v1';
              const stored = JSON.parse(localStorage.getItem(key) || '{}');
              const state = stored.state || {};
              state.enabledThreads = { ...(state.enabledThreads || {}), [threadId]: true };
              state.disabledThreads = { ...(state.disabledThreads || {}) };
              delete state.disabledThreads[threadId];
              state.domLevelByThreadId = {
                ...(state.domLevelByThreadId || {}),
                [threadId]: selected
              };
              localStorage.setItem(key, JSON.stringify({ ...stored, state }));
            }""",
            [thread_id, level],
        )
        # Reload forces the real persisted policy -> renderer activation path.
        page.reload(wait_until = "domcontentloaded", timeout = 120_000)
        page.wait_for_function(
            """([expected, actualPolicy, maximum]) => {
              const list = document.querySelector('[data-agent-history-scroll-list="true"]');
              const rows = [...document.querySelectorAll('[data-agent-history-row]')];
              return list?.getAttribute('data-agent-history-dom-level') === actualPolicy &&
                rows.length > 0 && rows.length <= maximum &&
                rows.every(row => row.getAttribute('aria-setsize') === String(expected));
            }""",
            arg = [messages, policy, cap],
            timeout = 60_000,
        )
        census(page, messages, f"isolated-dynamic-{level}")
        check_dynamic_streaming_tail_visibility(page, level, messages)
        check_dynamic_thumb_under_heavy_dom(page, level)
        check_native_thumb_track_stability(page, level)
        check_native_thumb_live_follow(page, level)
        check_native_thumb_drift_release(page, level)
        check_native_thumb_release_telemetry(page, level)


def check_chat_header_menu(page, messages: int) -> None:
    """Verify one menu with Dynamic/Fixed/Off/Inherited; legacy modes remain readable."""
    trigger = page.locator('[data-test-id="chat-header-more-menu-trigger"]')
    trigger.wait_for(state = "visible", timeout = 30_000)
    if messages != 400:
        trigger.click()
        page.keyboard.press("Escape")
        return

    def choose(level: str) -> None:
        # A deterministic fresh keyboard gesture avoids toggling the trigger
        # while Radix is still processing the previous radio selection.
        if trigger.get_attribute("aria-expanded") == "true":
            page.keyboard.press("Escape")
        trigger.focus()
        trigger.press("Enter")
        if trigger.get_attribute("aria-expanded") != "true":
            raise AssertionError(f"Chat header menu failed to open for {level}")
        submenu = page.locator('[data-test-id="agent-dom-level-trigger"]')
        try:
            submenu.wait_for(state = "visible", timeout = 30_000)
        except Exception:
            diagnostic = page.evaluate("""() => ({
              expanded: document.querySelector('[data-test-id="chat-header-more-menu-trigger"]')
                ?.getAttribute('aria-expanded') ?? null,
              menus: [...document.querySelectorAll('[role="menu"]')]
                .map(x => (x.textContent || '').slice(0, 350)),
              rows: document.querySelectorAll('[data-agent-history-row]').length,
              activeDomLevel: document.querySelector('[data-agent-history-scroll-list]')
                ?.getAttribute('data-agent-history-dom-level') ?? null
            })""")
            raise AssertionError(f"DOM mode submenu missing for {level}: {diagnostic}")
        submenu.focus()
        submenu.press("ArrowRight")
        option = page.locator(f'[data-test-id="agent-dom-level-{level}"]')
        option.wait_for(state = "visible", timeout = 30_000)
        option.focus()
        option.press("Enter")

    for level, dom_policy, cap in (
        ("dynamic", "medium", 16),
        ("fixed", "legacy", 32),  # Fixed-low now adapts 5..32 within a DOM budget
    ):
        choose(level)
        page.wait_for_function(
            """([expected, policy, maximum]) => {
              const list = document.querySelector('[data-agent-history-scroll-list="true"]');
              const rows = [...document.querySelectorAll('[data-agent-history-row]')];
              return list?.getAttribute('data-agent-history-dom-level') === policy &&
                rows.length >= 1 && rows.length <= maximum &&
                rows.every(row => row.getAttribute('aria-setsize') === String(expected));
            }""",
            arg = [messages, dom_policy, cap],
            timeout = 60_000,
        )
        census(page, messages, f"dom-budget-{level}")

        # The 15k-descendant native-scrollbar drag is covered by
        # check_isolated_dynamic_thumb_modes in a separate real-browser job.
        # Keep this test focused on all menu modes and transcript behavior.

    choose("off")
    page.wait_for_function(
        """() => document.querySelector('[data-agent-history-scroll-list="true"]') === null""",
        timeout = 30_000,
    )
    choose("inherit")
    page.wait_for_function(
        """expected => {
          const list = document.querySelector('[data-agent-history-scroll-list="true"]');
          return list?.getAttribute('data-agent-history-dom-level') === 'legacy' &&
            !!list.querySelector('[data-agent-history-row][aria-setsize="' + expected + '"]');
        }""",
        arg = messages,
        timeout = 60_000,
    )
    census(page, messages, "dom-budget-inherit-legacy")
    overrides = page.evaluate(
        """() => JSON.parse(localStorage.getItem(
          'unsloth_agent_chat_history_v1') || '{}').state?.domLevelByThreadId || {}"""
    )
    if overrides:
        raise AssertionError("Per-chat DOM mode override remained after inheritance")


def check_runtime_window_size(page, thread_id: str, messages: int) -> None:
    """Verify old 2/5/8/16/32 numeric v1 preferences still hydrate after UI consolidation.

    The new user-facing menu deliberately has no separate legacy row-size
    control. Seed only the disposable synthetic test profile, reload through
    the actual production hydration path, and assert the exact mounted rows.
    """

    def choose(size: int) -> None:
        page.evaluate(
            """([threadId, rows]) => {
              const key = 'unsloth_agent_chat_history_v1';
              const stored = JSON.parse(localStorage.getItem(key) || '{}');
              const state = stored.state || {};
              state.enabledThreads = { ...(state.enabledThreads || {}), [threadId]: true };
              state.domLevelByThreadId = { ...(state.domLevelByThreadId || {}) };
              delete state.domLevelByThreadId[threadId];
              state.windowRowsByThreadId = {
                ...(state.windowRowsByThreadId || {})
              };
              if (rows === (state.globalWindowRows || 5)) {
                delete state.windowRowsByThreadId[threadId];
              } else {
                state.windowRowsByThreadId[threadId] = rows;
              }
              localStorage.setItem(key, JSON.stringify({ ...stored, state }));
            }""",
            [thread_id, size],
        )
        page.reload(wait_until = "domcontentloaded", timeout = 120_000)
        page.wait_for_function(
            """([expected, rows]) => {
              const list = document.querySelector('[data-agent-history-scroll-list="true"]');
              const mounted = [...document.querySelectorAll('[data-agent-history-row]')];
              return list?.getAttribute('data-agent-history-dom-level') === 'legacy' &&
                mounted.length === rows &&
                mounted.every(row => row.getAttribute('aria-setsize') === String(expected));
            }""",
            arg = [messages, size],
            timeout = 60_000,
        )
        state = census(page, messages, f"legacy-window-{size}")
        if state["mounted"] != size:
            raise AssertionError(f"Legacy window {size} not honored: {state}")

    for size in (5, 2, 8, 16, 32):
        choose(size)
    choose(5)
    scroll_to(page, 0, messages, "five-top")
    scroll_to(page, 1, messages, "five-bottom")
    page.reload(wait_until = "domcontentloaded", timeout = 120_000)
    page.wait_for_function(
        """() => document.querySelectorAll('[data-agent-history-row]').length === 5""",
        timeout = 60_000,
    )
    census(page, messages, "window-default-restored")


def run(
    url: str,
    username: str,
    password: str,
    messages: int,
    critical_scrollbar_only: bool = False,
) -> None:
    from playwright.sync_api import sync_playwright

    base = url.rstrip("/")
    auth = authenticate(base, username, password)
    thread_id = None
    try:
        thread_id = create_fixture(base, auth, messages)
        # Read-after-write snapshot before any browser rendering, navigation or
        # search. A mere message count cannot detect silent edits or reordering.
        baseline = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
        baseline_messages = baseline["messages"]
        with sync_playwright() as driver:
            # Playwright normally starts headless Chromium with --hide-scrollbars,
            # making a native-thumb mouse drag impossible regardless of CSS.
            browser = driver.chromium.launch(
                headless = True, ignore_default_args = ["--hide-scrollbars"]
            )
            try:
                context = browser.new_context(viewport = {"width": 1440, "height": 960})
                # The auth bootstrap is rerun on each navigation. Preferences must
                # only be seeded once, or reloading wipes the value under test.
                history_state = {
                    "state": {"enabledThreads": {thread_id: True}},
                    "version": 0,
                }
                context.add_init_script(seed_init_script(auth, []))
                context.add_init_script(
                    "(() => { const key = 'unsloth_agent_chat_history_v1';"
                    " if (localStorage.getItem(key) !== null) return;"
                    " localStorage.setItem(key, "
                    + json.dumps(json.dumps(history_state))
                    + "); })();"
                )
                page = context.new_page()
                page.goto(
                    f"{base}/chat?thread={thread_id}",
                    wait_until = "domcontentloaded",
                    timeout = 120_000,
                )
                page.locator(LIST).wait_for(state = "attached", timeout = 60_000)
                page.wait_for_function(
                    """expected => document.querySelectorAll(
                      '[data-agent-history-row][aria-setsize="' + expected + '"]'
                    ).length > 0""",
                    arg = messages,
                    timeout = 60_000,
                )
                tail = census(page, messages, "initial")
                if tail["mounted"] != 5:
                    raise AssertionError(f"New agent-history default must mount five rows: {tail}")
                if critical_scrollbar_only:
                    check_isolated_dynamic_thumb_modes(page, thread_id, messages)
                    stored = auth_request_json(
                        auth, f"{base}/api/chat/threads/{thread_id}/messages"
                    )
                    if stored.get("messages") != baseline_messages:
                        raise AssertionError(
                            "Isolated scrollbar test changed persisted message content or order"
                        )
                    print(
                        json.dumps(
                            {
                                "gate": "native-scrollbar",
                                "messages": messages,
                                "modes": ["dynamic", "dynamic-high", "dynamic-low"],
                                "transcript_preserved": True,
                                "result": "passed",
                            },
                            sort_keys = True,
                        )
                    )
                    return
                check_chat_header_menu(page, messages)
                if messages == 400:
                    check_runtime_window_size(page, thread_id, messages)
                if tail["ordinal"][-1] != messages:
                    raise AssertionError(f"initial window not at transcript tail: {tail}")
                top = scroll_to(page, 0, messages, "top")
                middle = scroll_to(page, 0.5, messages, "middle")
                bottom = scroll_to(page, 1, messages, "bottom")
                # Reproduction: a query for message 6 must reach beyond the
                # 32 rows mounted at the transcript tail, without disabling
                # bounded virtualization or changing the persisted history.
                page.keyboard.press("Control+f")
                find_input = page.locator("[data-find-bar-layer] input")
                find_input.wait_for(state = "visible", timeout = 30_000)
                find_input.fill("row-00005")
                # Distinguish indexing / count from seek / mount / scroll.
                require_search_state(
                    page,
                    """() => document.querySelector(
                      '[data-find-bar-layer] [aria-live="polite"]'
                    )?.textContent?.includes('1/1') ?? false""",
                    "search-index",
                )
                require_search_state(
                    page,
                    """() => !!document.querySelector(
                      '[data-agent-history-row="5"][data-agent-history-find-active="true"]'
                    )""",
                    "search-reveal",
                )
                require_visible_search_row(page, 5, "search-early-visible")
                census(page, messages, "search-early-message")
                # Prove actual browser copy of a selected mounted message works
                # under the bounded renderer. This does NOT claim that Ctrl+A
                # across the *entire* transcript is yet equivalent to default.
                context.grant_permissions(["clipboard-read", "clipboard-write"])
                page.locator('[data-agent-history-row="5"]').click()
                selected = page.evaluate(
                    """() => {
                      const row = document.querySelector('[data-agent-history-row="5"]');
                      if (!row) return false;
                      const selection = window.getSelection();
                      const range = document.createRange();
                      range.selectNodeContents(row);
                      selection.removeAllRanges();
                      selection.addRange(range);
                      return !selection.isCollapsed && selection.toString().includes('row-00005');
                    }"""
                )
                if not selected:
                    raise AssertionError("Visible virtualized row could not be selected")
                page.keyboard.press("Control+c")
                copied = page.evaluate("async () => navigator.clipboard.readText()")
                if "agent-history-e2e-row-00005" not in copied:
                    raise AssertionError(
                        f"Copy of visible virtualized message differs: {copied[:160]!r}"
                    )
                page.evaluate("() => window.getSelection()?.removeAllRanges()")
                # Scrolling away from an active one-result query then pressing
                # Enter must re-seek even if React's selected hit is unchanged.
                scroll_to(page, 1, messages, "search-reader-scroll-away")
                find_input.focus()
                find_input.press("Enter")
                require_visible_search_row(page, 5, "search-repeat-visible")
                census(page, messages, "search-repeat-single-result")
                find_input.fill("unique-fixture")
                require_search_state(
                    page,
                    """expected => {
                      const counter = document.querySelector(
                        '[data-find-bar-layer] [aria-live="polite"]'
                      )?.textContent?.trim() ?? '';
                      const fields = counter.split('/');
                      const current = Number(fields[0]);
                      return fields.length === 2 && Number(fields[1]) === expected &&
                        Number.isInteger(current) && current >= 1 && current <= expected;
                    }""",
                    "search-all-rows",
                    arg = messages,
                )
                census(page, messages, "search-full-thread")
                find_input.fill(f"row-{messages - 1:05d}")
                require_search_state(
                    page,
                    """expected => !!document.querySelector(
                      '[data-agent-history-row="' + String(expected - 1) + '"][data-agent-history-find-active="true"]'
                    )""",
                    "search-last-row",
                    arg = messages,
                )
                require_visible_search_row(page, messages - 1, "search-last-visible")
                census(page, messages, "search-latest-message")
                page.keyboard.press("Escape")
                page.reload(wait_until = "domcontentloaded", timeout = 120_000)
                page.locator(LIST).wait_for(state = "attached", timeout = 60_000)
                page.wait_for_function(
                    """expected => Array.from(document.querySelectorAll(
                      '[data-agent-history-row]')).some(
                        row => row.getAttribute('aria-setsize') === String(expected)
                      )""",
                    arg = messages,
                    timeout = 60_000,
                )
                after_reload = census(page, messages, "reload")
                if after_reload["ordinal"][-1] != messages:
                    raise AssertionError("Reload lost the transcript tail")
                stored = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
                if len(stored.get("messages", [])) != messages:
                    raise AssertionError("Virtual scrolling changed the persisted transcript count")
                if stored["messages"] != baseline_messages:
                    raise AssertionError(
                        "Virtual scrolling or search altered persisted message content, "
                        "metadata, identity or order"
                    )
                print(
                    json.dumps(
                        {
                            "status": "PASS",
                            "messages": messages,
                            "mounted_initial": tail["mounted"],
                            "mounted_top": top["mounted"],
                            "mounted_middle": middle["mounted"],
                            "mounted_bottom": bottom["mounted"],
                            "mounted_reload": after_reload["mounted"],
                            "transcript_preserved": True,
                            "visible_message_clipboard_copy": True,
                        },
                        sort_keys = True,
                    )
                )
            finally:
                browser.close()
    finally:
        if thread_id is not None:
            auth_request_json(
                auth,
                f"{base}/api/chat/threads",
                method = "DELETE",
                body = {"ids": [thread_id]},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--url", required = True, help = "Disposable Studio instance only")
    parser.add_argument("--username", default = "unsloth")
    parser.add_argument("--password", required = True)
    parser.add_argument("--messages", type = int, default = 400)
    parser.add_argument(
        "--critical-scrollbar-only",
        action = "store_true",
        help = "Run the real native-thumb drag for both dynamic buffer modes only",
    )
    args = parser.parse_args()
    if args.messages < 128 or args.messages % 2:
        parser.error("--messages must be even and at least 128")
    if args.critical_scrollbar_only and args.messages != 400:
        parser.error("--critical-scrollbar-only requires --messages 400")
    run(
        args.url,
        args.username,
        args.password,
        args.messages,
        critical_scrollbar_only = args.critical_scrollbar_only,
    )


if __name__ == "__main__":
    main()
