// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images and Video pages open the Library's viewer: prompt-derived names, the clip's playback
// handed each way, and a chat hand-off that survives a dead signed link.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { mediaFileName, shortPrompt } from "../src/lib/prompt-text.ts";
import {
  fetchWithFreshLink,
  playWithMutedFallback,
  readPlayback,
} from "../src/features/video/viewer.ts";

const a59 = "a".repeat(59);

test("a prompt becomes a safe file name, cut by code point", () => {
  for (const [prompt, extension, expected] of [
    ['a cat: "sitting" on a/b\\c?', "png", "a cat sitting on a b c.png"],
    ["tabs\tand\nnewlines\r\n  here", "mp4", "tabs and newlines here.mp4"],
    ["bell\u0007 and del\u007f and c1\u0085", "png", "bell and del and c1.png"],
    ["", "png", "Untitled.png"],
    ["  ***  ???  ", "png", "Untitled.png"],
    ["...", "mp4", "Untitled.mp4"],
    ["a sunset...", "png", "a sunset.png"],
    [".hidden prompt", "png", "hidden prompt.png"],
    // The cut can land just after a space or a dot.
    [`${a59} tail`, "png", `${a59}.png`],
    [`${a59}.tail`, "png", `${a59}.png`],
    // By UTF-16 unit, 60 would split a cat.
    [`a${"🐱".repeat(80)}`, "png", `a${"🐱".repeat(59)}.png`],
  ]) {
    assert.equal(mediaFileName(prompt, extension), expected, prompt);
  }
  assert.equal(mediaFileName("abcdef", "png", 3), "abc.png");
});

test("the preview's name keeps a short prompt, cut at a word", () => {
  const long = "a very detailed painting of a lighthouse on a cliff at dusk, waves crashing below, gulls";
  for (const [prompt, max, expected] of [
    ["a red fox in snow", 80, "a red fox in snow"],
    ["  a red\n fox  ", 80, "a red fox"],
    ["", 80, ""],
    [" \n ", 80, ""],
    [long, 40, "a very detailed painting of a lighthouse…"],
    [long, 38, "a very detailed painting of a…"],
    // No space to cut at: cut where the limit falls.
    ["x".repeat(50), 10, `${"x".repeat(10)}…`],
    [`a${"🐱".repeat(20)}`, 10, `a${"🐱".repeat(9)}…`],
  ] as const) {
    assert.equal(shortPrompt(prompt, max), expected, prompt);
  }
});

test("a signed link refused with 401 or 403 is minted afresh once; other statuses are returned", async () => {
  const realFetch = globalThis.fetch;
  try {
    for (const [statuses, asked, final] of [
      [{ "/old": 200 }, ["/old"], 200],
      [{ "/old": 401, "/new": 200 }, ["/old", "/new"], 200],
      [{ "/old": 403, "/new": 200 }, ["/old", "/new"], 200],
      [{ "/old": 404 }, ["/old"], 404],
      [{ "/old": 401, "/new": 401 }, ["/old", "/new"], 401],
    ] as const) {
      const seen: string[] = [];
      globalThis.fetch = (async (url: string) => {
        seen.push(url);
        const status = (statuses as Record<string, number>)[url]!;
        return new Response(status === 200 ? "clip" : null, { status });
      }) as typeof fetch;
      const response = await fetchWithFreshLink("/old", async () => "/new");
      assert.equal(response.status, final);
      assert.deepEqual(seen, asked);
    }
  } finally {
    globalThis.fetch = realFetch;
  }
});

const START = { time: 12.5, playing: true, muted: false, volume: 0.4 };
const player = (overrides: object = {}) =>
  ({ currentTime: 3, paused: false, ended: false, muted: true, volume: 0.8, readyState: 4, ...overrides }) as unknown as HTMLVideoElement;

test("playback is the player's own once positioned, else the start with the player's mute", () => {
  const loaded = { time: 3, playing: true, muted: true, volume: 0.8 };
  const notSeeked = player({ currentTime: 0, paused: true, readyState: 1 });
  for (const [video, positioned, expected] of [
    [player(), undefined, loaded],
    [player({ paused: true }), undefined, { ...loaded, playing: false }],
    // Ended is not paused, but there is nothing left to resume.
    [player({ ended: true }), undefined, { ...loaded, playing: false }],
    // Before metadata its volume is unset; muted is set as it mounts.
    [player({ currentTime: 0, paused: true, readyState: 0, volume: 1 }), undefined, { ...START, muted: true }],
    [notSeeked, false, { ...START, muted: true }],
    [notSeeked, true, { time: 0, playing: false, muted: true, volume: 0.8 }],
    [null, undefined, START],
  ] as const) {
    assert.deepEqual(readPlayback(video, START, positioned), expected);
  }
});

test("a refusal of sound retries muted; an abort or a muted refusal is not retried, and nothing rejects", async () => {
  for (const [refusals, muted, plays] of [
    [[], false, [false]],
    [["NotAllowedError"], false, [false, true]],
    [["NotAllowedError", "NotAllowedError"], false, [false, true]],
    [["AbortError"], false, [false]],
    [["NotAllowedError"], true, [true]],
  ] as const) {
    const queue: string[] = [...refusals];
    const seen: boolean[] = [];
    const video = {
      muted,
      play() {
        seen.push(video.muted);
        const name = queue.shift();
        return name ? Promise.reject(Object.assign(new Error(name), { name })) : Promise.resolve();
      },
    };
    await playWithMutedFallback(video);
    assert.deepEqual(seen, plays, refusals.join());
    assert.equal(video.muted, plays.at(-1));
  }
});

// Between two markers of a page's source.
function slice(page: string, from: string, to: string): string {
  const start = page.indexOf(from);
  assert.ok(start >= 0, from);
  return page.slice(start, page.indexOf(to, start));
}

test("the Images preview is a button named for its prompt", () => {
  const preview = slice(readSrc("features/images/images-page.tsx"), "alt={selected.prompt}", "/>");
  for (const attribute of ['role="button"', "tabIndex={0}", "aria-label={openImageLabel(t, selected.prompt)}"]) {
    assert.ok(preview.includes(attribute), attribute);
  }
});

test("the Video viewer plays only if the inline clip was, and hands its place back", () => {
  const page = readSrc("features/video/video-page.tsx");
  const viewer = slice(page, "<video\n                ref={viewerVideoRef}", "</MediaViewer>");
  assert.ok(!/^\s*autoPlay(=|\s*$)/m.test(viewer), "no autoPlay");
  for (const line of [
    "if (viewer.from.playing) void playWithMutedFallback(video);",
    // Its time is only its own once it has seeked there.
    "viewerPositioned.current = true;",
    "readPlayback(event.currentTarget, viewer.from, viewerPositioned.current)",
  ]) {
    assert.ok(viewer.includes(line), line);
  }
  // Opened from its Open button: the native controls also take clicks on the frame.
  const inline = slice(page, "ref={previewRef}", "/>");
  assert.ok(!inline.includes("onClick") && !inline.includes("openViewer"));
  // The handback is dropped once another clip is shown, and waits for the shown clip's link.
  const effect = slice(page, "const last = handback.current;", "}, [viewer, shownId, selectedSrc]);");
  for (const line of [
    "if (last.id !== shownId) {\n      handback.current = null;",
    "if (!selectedSrc || !inline) return;",
    "playback.playing && activeRef.current",
  ]) {
    assert.ok(effect.includes(line), line);
  }
});
