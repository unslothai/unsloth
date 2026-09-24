// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Video page hands its clip to the full-window viewer and back: the time, whether it was
// playing, and its sound. A player that has not loaded reads 0 whatever it was asked to start at,
// so a quick close or a failed first load must not rewind the clip.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { playWithMutedFallback, readPlayback } = await import(
  "../src/features/video/viewer-playback.ts"
);

const START = { time: 12.5, playing: true, muted: false, volume: 0.4 };

function player(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    currentTime: 3,
    paused: false,
    ended: false,
    muted: true,
    volume: 0.8,
    readyState: 4,
    ...overrides,
  } as unknown as HTMLVideoElement;
}

test("a loaded player reports its own time, play state and sound", () => {
  assert.deepEqual(readPlayback(player(), START), {
    time: 3,
    playing: true,
    muted: true,
    volume: 0.8,
  });
  assert.equal(readPlayback(player({ paused: true }), START).playing, false);
  // An ended clip is not paused, but there is nothing left to resume.
  assert.equal(readPlayback(player({ ended: true }), START).playing, false);
});

test("before metadata the time, play state and volume stay where it was asked to start", () => {
  // Its volume is only set once it has loaded; muted is set as it mounts.
  const unloaded = player({ currentTime: 0, paused: true, readyState: 0, volume: 1 });
  assert.deepEqual(readPlayback(unloaded, START), {
    time: 12.5,
    playing: true,
    muted: true,
    volume: 0.4,
  });
});

test("a player with metadata that has not seeked yet is not positioned", () => {
  const loadedNotSeeked = player({ currentTime: 0, paused: true, readyState: 1 });
  assert.equal(readPlayback(loadedNotSeeked, START, false).time, 12.5);
  assert.equal(readPlayback(loadedNotSeeked, START, false).playing, true);
  assert.equal(readPlayback(loadedNotSeeked, START, true).time, 0);
});

test("no player at all hands back the start", () => {
  assert.deepEqual(readPlayback(null, START), START);
});

function fakePlayer(refusals: string[], muted = false) {
  const plays: boolean[] = [];
  const video = {
    muted,
    play() {
      plays.push(video.muted);
      const refusal = refusals.shift();
      if (!refusal) return Promise.resolve();
      return Promise.reject(Object.assign(new Error(refusal), { name: refusal }));
    },
  };
  return { video, plays };
}

test("a clip that may play with sound plays once, unmuted", async () => {
  const { video, plays } = fakePlayer([]);
  await playWithMutedFallback(video);
  assert.deepEqual(plays, [false]);
  assert.equal(video.muted, false);
});

test("a refusal to play with sound falls back to playing muted", async () => {
  const { video, plays } = fakePlayer(["NotAllowedError"]);
  await playWithMutedFallback(video);
  assert.deepEqual(plays, [false, true]);
  assert.equal(video.muted, true);
});

test("an aborted play, or a muted one refused, is not retried and never rejects", async () => {
  const aborted = fakePlayer(["AbortError"]);
  await playWithMutedFallback(aborted.video);
  assert.deepEqual(aborted.plays, [false]);
  assert.equal(aborted.video.muted, false);

  const mutedRefused = fakePlayer(["NotAllowedError"], true);
  await playWithMutedFallback(mutedRefused.video);
  assert.deepEqual(mutedRefused.plays, [true]);

  const refusedTwice = fakePlayer(["NotAllowedError", "NotAllowedError"]);
  await playWithMutedFallback(refusedTwice.video);
  assert.deepEqual(refusedTwice.plays, [false, true]);
});

const page = readSrc("features/video/video-page.tsx");
const viewer = page.slice(
  page.indexOf("<video\n                ref={viewerVideoRef}"),
  page.indexOf("</MediaViewer>"),
);

test("the viewer's player only plays when the inline one was playing", () => {
  assert.ok(viewer.length > 0, "the viewer's player must exist");
  assert.ok(!/^\s*autoPlay(=|\s*$)/m.test(viewer), "no autoPlay attribute");
  assert.ok(viewer.includes("if (viewer.from.playing) void playWithMutedFallback(video);"));
  // Its time is only its own once it has seeked there.
  assert.ok(viewer.includes("viewerPositioned.current = true;"));
  assert.ok(viewer.includes("readPlayback(event.currentTarget, viewer.from, viewerPositioned.current)"));
});

test("the inline clip opens the viewer from its Open button, not a click on the frame", () => {
  const inline = page.slice(
    page.indexOf("                  ref={previewRef}"),
    page.indexOf("/>", page.indexOf("                  ref={previewRef}")),
  );
  assert.ok(inline.length > 0);
  assert.ok(!inline.includes("onClick"));
  assert.ok(!inline.includes("openViewer"));
});

test("the handback is dropped once another clip is shown, and waits for the shown clip's link", () => {
  const effect = page.slice(
    page.indexOf("const last = handback.current;"),
    page.indexOf("}, [viewer, shownId, selectedSrc]);"),
  );
  assert.ok(effect.length > 0);
  assert.ok(effect.includes("if (last.id !== shownId) {\n      handback.current = null;"));
  assert.ok(effect.includes("if (!selectedSrc || !inline) return;"));
  assert.ok(effect.includes("playback.playing && activeRef.current"));
});
