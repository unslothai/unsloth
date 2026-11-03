// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { AudioPlaybackLinks } from "../src/features/audio/audio-playback-links.ts";

test("an expired selection can be refreshed without discarding a playing source", () => {
  const links = new AudioPlaybackLinks();
  links.set("song", "/song?token=first", 1000);
  assert.equal(links.isFresh("song", 1000 + 49 * 60 * 1000), true);
  assert.equal(links.isFresh("song", 1000 + 50 * 60 * 1000), false);
  assert.equal(links.get("song"), "/song?token=first");
  assert.deepEqual(links.toRecord(), { song: "/song?token=first" });
});

test("a media failure retries the current source once, including after the link is replaced", () => {
  const links = new AudioPlaybackLinks();
  links.set("song", "/song?token=first");
  assert.equal(links.retry("song", "/song?token=older"), false);
  assert.equal(links.retry("song", "/song?token=first"), true);
  assert.equal(links.get("song"), undefined);
  links.set("song", "/song?token=second");
  assert.equal(links.retry("song", "/song?token=first"), false);
  assert.equal(links.retry("song", "/song?token=second"), false);
  assert.equal(links.get("song"), "/song?token=second");
});

test("dropping a clip or clearing the gallery drops its retry history too", () => {
  const links = new AudioPlaybackLinks();
  links.set("song", "/song?token=first");
  links.retry("song", "/song?token=first");
  links.delete("song");
  links.set("song", "/song?token=second");
  assert.equal(links.retry("song", "/song?token=second"), true);
  links.set("other", "/other?token=first");
  links.clear();
  assert.deepEqual(links.toRecord(), {});
  assert.equal(links.isFresh("other"), false);
  links.set("song", "/song?token=third");
  assert.equal(links.retry("song", "/song?token=third"), true);
});

test("a recovered link can be renewed again when it expires later", () => {
  const links = new AudioPlaybackLinks();
  links.set("song", "/song?token=first", 0);
  assert.equal(links.retry("song", "/song?token=first", 1000), true);
  links.set("song", "/song?token=second", 1000);
  assert.equal(links.retry("song", "/song?token=second", 2000), false);
  assert.equal(
    links.retry("song", "/song?token=second", 1000 + 50 * 60 * 1000),
    true,
  );
  links.set("song", "/song?token=third", 1000 + 50 * 60 * 1000);
  assert.equal(links.retry("song", "/song?token=second"), false);
  assert.equal(
    links.retry("song", "/song?token=third", 2000 + 50 * 60 * 1000),
    false,
  );
});
