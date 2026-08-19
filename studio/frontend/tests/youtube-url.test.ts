// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { extractYoutubeVideoId } from "../src/features/chat/utils/youtube-url.ts";

const ID = "dQw4w9WgXcQ";

test("recognises the YouTube link shapes the composer offer covers", () => {
  for (const url of [
    `https://www.youtube.com/watch?v=${ID}`,
    `https://youtube.com/watch?v=${ID}&t=42s`,
    `https://m.youtube.com/watch?v=${ID}`,
    `https://music.youtube.com/watch?v=${ID}`,
    `https://youtu.be/${ID}`,
    `https://youtu.be/${ID}?t=42`,
    `https://www.youtube.com/shorts/${ID}`,
    `https://www.youtube.com/embed/${ID}`,
    `https://www.youtube.com/live/${ID}`,
    `https://www.youtube-nocookie.com/embed/${ID}`,
    `  https://www.youtube.com/watch?v=${ID}  `,
  ]) {
    assert.equal(extractYoutubeVideoId(url), ID, url);
  }
});

test("rejects look-alike hosts and non-video URLs", () => {
  for (const url of [
    // A path segment must not stand in for the host, or any site could trigger the offer.
    `https://evil.com/youtube.com/watch?v=${ID}`,
    `https://youtube.com.evil.com/watch?v=${ID}`,
    `https://notyoutube.com/watch?v=${ID}`,
    `javascript:alert(1)//youtube.com/watch?v=${ID}`,
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/@somechannel",
    "https://www.youtube.com/results?search_query=cats",
    "look at this video",
    "",
  ]) {
    assert.equal(extractYoutubeVideoId(url), null, url);
  }
});
