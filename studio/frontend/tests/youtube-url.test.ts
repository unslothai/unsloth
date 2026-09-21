// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  extractYoutubeVideoId,
  extractYoutubeVideoUrlFromClipboard,
} from "../src/features/chat/utils/youtube-url.ts";

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

test("finds YouTube links in clipboard text and URI payloads", () => {
  const shortUrl = `https://youtu.be/${ID}`;
  const cases: Array<[Record<string, string>, string | null]> = [
    [{ "text/plain": `Rick Astley\n${shortUrl}` }, shortUrl],
    [{ "text/uri-list": `# copied link\r\n${shortUrl}\r\n` }, shortUrl],
    [{ "text/plain": "Rick Astley", "text/uri-list": shortUrl }, shortUrl],
    [
      { "text/uri-list": `# source ${shortUrl}\r\nhttps://example.com/video` },
      null,
    ],
  ];
  for (const [data, expected] of cases) {
    assert.equal(
      extractYoutubeVideoUrlFromClipboard({
        getData: (type: string) => data[type] ?? "",
      }),
      expected,
    );
  }
});

test("rejects clipboard payloads without a YouTube video link", () => {
  assert.equal(extractYoutubeVideoUrlFromClipboard(null), null);
  assert.equal(
    extractYoutubeVideoUrlFromClipboard({
      getData: () => "Read https://example.com/watch?v=dQw4w9WgXcQ",
    }),
    null,
  );
});
