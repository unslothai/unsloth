// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Chat about this" on the Images and Video viewers attaches the file under a name taken from its
// prompt, and reads a clip through a signed link that a server restart can invalidate.

import assert from "node:assert/strict";
import test from "node:test";

import { mediaFileName } from "../src/features/library/media-file-name.ts";
import { fetchWithFreshLink } from "../src/features/video/signed-link-fetch.ts";

test("a prompt becomes a file name without the characters a file system refuses", () => {
  assert.equal(mediaFileName('a cat: "sitting" on a/b\\c?', "png"), "a cat sitting on a b c.png");
  assert.equal(mediaFileName("tabs\tand\nnewlines\r\n  here", "mp4"), "tabs and newlines here.mp4");
  assert.equal(mediaFileName("bell\u0007 and del\u007f and c1\u0085", "png"), "bell and del and c1.png");
});

test("an empty or unusable prompt falls back to Untitled", () => {
  assert.equal(mediaFileName("", "png"), "Untitled.png");
  assert.equal(mediaFileName('  ***  ???  ', "png"), "Untitled.png");
  assert.equal(mediaFileName("...", "mp4"), "Untitled.mp4");
});

test("the name never ends in a dot or a space, nor starts with a dot", () => {
  assert.equal(mediaFileName("a sunset...", "png"), "a sunset.png");
  assert.equal(mediaFileName(".hidden prompt", "png"), "hidden prompt.png");
  // The cut can land just after a space or a dot.
  assert.equal(mediaFileName(`${"a".repeat(59)} tail`, "png"), `${"a".repeat(59)}.png`);
  assert.equal(mediaFileName(`${"a".repeat(59)}.tail`, "png"), `${"a".repeat(59)}.png`);
});

test("the cut counts code points, so an emoji is never split", () => {
  // Cut by UTF-16 unit, 60 would end half way through the 30th cat.
  const name = mediaFileName(`a${"🐱".repeat(80)}`, "png");
  assert.equal(name, `a${"🐱".repeat(59)}.png`);
  assert.ok(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/.test(name), "no lone high surrogate");
  assert.equal(mediaFileName("abcdef", "png", 3), "abc.png");
});

function response(status: number): Response {
  return new Response(status === 200 ? "clip" : null, { status });
}

test("a working signed link is used as is", async () => {
  const asked: string[] = [];
  let minted = 0;
  const res = await fetchWithFreshLink(
    "/old",
    async () => {
      minted += 1;
      return "/new";
    },
    async (url) => {
      asked.push(url);
      return response(200);
    },
  );
  assert.equal(res.status, 200);
  assert.deepEqual(asked, ["/old"]);
  assert.equal(minted, 0);
});

for (const status of [401, 403]) {
  test(`a signed link refused with ${status} is minted afresh once`, async () => {
    const asked: string[] = [];
    const res = await fetchWithFreshLink(
      "/old",
      async () => "/new",
      async (url) => {
        asked.push(url);
        return response(url === "/old" ? status : 200);
      },
    );
    assert.equal(res.status, 200);
    assert.deepEqual(asked, ["/old", "/new"]);
  });
}

test("other failures are not retried, and a fresh link that also fails is returned", async () => {
  const asked: string[] = [];
  const missing = await fetchWithFreshLink("/old", async () => "/new", async (url) => {
    asked.push(url);
    return response(404);
  });
  assert.equal(missing.status, 404);
  assert.deepEqual(asked, ["/old"]);

  const stillRefused = await fetchWithFreshLink("/old", async () => "/new", async () => response(401));
  assert.equal(stillRefused.status, 401);
});
