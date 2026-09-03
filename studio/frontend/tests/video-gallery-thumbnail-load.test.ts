// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ThumbnailRequestQueue,
  withThumbnailRetries,
} from "../src/features/video/thumbnail-request-queue.ts";

const BROKEN_POSTER = /broken poster/;

function source(path: string): string {
  return readFileSync(
    fileURLToPath(new URL(path, import.meta.url)),
    "utf8",
  ).replace(/\r\n/g, "\n");
}

function between(text: string, start: string, end: string): string {
  const from = text.indexOf(start);
  const to = text.indexOf(end, from + start.length);
  if (from === -1 || to <= from) {
    throw new Error(`markers not found: ${start} / ${end}`);
  }
  return text.slice(from, to);
}

test("video gallery tiles use still posters instead of media pipelines", () => {
  const page = source("../src/features/video/video-page.tsx");
  const strip = between(
    page,
    "{/* In-progress generation: a placeholder tile",
    "{/* Tail spinner while older pages stream in on scroll.",
  );

  assert.ok(strip.includes("thumbnailById[video.id]"));
  assert.ok(strip.includes("<img"));
  assert.ok(strip.includes("src={thumbnailById[video.id]}"));
  assert.ok(!strip.includes("<video"));
  assert.ok(!strip.includes("srcById[video.id]"));
});

test("selected playback does not depend on poster decoding", () => {
  const page = source("../src/features/video/video-page.tsx");
  const selected = between(
    page,
    "// The preview player is what the user watches",
    "// Bumped by every LOCAL change",
  );

  assert.ok(selected.includes("void ensureThumbnail(selected);"));
  assert.ok(selected.includes("void ensureSrc(selected);"));
  assert.ok(page.includes("thumbnailById: new BlobUrlCache(32 * 1024 * 1024)"));
  assert.ok(page.includes("galleryCache.thumbnailById.clear();"));
});

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("video poster requests stay within their decoder concurrency cap", async () => {
  const queue = new ThumbnailRequestQueue(3);
  const gates = Array.from({ length: 8 }, deferred);
  const started: number[] = [];
  let active = 0;
  let peak = 0;

  const all = gates.map((gate, index) =>
    queue.run(async () => {
      started.push(index);
      active += 1;
      peak = Math.max(peak, active);
      await gate.promise;
      active -= 1;
    }),
  );

  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(started, [0, 1, 2]);
  assert.equal(queue.active, 3);
  assert.equal(queue.pending, 5);
  for (const gate of gates) {
    gate.resolve();
  }
  await Promise.all(all);
  assert.equal(peak, 3);
  assert.equal(queue.active, 0);
  assert.equal(queue.pending, 0);
});

test("a failed poster request releases its queue slot", async () => {
  const queue = new ThumbnailRequestQueue(1);
  const seen: string[] = [];
  const failed = queue.run(() => {
    seen.push("failed");
    return Promise.reject(new Error("broken poster"));
  });
  const recovered = queue.run(() => {
    seen.push("recovered");
    return Promise.resolve(7);
  });

  await assert.rejects(failed, BROKEN_POSTER);
  assert.equal(await recovered, 7);
  assert.deepEqual(seen, ["failed", "recovered"]);
});

test("a poster request that blinked is retried before the clip is called undecodable", async () => {
  let calls = 0;
  const poster = await withThumbnailRetries(
    () =>
      calls++ < 2
        ? Promise.reject(new Error("broken poster"))
        : Promise.resolve("poster"),
    2,
    0,
  );

  assert.equal(poster, "poster");
  assert.equal(calls, 3);
});

test("a poster request that keeps failing gives up so the tile stops asking", async () => {
  let calls = 0;
  const attempt = withThumbnailRetries(
    () => {
      calls += 1;
      return Promise.reject(new Error("broken poster"));
    },
    2,
    0,
  );

  await assert.rejects(attempt, BROKEN_POSTER);
  assert.equal(calls, 3);
});

test("only an exhausted poster request marks a clip undecodable", () => {
  const page = source("../src/features/video/video-page.tsx");
  const ensure = between(
    page,
    "const ensureThumbnail = useCallback(",
    "// A media error on a playing clip",
  );

  assert.ok(ensure.includes("withThumbnailRetries(() =>"));
  assert.match(
    ensure,
    /withThumbnailRetries\(\(\) =>[\s\S]*?\} catch \{\s*\n\s*galleryCache\.thumbnailFailed\.add\(video\.id\);/,
  );
});

test("video posters are fetched through the authenticated thumbnail route", () => {
  const api = source("../src/features/video/api.ts");
  const page = source("../src/features/video/video-page.tsx");
  const helper = between(
    api,
    "export async function fetchGalleryVideoThumbnail(",
    "/** Server-side transcode",
  );

  assert.ok(helper.includes("authFetch("));
  assert.ok(helper.includes("/content?variant=thumbnail`"));
  assert.ok(helper.includes("URL.createObjectURL(blob)"));
  assert.ok(page.includes("videoThumbnailQueue.run(() =>"));
  assert.ok(page.includes("galleryCache.epoch !== epochAtStart"));
});

test("archived video rows reuse still posters", () => {
  const archived = source(
    "../src/features/settings/components/archived-media-dialog.tsx",
  );
  assert.ok(archived.includes("fetchGalleryVideoThumbnail(row.id)"));
  assert.ok(archived.includes("videoThumbnailQueue.run(() =>"));
  assert.ok(!archived.includes("fetchGalleryVideoSignedUrl"));
  const row = between(archived, "{rows.map((row) => (", "{hasMore ? (");
  assert.ok(row.includes("<img"));
  assert.ok(row.includes("src={thumbs[row.id]}"));
  assert.ok(!row.includes("<video"));
});
