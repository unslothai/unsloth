// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DECODE_FAILURE_ALLOWANCE,
  MAX_MODEL_IMAGES,
  MAX_TOTAL_MCP_IMAGES,
  MCP_IMAGES_MARKER,
  boundMcpImageEnvelopes,
  mcpImagesEnvelope,
  splitMcpImages,
} from "../src/features/chat/api/mcp-images.ts";
import { isMcpToolName } from "../src/features/chat/utils/mcp-tool-name.ts";

const IMAGES = [{ data: "QUJD", mimeType: "image/png" }];
const RESULT = `[1 image returned]${mcpImagesEnvelope(IMAGES)}`;

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("a valid envelope splits into the text and its images", () => {
  assert.deepEqual(splitMcpImages(RESULT), {
    text: "[1 image returned]",
    images: IMAGES,
  });
});

test("text that only mentions the marker is left whole", () => {
  const result = `the marker is${MCP_IMAGES_MARKER} and nothing follows`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("an envelope that is not an image array is left whole", () => {
  const result = `log${MCP_IMAGES_MARKER}["not", "image", "dicts"]`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("an unparseable envelope is left whole", () => {
  const result = `log${MCP_IMAGES_MARKER}{oops`;
  assert.deepEqual(splitMcpImages(result), { text: result, images: [] });
});

test("replaying a tool result re-attaches its images for the backend", () => {
  assert.match(adapter, /content \+= mcpImagesEnvelope\(result\.images\);/);
});

test("only an MCP tool call carries the privileged envelope", () => {
  // A client tool is free to answer {text, images:[{data, mimeType}]}, which is
  // the shape isMcpImageToolResult accepts. Without the provenance gate its bytes
  // would be promoted into model image input on the next request.
  assert.equal(isMcpToolName("mcp__fs__read_media_file"), true);
  assert.equal(isMcpToolName("render_chart"), false);
  assert.equal(isMcpToolName(undefined), false);
  // The envelope is gated...
  assert.match(
    adapter,
    /if \(isMcpImageToolResult\(result\) && isMcpToolName\(tc\.toolName\)\) \{\n\s*content \+= mcpImagesEnvelope\(result\.images\);/,
  );
  // ...but the wrapper branch is NOT: excluding a non-MCP wrapper there dropped it
  // into JSON.stringify, which replays the whole base64 array as prompt text.
  assert.match(adapter, /^\s*isMcpImageToolResult\(result\) \|\|$/m);
});

const shot = (n: number, name = "mcp__fs__screenshot") => ({
  role: "tool",
  name,
  content:
    `[3 images returned]` +
    mcpImagesEnvelope([
      { data: `A${n}`, mimeType: "image/png" },
      { data: `B${n}`, mimeType: "image/png" },
      { data: `C${n}`, mimeType: "image/png" },
    ]),
});

const countImages = (messages: { content?: unknown }[]) =>
  messages.reduce(
    (n, m) =>
      n +
      (typeof m.content === "string"
        ? splitMcpImages(m.content).images.length
        : 0),
    0,
  );

test("history is bounded before it is uploaded, not after", () => {
  // The backend's cap runs after the body is parsed, so it cannot bound transport.
  // The ceiling counts CANDIDATES, which sit above the promotable budget so the
  // backend can still scan past entries it cannot decode.
  const messages = [0, 1, 2, 3, 4].map((n) => shot(n));

  const bounded = boundMcpImageEnvelopes(messages);

  assert.equal(countImages(messages), 15);
  assert.ok(countImages(bounded) < countImages(messages), "nothing was bounded");
  assert.ok(
    countImages(bounded) <= MAX_TOTAL_MCP_IMAGES + DECODE_FAILURE_ALLOWANCE,
    `uploaded ${countImages(bounded)} candidates`,
  );
  // The oldest result is still dropped outright.
  assert.equal(splitMcpImages(bounded[0].content).images.length, 0);
});

test("the newest pictures are the ones kept", () => {
  const bounded = boundMcpImageEnvelopes([0, 1, 2, 3, 4].map((n) => shot(n)));

  assert.equal(splitMcpImages(bounded[4].content).images.length, 3);
  assert.equal(splitMcpImages(bounded[0].content).images.length, 0);
  // The note survives even when its payload does not.
  assert.match(bounded[0].content, /^\[3 images returned\]$/);
});

test("a conversation inside the budget is left byte-identical", () => {
  const messages = [0, 1].map((n) => shot(n));

  assert.deepEqual(boundMcpImageEnvelopes(messages), messages);
});

test("a non-MCP tool result is never rewritten", () => {
  const messages = [shot(0, "bash")];

  assert.deepEqual(boundMcpImageEnvelopes(messages), messages);
});

const shotOf = (n: string, count: number) => ({
  role: "tool",
  name: "mcp__fs__screenshot",
  content:
    `[${count} images returned]` +
    mcpImagesEnvelope(
      Array.from({ length: count }, (_, i) => ({
        data: `${n}${i}`,
        mimeType: "image/png",
      })),
    ),
});

test("a fat newest result does not evict older usable images", () => {
  // Budget is charged at the promotable rate, so an oversized newest envelope
  // cannot spend the whole history allowance on images the backend would drop.
  const bounded = boundMcpImageEnvelopes([shotOf("old", 4), shotOf("new", 8)]);

  assert.equal(splitMcpImages(bounded[0].content).images.length, 4);
});

test("no single result uploads more than the backend could ever decode", () => {
  // Bounded, but above the quota: the backend counts successful decodes and this
  // side cannot tell which entries will decode.
  const bounded = boundMcpImageEnvelopes([shotOf("only", 12)]);

  assert.equal(
    splitMcpImages(bounded[0].content).images.length,
    MAX_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE,
  );
});

test("candidates survive the transport bound for the backend to decode", () => {
  // The backend's quota counts SUCCESSFUL decodes and scans past failures. This
  // side cannot decode, so cutting at the quota drops valid PNGs sitting behind
  // formats Pillow rejects -- shown on the first turn, lost on the replay.
  const bounded = boundMcpImageEnvelopes([shotOf("a", 8)]);

  assert.equal(
    splitMcpImages(bounded[0].content).images.length,
    MAX_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE,
  );
});

test("the spare candidates do not evict an older result", () => {
  const bounded = boundMcpImageEnvelopes([shotOf("old", 4), shotOf("new", 8)]);

  assert.equal(splitMcpImages(bounded[0].content).images.length, 4);
});

test("a partly-spent budget still leaves an older result its decode spares", () => {
  // A newer one-image result leaves 7 of 8. The older result has corrupt entries
  // ahead of valid PNGs, so slicing it to the remaining budget alone strands the
  // last valid one even though both quotas had room.
  const bounded = boundMcpImageEnvelopes([shotOf("old", 8), shotOf("new", 1)]);

  assert.equal(
    splitMcpImages(bounded[0].content).images.length,
    MAX_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE,
  );
  assert.equal(splitMcpImages(bounded[1].content).images.length, 1);
});
