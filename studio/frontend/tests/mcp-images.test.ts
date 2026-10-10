// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DECODE_FAILURE_ALLOWANCE,
  MAX_MODEL_IMAGES,
  MAX_TOTAL_MCP_IMAGES,
  MAX_MCP_IMAGE_MIME_CHARS,
  MAX_TOTAL_MCP_IMAGE_CHARS,
  planMcpImageBound,
  MCP_IMAGES_MARKER,
  boundMcpImageEnvelopes,
  mcpImagesEnvelope,
  isImageToolName,
  splitMcpImages,
  stripMcpImageEnvelopes,
} from "../src/features/chat/api/mcp-images.ts";
import { providerModelTakesMcpImages } from "../src/features/chat/external-providers.ts";
import { localToolExchangeIndexes } from "../src/features/chat/codex-reasoning.ts";
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

test("only an image tool call carries the privileged envelope", () => {
  // Without the provenance gate a client tool's images would become model input.
  assert.equal(isMcpToolName("mcp__fs__read_media_file"), true);
  assert.equal(isMcpToolName("render_chart"), false);
  assert.equal(isMcpToolName(undefined), false);
  assert.match(
    adapter,
    /if \(isMcpImageToolResult\(result\) && isImageToolName\(tc\.toolName\)\) \{\n\s*content \+= mcpImagesEnvelope\(result\.images\);/,
  );
  // The wrapper branch is not gated, or JSON.stringify replays base64 as prompt text.
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
  // The backend cap runs after parsing, so it cannot bound transport.
  const messages = [0, 1, 2, 3, 4].map((n) => shot(n));

  const bounded = boundMcpImageEnvelopes(messages);

  assert.equal(countImages(messages), 15);
  assert.ok(countImages(bounded) < countImages(messages), "nothing was bounded");
  assert.ok(
    countImages(bounded) <= MAX_TOTAL_MCP_IMAGES + DECODE_FAILURE_ALLOWANCE,
    `uploaded ${countImages(bounded)} candidates`,
  );
  assert.equal(splitMcpImages(bounded[0].content).images.length, 0);
});

test("the newest pictures are the ones kept", () => {
  const bounded = boundMcpImageEnvelopes([0, 1, 2, 3, 4].map((n) => shot(n)));

  assert.equal(splitMcpImages(bounded[4].content).images.length, 3);
  assert.equal(splitMcpImages(bounded[0].content).images.length, 0);
  assert.match(bounded[0].content, /^\[3 images returned\]$/);
});

test("a conversation inside the budget is left byte-identical", () => {
  const messages = [0, 1].map((n) => shot(n));

  assert.deepEqual(boundMcpImageEnvelopes(messages), messages);
});

test("a non-MCP tool result keeps its text and loses only the envelope", () => {
  const message = {
    role: "tool",
    name: "read_file",
    content: "plain text" + mcpImagesEnvelope([{ data: "QUJD", mimeType: "image/png" }]),
  };

  const [bounded] = boundMcpImageEnvelopes([message]);

  assert.equal(bounded.content, "plain text");
  assert.equal(bounded.name, "read_file");
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
  const bounded = boundMcpImageEnvelopes([shotOf("old", 4), shotOf("new", 8)]);

  assert.equal(splitMcpImages(bounded[0].content).images.length, 4);
});

test("no single result uploads more than the backend could ever decode", () => {
  // The backend counts successful decodes, which this side cannot predict.
  const bounded = boundMcpImageEnvelopes([shotOf("only", 12)]);

  assert.equal(
    splitMcpImages(bounded[0].content).images.length,
    MAX_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE,
  );
});

test("candidates survive the transport bound for the backend to decode", () => {
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
  const bounded = boundMcpImageEnvelopes([shotOf("old", 8), shotOf("new", 1)]);

  assert.equal(
    splitMcpImages(bounded[0].content).images.length,
    MAX_MODEL_IMAGES + DECODE_FAILURE_ALLOWANCE,
  );
  assert.equal(splitMcpImages(bounded[1].content).images.length, 1);
});

test("the decode-failure allowance survives an undecodable newest result", () => {
  const undecodable = Array.from({ length: 4 }, (_, i) => ({
    data: `SVG${i}`,
    mimeType: "image/svg+xml",
  }));
  const valid = (tag: string) =>
    Array.from({ length: 4 }, (_, i) => ({ data: `${tag}${i}`, mimeType: "image/png" }));

  const messages = [
    { role: "tool", name: "mcp__s__a", content: "oldest" + mcpImagesEnvelope(valid("A")) },
    { role: "tool", name: "mcp__s__b", content: "middle" + mcpImagesEnvelope(valid("B")) },
    { role: "tool", name: "mcp__s__c", content: "newest" + mcpImagesEnvelope(undecodable) },
  ];

  const bounded = boundMcpImageEnvelopes(messages);
  const kept = bounded.map((m) => splitMcpImages(m.content as string).images);

  assert.equal(kept[2].length, 4, "the newest result is kept as candidates");
  assert.equal(kept[1].length, 4, "the middle result keeps its four real pictures");
  assert.equal(
    kept[0].length,
    4,
    "the oldest result still ships candidates on the shared allowance",
  );

  const total = kept.reduce((n, images) => n + images.length, 0);
  assert.ok(
    total <= MAX_TOTAL_MCP_IMAGES + DECODE_FAILURE_ALLOWANCE,
    `bounded: ${total} candidates uploaded`,
  );
});

test("a replayed history is bounded by bytes, not only by image count", () => {
  // One result may carry 12M chars of base64, so a size bound is also needed.
  const huge = "A".repeat(4_000_000);
  const messages = Array.from({ length: 6 }, (_, i) => ({
    role: "tool",
    name: `mcp__s__shot${i}`,
    content:
      `result ${i}` +
      mcpImagesEnvelope([{ data: huge, mimeType: "image/png" }]),
  }));

  const bounded = boundMcpImageEnvelopes(messages);
  const kept = bounded.flatMap((m) => splitMcpImages(m.content as string).images);
  const chars = kept.reduce((n, image) => n + image.data.length, 0);

  assert.ok(
    chars <= MAX_TOTAL_MCP_IMAGE_CHARS,
    `${chars} characters uploaded against a budget of ${MAX_TOTAL_MCP_IMAGE_CHARS}`,
  );
  assert.ok(kept.length > 0, "the budget must not starve the newest result");
  const newest = splitMcpImages(bounded[5].content as string).images;
  assert.equal(newest.length, 1, "the newest result keeps its picture");
  assert.equal(
    splitMcpImages(bounded[0].content as string).images.length,
    0,
    "the oldest gives its bytes up first",
  );
});

test("an ordinary conversation is untouched by the byte budget", () => {
  const small = "B".repeat(2048);
  const messages = Array.from({ length: 2 }, (_, i) => ({
    role: "tool",
    name: `mcp__s__shot${i}`,
    content: `r${i}` + mcpImagesEnvelope([{ data: small, mimeType: "image/png" }]),
  }));

  const bounded = boundMcpImageEnvelopes(messages);

  assert.deepEqual(bounded, messages, "nothing was rewritten");
});

test("an oversized replay image is skipped, not the rest of its result", () => {
  const newest = "N".repeat(9_000_000);
  const oversized = "X".repeat(5_000_000);
  const small = (tag: string) => tag.repeat(900_000);

  const messages = [
    {
      role: "tool",
      name: "mcp__s__old",
      content:
        "old" +
        mcpImagesEnvelope([
          { data: oversized, mimeType: "image/png" },
          { data: small("a"), mimeType: "image/png" },
          { data: small("b"), mimeType: "image/png" },
        ]),
    },
    { role: "tool", name: "mcp__s__new", content: "new" + mcpImagesEnvelope([{ data: newest, mimeType: "image/png" }]) },
  ];

  const bounded = boundMcpImageEnvelopes(messages);
  const older = splitMcpImages(bounded[0].content as string).images;

  assert.ok(
    !older.some((image) => image.data === oversized),
    "the oversized candidate should not fit",
  );
  assert.equal(older.length, 2, "the two that DO fit must survive it");
  const chars = bounded
    .flatMap((m) => splitMcpImages(m.content as string).images)
    .reduce((n, image) => n + image.data.length, 0);
  assert.ok(chars <= MAX_TOTAL_MCP_IMAGE_CHARS, `${chars} characters uploaded`);
});

test("a bounded envelope records how many the tool returned", () => {
  const images = Array.from({ length: 20 }, (_, i) => ({ data: `IMG${i}`, mimeType: "image/png" }));
  const [bounded] = boundMcpImageEnvelopes([
    { role: "tool", name: "mcp__s__shot", content: "r" + mcpImagesEnvelope(images) },
  ]);
  const kept = splitMcpImages(bounded.content as string).images;
  assert.ok(kept.length < 20, "the bound must actually have cut");
  assert.equal(kept[0].returned, 20, "the original count survives on the first entry");
});

test("a named non-MCP result has its envelope stripped, not re-uploaded", () => {
  const huge = "Z".repeat(3_000_000);
  const [bounded] = boundMcpImageEnvelopes([
    { role: "tool", name: "read_file", content: "here" + mcpImagesEnvelope([{ data: huge, mimeType: "image/png" }]) },
  ]);
  assert.equal(bounded.content, "here");
});

test("a text-only target is sent no envelopes at all", () => {
  const messages = [
    { role: "user", content: "what did it show" },
    {
      role: "tool",
      name: "mcp__s__shot",
      content: "[1 image returned]" + mcpImagesEnvelope([{ data: "A".repeat(50_000), mimeType: "image/png" }]),
    },
    { role: "tool", name: "read_file", content: "plain" },
  ];

  const stripped = stripMcpImageEnvelopes(messages);

  assert.equal(stripped[1].content, "[1 image returned]");
  assert.equal(stripped[0], messages[0], "non-tool messages are the same object");
  assert.equal(stripped[2], messages[2], "a tool result with no envelope is untouched");
});

const round = (n: number, results = 1, perResult = 3) => [
  { role: "assistant", content: `call ${n}` },
  ...Array.from({ length: results }, (_, k) => ({
    role: "tool",
    name: "mcp__fs__screenshot",
    content:
      `[${perResult} images returned]` +
      mcpImagesEnvelope(
        Array.from({ length: perResult }, (_, m) => ({
          data: `R${n}K${k}M${m}`,
          mimeType: "image/png",
        })),
      ),
  })),
  { role: "assistant", content: `answer ${n}` },
];

const imagesPerToolResult = (messages: { role?: string; content?: unknown }[]) =>
  messages
    .filter((m) => m.role === "tool" && typeof m.content === "string")
    .map((m) => splitMcpImages(m.content as string).images.length);

test("a marker target is charged one picture per round, so every round keeps one", () => {
  // Eight rounds of three: the backend's local path replays one per round.
  const messages = Array.from({ length: 8 }, (_, n) => round(n)).flat();

  const parts = imagesPerToolResult(boundMcpImageEnvelopes(messages));
  assert.ok(parts.some((n) => n === 0), `default still strips old rounds: ${parts}`);

  const markers = imagesPerToolResult(
    boundMcpImageEnvelopes(messages, { localMarkers: true }),
  );
  assert.equal(markers.length, 8);
  assert.ok(markers.every((n) => n >= 1), `every round keeps a candidate: ${markers}`);
  // Spares ride with the newest rounds only; the rest carry exactly their one.
  assert.equal(markers[markers.length - 1], 3);
  assert.equal(markers[0], 1);
});

test("consecutive results are one batch on a marker target and share one charge", () => {
  const messages = Array.from({ length: 8 }, (_, n) => round(n, 3, 2)).flat();

  const markers = imagesPerToolResult(
    boundMcpImageEnvelopes(messages, { localMarkers: true }),
  );
  assert.equal(markers.length, 24);
  for (let n = 0; n < 8; n++) {
    const batch = markers.slice(n * 3, n * 3 + 3);
    assert.ok(batch.some((k) => k >= 1), `round ${n} keeps a candidate: ${batch}`);
  }
});

test("the send path bounds the run's own results before serializing them", () => {
  assert.match(
    adapter,
    /const mcpImagesLocalMarkers =\n\s*!isExternalRequest &&\n\s*runtime\.models\.find\(\(model\) => model\.id === runtime\.params\.checkpoint\)\n\s*\?\.isGguf === false;/,
  );
  assert.match(
    adapter,
    /const messages = boundMcpImageResults\(rawMessages, \{\n\s*readsImages: targetReadsImages,\n\s*localMarkers: mcpImagesLocalMarkers,\n\s*\}\);\n\s*const replayReasoning = [\s\S]*?;\n\s*const survivingMessages = pruneOutboundHistory\(messages, replayReasoning\);/,
  );
  assert.doesNotMatch(adapter, /boundMcpImageEnvelopes\(outboundMessages/);
  assert.doesNotMatch(adapter, /stripMcpImageEnvelopes\(outboundMessages/);
});

test("the upload gate is the backend's external MCP gate, not provider-level vision", () => {
  // The backend sends no images to unknown models in mixed catalogs.
  assert.equal(providerModelTakesMcpImages("openrouter", "some/unknown-model"), false);
  assert.equal(providerModelTakesMcpImages("huggingface", "org/unknown"), false);
  assert.equal(providerModelTakesMcpImages("qwen", "qwen2.5-72b-instruct"), false);
  assert.equal(providerModelTakesMcpImages("mistral", "mistral-large"), false);
  assert.equal(providerModelTakesMcpImages("anthropic", "claude-x"), true);
  assert.equal(providerModelTakesMcpImages(undefined, undefined), true);
});

test("both local paths read the model's vision flag, not just multimodal", () => {
  assert.match(
    adapter,
    /function localTargetReadsImages\([\s\S]*?if \(typeof activeModel\?\.isVision === "boolean"\) return activeModel\.isVision;\n\s*return state\.loadedIsMultimodal !== false;/,
  );
  assert.match(adapter, /: localTargetReadsImages\(runtime\);/);
  assert.match(
    adapter,
    /const messages = boundMcpImageResults\(rawMessages, \{\n\s*readsImages: localTargetReadsImages\(runtimeState\),/,
  );
  assert.match(
    adapter,
    /\? providerModelTakesMcpImages\(\n\s*externalProvider\?\.providerType,\n\s*externalSelection\?\.modelId,\n\s*\)/,
  );
});

test("an entry with an unbounded mimeType is dropped, not charged", () => {
  // A token mimeType subtype has no length bound, so it is charged too.
  const heavy = (n: number) => ({
    role: "tool",
    name: "mcp__fs__screenshot",
    content:
      "[1 image returned]" +
      mcpImagesEnvelope([
        { data: `A${n}`, mimeType: `image/${"x".repeat(MAX_MCP_IMAGE_MIME_CHARS + 1)}` },
        { data: `B${n}`, mimeType: "image/png" },
      ]),
  });
  const bounded = boundMcpImageEnvelopes([0, 1].map(heavy));
  for (const message of bounded) {
    const kept = splitMcpImages(message.content).images;
    assert.deepEqual(
      kept.map((image) => image.mimeType),
      ["image/png"],
      "the bounded-mime entry stays, the unbounded one goes",
    );
  }
});

test("a picture the live turn accepted at the limit still fits its own replay", () => {
  const messages = [
    {
      role: "tool",
      name: "mcp__fs__screenshot",
      content:
        "[1 image returned]" +
        mcpImagesEnvelope([
          { data: "A".repeat(MAX_TOTAL_MCP_IMAGE_CHARS), mimeType: "image/png" },
        ]),
    },
  ];
  const [bounded] = boundMcpImageEnvelopes(messages);
  assert.equal(splitMcpImages(bounded.content).images.length, 1);
});

test("the byte filter scans past candidates that do not fit", () => {
  const big = (tag: string, n: number, size: number) =>
    Array.from({ length: n }, (_, i) => ({ data: tag.repeat(size / tag.length) + i, mimeType: "image/png" }));
  const messages = [
    {
      role: "tool",
      name: "mcp__fs__a",
      content: "older" + mcpImagesEnvelope([...big("O", 8, 1_400_000), { data: "tiny", mimeType: "image/png" }]),
    },
    { role: "assistant", content: "next" },
    {
      role: "tool",
      name: "mcp__fs__b",
      content: "newer" + mcpImagesEnvelope(big("N", 1, 11_000_000)),
    },
  ];
  const bounded = boundMcpImageEnvelopes(messages);
  const older = splitMcpImages(bounded[0].content).images;
  assert.deepEqual(older.map((image) => image.data), ["tiny"]);
});

test("the planner is the bound the envelope form applies", () => {
  // Both carriers plan through one function so the bounds agree.
  const rounds = Array.from({ length: 8 }, (_, n) => round(n, 3, 2)).flat();
  const envelopes = boundMcpImageEnvelopes(rounds, { localMarkers: true });
  const viaEnvelopes = imagesPerToolResult(envelopes);
  const batches = Array.from({ length: 8 }, (_, n) =>
    rounds
      .slice(n * 5 + 1, n * 5 + 4)
      .map((m) => splitMcpImages(m.content as string).images),
  );
  const viaPlanner = planMcpImageBound(batches, { localMarkers: true })
    .flat()
    .map((kept) => kept.length);
  assert.deepEqual(viaPlanner, viaEnvelopes);
  const dropped = planMcpImageBound([[[{ data: "x", mimeType: "image/png" }]], [[Array.from({ length: 12 }, (_, i) => ({ data: `${i}`, mimeType: "image/png" }))].flat()]]);
  assert.equal(dropped[0][0].length, 1);
});

test("a message's results are batched by replay exchange, not as one block", () => {
  // Local rounds share one assistant message but serialize as separate exchanges.
  type P = { round: number | null; flush: boolean };
  const parts: P[] = [
    { round: 0, flush: false },
    { round: 0, flush: false },
    { round: 1, flush: false },
    { round: 2, flush: false },
    { round: null, flush: true },
    { round: null, flush: true },
  ];
  const indexes = localToolExchangeIndexes(
    parts,
    (p) => p.round,
    (p) => p.flush,
  );
  assert.deepEqual(indexes, [0, 0, 1, 2, 3, 4]);
  assert.match(adapter, /startsNewCodexToolRound\(pendingLocalToolRoundId, localRoundId\)/);
  assert.match(adapter, /localRoundId === null && shouldFlushCompletedLocalToolPair\(toolPart\)/);
  assert.match(
    adapter,
    /localToolExchangeIndexes\(\n\s*toolParts,\n\s*\(\{ part \}\) => codexLocalToolRoundId\(getToolReplayProvenance\(part\)\),\n\s*\(\{ part \}\) => shouldFlushCompletedLocalToolPair\(part\),/,
  );
});

test("a client tool's structured result is not unwrapped as the MCP wrapper", () => {
  // Only the bare live-parser wrapper is unwrapped; other fields must survive.
  assert.match(
    adapter,
    /\(isMcpImageToolResult\(result\) &&\n\s*\(isImageToolName\(tc\.toolName\) \|\| isBareMcpImageWrapper\(result\)\)\) \|\|/,
  );
  assert.match(
    adapter,
    /export function isBareMcpImageWrapper\(val: unknown\): boolean \{\n\s*if \(!isMcpImageToolResult\(val\)\) return false;\n\s*const keys = Object\.keys\(val as object\)\.filter\(\(key\) => key !== "text" && key !== "images"\);\n\s*return keys\.length === 0;/,
  );
});


test("sandbox image viewer replays bounded images without trusting arbitrary tools", () => {
  assert.equal(isImageToolName("view_image"), true);
  assert.equal(isImageToolName("python"), false);
  assert.equal(isImageToolName("mcp__files__image"), true);
  const messages = [
    { role: "tool", name: "view_image", content: RESULT },
    { role: "tool", name: "python", content: RESULT },
  ];
  const bounded = boundMcpImageEnvelopes(messages);
  assert.equal(splitMcpImages(bounded[0].content).images.length, 1);
  assert.equal(splitMcpImages(bounded[1].content).images.length, 0);
});
