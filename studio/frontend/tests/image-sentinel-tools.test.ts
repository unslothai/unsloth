// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

const {
  IMAGE_SENTINEL_TOOLS,
  MEDIA_CHAT_TOOLS,
  SANDBOX_FILE_TOOLS,
  SANDBOX_RUN_TOOLS,
  hasCreatedFiles,
} = await import("../src/components/assistant-ui/sandbox-files.ts");

test("only the tools that emit the image envelope are in the set", () => {
  for (const emitter of ["python", "terminal", "code_execution"]) {
    assert.ok(IMAGE_SENTINEL_TOOLS.has(emitter), emitter);
  }
  for (const reader of ["mcp__fs__read_file", "web_search", "search_knowledge_base", ""]) {
    assert.ok(!IMAGE_SENTINEL_TOOLS.has(reader), reader);
  }
  for (const sandbox of SANDBOX_RUN_TOOLS) {
    assert.ok(IMAGE_SENTINEL_TOOLS.has(sandbox), `${sandbox} emits both envelopes`);
  }
  // edit_file emits the file envelope only.
  assert.ok(!IMAGE_SENTINEL_TOOLS.has("edit_file"));
});

test("a file edit_file created gets the files row", () => {
  assert.ok(SANDBOX_FILE_TOOLS.has("edit_file"));
  assert.ok(!SANDBOX_RUN_TOOLS.has("edit_file"));
  const wrapped = {
    text: "Created index.html (2 lines)",
    images: [],
    sessionId: "chat-1",
    files: [{ name: "index.html", size: 12 }],
  };
  assert.ok(hasCreatedFiles("edit_file", wrapped));
  assert.ok(!hasCreatedFiles("edit_file", "Edited index.html (1 replacement)"));
  assert.ok(!hasCreatedFiles("mcp__fs__write_file", wrapped));
});

// Only tools that emit the envelope may be sliced; otherwise the card fetches a file never written.
test("the adapter slices the image envelope only for the tools that emit it", () => {
  const source = readSrc("features/chat/api/chat-adapter.ts");
  const gate = source.indexOf(
    'IMAGE_SENTINEL_TOOLS.has(\n                      toolCallParts[idx].toolName ?? "",\n                    )',
  );
  const slice = source.indexOf("rawResult.lastIndexOf(imgMarker)");
  assert.ok(gate >= 0, "the __IMAGES__ slice is no longer gated on IMAGE_SENTINEL_TOOLS");
  assert.ok(slice > gate, "the slice runs before the gate");
  assert.equal(
    source.indexOf("rawResult.lastIndexOf(imgMarker)", slice + 1),
    -1,
    "a second, ungated __IMAGES__ slice appeared",
  );
});

test("the media chat tools emit the file envelope, and generate_image the image one", () => {
  assert.deepEqual([...MEDIA_CHAT_TOOLS].sort(), ["generate_image", "generate_video"]);
  for (const tool of MEDIA_CHAT_TOOLS) {
    assert.ok(SANDBOX_FILE_TOOLS.has(tool), `${tool} replays as its text, not the wrapper`);
    assert.ok(!SANDBOX_RUN_TOOLS.has(tool), `${tool} runs no code`);
  }
  assert.ok(IMAGE_SENTINEL_TOOLS.has("generate_image"));
  // The video is a file, never an __IMAGES__ entry.
  assert.ok(!IMAGE_SENTINEL_TOOLS.has("generate_video"));
  assert.ok(
    hasCreatedFiles("generate_video", {
      text: "Generated a 4 s video.",
      images: [],
      sessionId: "chat-1",
      files: [{ name: "generated-video-1a2b3c4d.mp4", size: 1024 }],
    }),
  );
});

test("the frontend media tool set matches the backend's", () => {
  const backend = readText("../../backend/core/inference/media_chat_tools.py");
  const match = backend.match(/^MEDIA_CHAT_TOOLS = frozenset\(\{([^}]*)\}\)/m);
  assert.ok(match, "media_chat_tools.py no longer declares MEDIA_CHAT_TOOLS as a frozenset literal");
  const names = [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort();
  assert.deepEqual(names, [...MEDIA_CHAT_TOOLS].sort());
});

test("the hosted image tool name is never stripped as a local media tool, nor the reverse", () => {
  // SERVER_SIDE_BUILTIN_TOOL_NAMES drops a part from outbound history; the local tools' results
  // are real tool messages the model must see again.
  for (const path of ["features/chat/utils/provider-compaction.ts", "features/chat/api/chat-adapter.ts"]) {
    const source = readSrc(path);
    const start = source.indexOf("const SERVER_SIDE_BUILTIN_TOOL_NAMES");
    assert.ok(start >= 0, path);
    const block = source.slice(start, source.indexOf("]);", start));
    assert.ok(block.includes('"image_generation"'), path);
    for (const tool of MEDIA_CHAT_TOOLS) assert.ok(!block.includes(tool), `${path}: ${tool}`);
  }
  assert.ok(!MEDIA_CHAT_TOOLS.has("image_generation"));
});
