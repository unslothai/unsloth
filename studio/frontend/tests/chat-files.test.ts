// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { collectChatFiles } from "../src/features/chat/lib/chat-files.ts";

import { readText } from "./helpers/kit.ts";

const sandboxResult = (
  sessionId: string,
  files: string[],
  images: string[] = [],
) => ({
  text: "done",
  sessionId,
  images,
  files: files.map((name) => ({ name, size: 10 })),
});

const python = (id: string, result: unknown) => ({
  id,
  role: "assistant",
  content: [{ type: "tool-call", toolName: "python", args: {}, result }],
});

test("lists tool files, render_html pages, generated images and fenced code", () => {
  const messages = [
    {
      id: "u1",
      role: "user",
      content: [{ type: "text", text: "```py\nignored()\n```" }],
    },
    python("a1", sandboxResult("t1", ["out/report.pdf"], ["plot.png"])),
    {
      id: "a2",
      role: "assistant",
      content: [
        { type: "text", text: "Here:\n```python\nprint(1)\n```\n" },
        {
          type: "tool-call",
          toolName: "render_html",
          args: { code: "<p>x</p>", title: "Demo" },
          result: "ok",
        },
        {
          type: "tool-call",
          toolName: "image_generation",
          args: {},
          result: { image_b64: "AAAA", image_mime: "image/webp" },
        },
      ],
    },
  ];
  const files = collectChatFiles(messages);
  assert.deepEqual(
    files.map((f) => [f.name, f.kind, f.messageId]),
    [
      ["report.pdf", "file", "a1"],
      ["plot.png", "image", "a1"],
      ["snippet.py", "code", "a2"],
      [files[3].name, "html", "a2"],
      ["generated-image.webp", "image", "a2"],
    ],
  );
  assert.equal(files[0].sandbox?.name, "out/report.pdf");
  assert.equal(files[4].dataUrl, "data:image/webp;base64,AAAA");
});

test("the same sandbox file named twice is listed once; equal names get a suffix", () => {
  const files = collectChatFiles([
    python("a1", sandboxResult("t1", ["data.csv"])),
    python("a2", sandboxResult("t1", ["data.csv", "sub/data.csv"])),
  ]);
  assert.deepEqual(
    files.map((f) => f.name),
    ["data.csv", "data-2.csv"],
  );
});

test("folder files no message names are appended without a message", () => {
  const files = collectChatFiles(
    [python("a1", sandboxResult("t1", ["a.txt"]))],
    {
      sessionId: "t1",
      names: ["a.txt", "uploaded.xlsx"],
    },
  );
  assert.deepEqual(
    files.map((f) => [f.name, f.messageId]),
    [
      ["a.txt", "a1"],
      ["uploaded.xlsx", null],
    ],
  );
  assert.deepEqual(files[1].sandbox, {
    sessionId: "t1",
    name: "uploaded.xlsx",
  });
});

test("a file the folder no longer holds is dropped, but only for that folder's session", () => {
  const messages = [
    python("a1", sandboxResult("t1", ["gone.txt", "kept.txt"])),
    python("a2", sandboxResult("other", ["elsewhere.txt"])),
  ];
  const files = collectChatFiles(messages, {
    sessionId: "t1",
    names: ["kept.txt"],
  });
  assert.deepEqual(
    files.map((f) => f.name),
    ["kept.txt", "elsewhere.txt"],
  );
});

test("a listing at the backend cap is not trusted to prove absence", () => {
  const names = Array.from({ length: 2000 }, (_, i) => `f${i}.txt`);
  const files = collectChatFiles(
    [python("a1", sandboxResult("t1", ["late.txt"]))],
    {
      sessionId: "t1",
      names,
    },
  );
  assert.equal(files[0].name, "late.txt");
  assert.equal(files.length, 2001);
});

test("a hostile image mime cannot inject path characters into the file name", () => {
  const files = collectChatFiles([
    {
      id: "a1",
      role: "assistant",
      content: [
        {
          type: "tool-call",
          toolName: "image_generation",
          result: { image_b64: "AA", image_mime: "image/../x;y" },
        },
      ],
    },
  ]);
  assert.match(files[0].name, /^generated-image\.[a-z0-9]+$/);
});

test("the listing cap mirrors the backend's _MAX_SNAPSHOT_FILES", () => {
  const backend = readText("../../backend/core/inference/tools.py");
  const source = readText("../src/features/chat/lib/chat-files.ts");
  const cap = /^_MAX_SNAPSHOT_FILES = (\d+)/m.exec(backend)?.[1];
  assert.ok(cap);
  assert.match(source, new RegExp(`FOLDER_LISTING_CAP = ${cap};`));
});
