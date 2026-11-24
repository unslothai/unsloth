// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  collectChatFiles,
  createChatFilesCollector,
} from "../src/features/chat/lib/chat-files.ts";

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
    [
      { sessionId: "t1", names: ["a.txt", "uploaded.xlsx"] },
      { sessionId: "project-p", names: ["shared.md"] },
    ],
  );
  assert.deepEqual(
    files.map((f) => [f.name, f.messageId]),
    [
      ["a.txt", "a1"],
      ["uploaded.xlsx", null],
      ["shared.md", null],
    ],
  );
  assert.deepEqual(files[1].sandbox, {
    sessionId: "t1",
    name: "uploaded.xlsx",
  });
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

test("a deleted folder file stays listed: the listing may be cut short, so absence proves nothing", () => {
  const files = collectChatFiles(
    [python("a1", sandboxResult("t1", ["kept.txt"]))],
    [{ sessionId: "t1", names: [] }],
  );
  assert.deepEqual(
    files.map((f) => f.name),
    ["kept.txt"],
  );
});

test("an indented fence downloads without the opener's indentation", () => {
  const files = collectChatFiles([
    {
      id: "a1",
      role: "assistant",
      content: [
        {
          type: "text",
          text: "1. Run:\n   ```python\n   def answer():\n       return 42\n   ```\n",
        },
      ],
    },
  ]);
  assert.equal(files[0].text, "def answer():\n    return 42");
});

test("a fence opened right after a list marker closes on its own fence", () => {
  const text = "- ```python\n  print(1)\n  ```\n\n```js\nconsole.log(2)\n```\n";
  const files = collectChatFiles([
    { id: "a1", role: "assistant", content: [{ type: "text", text }] },
  ]);
  assert.deepEqual(
    files.map((f) => [f.name, f.text]),
    [
      ["snippet.py", "print(1)"],
      ["snippet.js", "console.log(2)"],
    ],
  );
});

const fenceText = (text: string) =>
  collectChatFiles([
    { id: "a1", role: "assistant", content: [{ type: "text", text }] },
  ]);

test("a thematic break of repeated list markers scans in linear time", () => {
  const start = performance.now();
  fenceText(`${"-  ".repeat(5000)}\n`);
  assert.ok(performance.now() - start < 200);
});

test("a four-space-indented fence line is code, not a top-level closer", () => {
  const files = fenceText(
    '```python\ndef f():\n    """\n    ```\n    """\n```\n',
  );
  assert.equal(files[0].text, 'def f():\n    """\n    ```\n    """');
});

test("CRLF replies keep their code blocks", () => {
  const files = fenceText("```python\r\nprint(1)\r\n```\r\n");
  assert.equal(files[0].text, "print(1)");
});

test("a fence in a list item's continuation paragraph is found with its indentation removed", () => {
  const files = fenceText(
    "1. Step one\n\n   Then:\n\n   ```python\n   x = 1\n   ```\n",
  );
  assert.deepEqual(
    files.map((f) => [f.name, f.text]),
    [["snippet.py", "x = 1"]],
  );
});

test("standard assistant image parts (imported chats) are listed", () => {
  const files = collectChatFiles([
    {
      id: "a1",
      role: "assistant",
      content: [{ type: "image", image: "data:image/jpeg;base64,AA" }],
    },
  ]);
  assert.deepEqual(
    files.map((f) => [f.name, f.kind, f.dataUrl]),
    [["generated-image.jpg", "image", "data:image/jpeg;base64,AA"]],
  );
});

test("a suffixed name never repeats a name already listed", () => {
  const files = collectChatFiles([
    python(
      "a1",
      sandboxResult("t1", ["data.csv", "sub/data.csv", "data-2.csv"]),
    ),
  ]);
  assert.deepEqual(
    files.map((f) => f.name),
    ["data.csv", "data-2.csv", "data-2-2.csv"],
  );
});

test("remote assistant image URLs are left out", () => {
  const files = collectChatFiles([
    {
      id: "a1",
      role: "assistant",
      content: [{ type: "image", image: "https://example.com/cat.png" }],
    },
  ]);
  assert.equal(files.length, 0);
});

test("the collector reuses its result while the settled messages are unchanged", () => {
  const collect = createChatFilesCollector();
  const first = python("a1", sandboxResult("t1", ["a.txt"]));
  const folders = [{ sessionId: "t1", names: ["a.txt"] }];
  const before = collect([first], folders);
  assert.equal(collect([first], folders), before);
  const after = collect(
    [first, python("a2", sandboxResult("t1", ["b.txt"]))],
    folders,
  );
  assert.notEqual(after, before);
  assert.deepEqual(
    after.map((f) => f.name),
    ["a.txt", "b.txt"],
  );
});
