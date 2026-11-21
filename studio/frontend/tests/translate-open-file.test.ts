// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import { SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS } from "../src/features/chat/open-document-accept.ts";

import { readSrc } from "./helpers/kit.ts";

const MAX_CHARS = 50_000;
const READ_BYTES = MAX_CHARS * 4 + 4;

const page = readSrc("features/translate/translate-page.tsx");
const fileHelpers = page.slice(
  page.indexOf("const TEXT_FILE_EXTENSIONS"),
  page.indexOf("const IS_MAC"),
);
const { readTextPrefix, serverUpload } = new Function(
  "SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS",
  "TRANSLATE_SOURCE_MAX_CHARS",
  `${
    ts.transpileModule(fileHelpers, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText
  }; return { readTextPrefix, serverUpload };`,
)(SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS, MAX_CHARS) as {
  readTextPrefix: (file: File) => Promise<string | null>;
  serverUpload: (file: File) => File;
};

test("text files not saved as UTF-8 go to the server to detect their encoding", async () => {
  const files = [
    new File([Buffer.from("Réunion du 12 décembre", "latin1")], "notes.txt", {
      type: "text/plain",
    }),
    new File([new Uint8Array([0x89, 0xef, 0x8b, 0x63])], "subs.srt"),
    new File(
      [Buffer.from("﻿Name\tCity\r\nJosé\tMünchen", "utf16le")],
      "export.txt",
      { type: "text/plain" },
    ),
    new File([Buffer.from("Name\tCity", "utf16le")], "plain16.txt"),
  ];
  for (const file of files) {
    assert.equal(await readTextPrefix(file), null, file.name);
  }
});

test("UTF-8 text files load exactly as saved", async () => {
  const cases: [string, string][] = [
    ["code.md", "    first\n    second\n"],
    ["table.tsv", "\tA\tB\r\n0\tx\ty\n"],
    ["Berlin.html", "<p>Berlin</p>"],
    ["README", "notes"],
  ];
  for (const [name, text] of cases) {
    const file = new File([text], name, { type: "text/plain" });
    assert.equal(await readTextPrefix(file), text, name);
  }
});

test("a Latin-1 file ending in an accented letter goes to the server", async () => {
  const file = new File([Buffer.from("café", "latin1")], "menu.txt");
  assert.equal(await readTextPrefix(file), null);
});

test("a UTF-8 character split by the read limit stays on the UTF-8 path", async () => {
  const text = `${"a".repeat(READ_BYTES - 1)}é`;
  const read = await readTextPrefix(new File([text], "long.txt"));
  assert.equal(read, "a".repeat(READ_BYTES - 1));
});

test("only a bounded prefix of a large text file is uploaded", () => {
  const big = new File([new Uint8Array(READ_BYTES * 3).fill(0xe9)], "big.log");
  const upload = serverUpload(big);
  assert.equal(upload.size, READ_BYTES);
  assert.equal(upload.name, "big.log.txt");
  const pdf = new File([new Uint8Array(READ_BYTES * 3)], "scan.pdf");
  assert.equal(serverUpload(pdf), pdf);
});

test("the server decodes text files as plain text, so HTML keeps its markup", () => {
  const html = new File(
    [Buffer.from("<p>Réunion</p>", "latin1")],
    "page.html",
    {
      type: "text/html",
    },
  );
  assert.equal(serverUpload(html).name, "page.html.txt");
});

test("a text file named without a known extension is still accepted by the server", () => {
  const readme = new File([Buffer.from("Grüße", "latin1")], "README", {
    type: "text/plain",
  });
  assert.equal(serverUpload(readme).name, "README.txt");
});

test("HTML that declares a legacy charset is decoded with it, markup and all", async () => {
  const pages: [string, string][] = [
    [
      '<meta charset="windows-1251"><p class="intro">Привет, мир!</p>',
      "3c6d65746120636861727365743d2277696e646f77732d31323531223e3c7020636c6173733d22696e74726f223ecff0e8e2e5f22c20ece8f0213c2f703e",
    ],
    [
      '<meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS"><p>今日は良い天気です。</p>',
      "3c6d65746120687474702d65717569763d22436f6e74656e742d547970652220636f6e74656e743d22746578742f68746d6c3b20636861727365743d53686966745f4a4953223e3c703e8da193fa82cd97c782a293568b4382c582b781423c2f703e",
    ],
  ];
  for (const [html, hex] of pages) {
    const file = new File([Buffer.from(hex, "hex")], "page.html", {
      type: "text/html",
    });
    assert.equal(await readTextPrefix(file), html);
  }
});
