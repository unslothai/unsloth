// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import { SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS } from "../src/features/chat/open-document-accept.ts";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/translate/translate-page.tsx");
const fileHelpers = page.slice(
  page.indexOf("const TEXT_FILE_EXTENSIONS"),
  page.indexOf("const IS_MAC"),
);
const readsInBrowser = new Function(
  "SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS",
  `${
    ts.transpileModule(fileHelpers, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText
  }; return readsInBrowser;`,
)(SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS) as (file: File) => boolean;

test("text files opened in Translate are decoded by the server, in the encoding they were saved in", () => {
  const files = [
    new File([Buffer.from("Réunion du 12 décembre", "latin1")], "notes.txt", {
      type: "text/plain",
    }),
    new File([new Uint8Array([0x89, 0xef, 0x8b, 0x63])], "subs.srt"),
    new File(
      [Buffer.from("\ufeffName\tCity\r\nJosé\tMünchen", "utf16le")],
      "export.txt",
      {
        type: "text/plain",
      },
    ),
    new File(["<p>Berlin</p>"], "Berlin.html", { type: "text/html" }),
  ];
  for (const file of files) {
    assert.equal(readsInBrowser(file), false, file.name);
  }
});

test("a text file of a type the server does not parse is still read in the browser", () => {
  assert.equal(
    readsInBrowser(new File(["notes"], "README", { type: "text/plain" })),
    true,
  );
});
