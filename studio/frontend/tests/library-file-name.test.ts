// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A Library item's shown name can be renamed to anything. What Download saves and "Chat about
// this" attaches must still be a name every OS accepts, with the extension that says what the
// bytes are, and typed so the composer routes it to the right adapter.

import assert from "node:assert/strict";
import test from "node:test";

import {
  embeddedBlobType,
  libraryFileName,
  libraryFileType,
} from "../src/features/library/file-name.ts";

test("a rename that dropped the extension gets the file's own back", () => {
  assert.equal(libraryFileName({ name: "Q3 report", fileName: "report.pdf" }), "Q3 report.pdf");
  assert.equal(libraryFileName({ name: "notes.v2", fileName: "notes.md" }), "notes.v2.md");
  assert.equal(libraryFileName({ name: "script.PY", fileName: "script.py" }), "script.PY");
});

test("characters Windows refuses are replaced, and trailing dots and spaces dropped", () => {
  assert.equal(
    libraryFileName({ name: 'a<b>c:d"e/f\\g|h?i*j', fileName: "x.txt" }),
    "a_b_c_d_e_f_g_h_i_j.txt",
  );
  assert.equal(libraryFileName({ name: "tab\there\u0007", fileName: "x.md" }), "tab_here_.md");
  assert.equal(libraryFileName({ name: "draft. . .", fileName: "draft.md" }), "draft.md");
  assert.equal(libraryFileName({ name: "../../etc/passwd", fileName: "a.txt" }), ".._.._etc_passwd.txt");
});

test("reserved device names are prefixed, whatever follows them", () => {
  assert.equal(libraryFileName({ name: "CON", fileName: "con.txt" }), "_CON.txt");
  assert.equal(libraryFileName({ name: "nul.backup", fileName: "x.txt" }), "_nul.backup.txt");
  assert.equal(libraryFileName({ name: "com1", fileName: "a.py" }), "_com1.py");
  assert.equal(libraryFileName({ name: "console", fileName: "a.py" }), "console.py");
});

test("an empty or dot-only name still saves, and a dotfile keeps its name", () => {
  assert.equal(libraryFileName({ name: "", fileName: "image.png" }), "file.png");
  assert.equal(libraryFileName({ name: "...", fileName: "image.png" }), "file.png");
  assert.equal(libraryFileName({ name: ".env", fileName: ".env" }), ".env");
});

test("text-only chat uploads say so with .txt", () => {
  assert.equal(libraryFileName({ name: "paper.pdf", textOnly: true }), "paper.pdf.txt");
  assert.equal(libraryFileName({ name: "notes.txt", textOnly: true }), "notes.txt");
});

test("a very long name is cut, keeping its extension", () => {
  const name = libraryFileName({ name: "x".repeat(400), fileName: "a.jsonl" });
  assert.equal(name.length, 200);
  assert.ok(name.endsWith(".jsonl"));
});

test("a file typed opaquely by the server gets a type from its extension", () => {
  assert.equal(libraryFileType("run.py", ""), "text/plain");
  assert.equal(libraryFileType("run.py", "application/octet-stream"), "text/plain");
  assert.equal(libraryFileType("photo.JPG", ""), "image/jpeg");
  assert.equal(libraryFileType("blob.bin", ""), "application/octet-stream");
  assert.equal(libraryFileType("run.py", "text/x-python; charset=utf-8"), "text/x-python");
});

test("nothing embedded directly is typed as a scriptable document", () => {
  assert.equal(embeddedBlobType("pdf", "text/html"), "application/pdf");
  assert.equal(embeddedBlobType("pdf", "application/octet-stream"), "application/pdf");
  assert.equal(embeddedBlobType("image", "image/svg+xml"), "application/octet-stream");
  assert.equal(embeddedBlobType("image", "text/html"), "application/octet-stream");
  assert.equal(embeddedBlobType("image", "image/png"), "image/png");
  assert.equal(embeddedBlobType("video", "video/mp4"), "video/mp4");
  assert.equal(embeddedBlobType("audio", "video/mp4"), "application/octet-stream");
});
