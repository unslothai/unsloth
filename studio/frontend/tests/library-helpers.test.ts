// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

import {
  embeddedBlobType,
  itemVersion,
  libraryFileName,
  libraryFileType,
  streamsPreview,
  uniqueFileNames,
} from "../src/features/library/file-name.ts";
import { decodeNote, encodeNote } from "../src/features/library/note-text.ts";
import {
  acquireObjectUrl,
  cachedObjectUrlCount,
  clearCachedObjectUrls,
} from "../src/features/library/object-url-cache.ts";
import { isLoopbackHost, revealLabelFor } from "../src/features/library/reveal-label.ts";

function table<A extends unknown[], O>(name: string, run: (...args: A) => O, rows: [A, O][]) {
  test(name, () => {
    for (const [args, expected] of rows) assert.deepEqual(run(...args), expected, String(args));
  });
}

const fileName = (name: string, file?: string, textOnly?: boolean) =>
  libraryFileName({ name, fileName: file, textOnly });

table("a renamed item saves under a name every OS takes, with its file's extension", fileName, [
  [["Q3 report", "report.pdf"], "Q3 report.pdf"],
  [["notes.v2", "notes.md"], "notes.v2.md"],
  [["script.PY", "script.py"], "script.PY"],
  [['a<b>c:d"e/f\\g|h?i*j', "x.txt"], "a_b_c_d_e_f_g_h_i_j.txt"],
  [["tab\there\u0007", "x.md"], "tab_here_.md"],
  [["draft. . .", "draft.md"], "draft.md"],
  [["../../etc/passwd", "a.txt"], ".._.._etc_passwd.txt"],
  [["CON", "con.txt"], "_CON.txt"],
  [["nul.backup", "x.txt"], "_nul.backup.txt"],
  [["com1", "a.py"], "_com1.py"],
  [["console", "a.py"], "console.py"],
  [["", "image.png"], "file.png"],
  [["...", "image.png"], "file.png"],
  [[".env", ".env"], ".env"],
  [["paper.pdf", undefined, true], "paper.pdf.txt"],
  [["notes.txt", undefined, true], "notes.txt"],
  [["x".repeat(400), "a.jsonl"], `${"x".repeat(194)}.jsonl`],
  [["字".repeat(200), "a.txt"], `${"字".repeat(65)}.txt`],
  [["😀".repeat(100), "a.png"], `${"😀".repeat(49)}.png`],
]);

test("an item's version changes with its size, as an attachment rewritten in place keeps its time", () => {
  const item = { id: "attachment:m:a", updatedAt: 5, sizeBytes: 10 };
  assert.notEqual(itemVersion(item), itemVersion({ ...item, sizeBytes: 11 }));
  assert.notEqual(itemVersion(item), itemVersion({ ...item, updatedAt: 6 }));
  assert.equal(itemVersion({ ...item, sizeBytes: null }), "attachment:m:a@5.");
});

test("a folder that is gone sends its view back to where it was, or to Folders", () => {
  const page = readSrc("features/library/library-page.tsx");
  const start = page.indexOf("const folderGone =");
  const effect = page.slice(start, page.indexOf("}, [folderGone", start));
  for (const line of [
    'status === "ready" && folderId !== null && currentFolder === null',
    "parentOfOpen && folderById.has(parentOfOpen) ? parentOfOpen : undefined",
    'go({ folder: parent, show: "folders" }, true);',
  ]) {
    assert.ok(effect.includes(line), line);
  }
});

test("names in one download are made unique, ignoring case", () => {
  assert.deepEqual(
    uniqueFileNames(["a.txt", "A.txt", "a.txt", "b", "b", "a (2).txt"]),
    ["a.txt", "A (2).txt", "a (3).txt", "b", "b (2)", "a (2) (2).txt"],
  );
});

table("a file the server types opaquely gets a type from its extension", libraryFileType, [
  [["run.py", ""], "text/plain"],
  [["run.py", "application/octet-stream"], "text/plain"],
  [["photo.JPG", ""], "image/jpeg"],
  [["blob.bin", ""], "application/octet-stream"],
  [["run.py", "text/x-python; charset=utf-8"], "text/x-python"],
]);

table("nothing embedded directly is typed as a scriptable document", embeddedBlobType, [
  [["pdf", "text/html"], "application/pdf"],
  [["pdf", "application/octet-stream"], "application/pdf"],
  [["image", "image/svg+xml"], "application/octet-stream"],
  [["image", "text/html"], "application/octet-stream"],
  [["image", "image/png"], "image/png"],
  [["video", "video/mp4"], "video/mp4"],
  [["audio", "video/mp4"], "application/octet-stream"],
]);

test("audio and video with a file of their own stream; the rest keep a blob", () => {
  for (const id of ["upload:abc", "audio:a1", "video:v1", "sandbox:t-1:out/song.mp3"]) {
    assert.ok(streamsPreview(id, "audio") && streamsPreview(id, "video"), id);
  }
  for (const [id, body] of [
    ["attachment:m:a", "video"], ["upload:abc", "image"], ["upload:abc", "pdf"], ["upload:abc", null],
    ["model:training:/x", "video"], ["upload", "video"], [":upload", "video"],
  ] as const) {
    assert.equal(streamsPreview(id, body), false, `${id} ${body}`);
  }
});

const utf8 = (text: string) => new TextEncoder().encode(text);
const bytes = (...values: number[]) => new Uint8Array(values);
const utf16le = (text: string) =>
  bytes(0xff, 0xfe, ...[...text].flatMap((c) => [c.charCodeAt(0) & 0xff, c.charCodeAt(0) >> 8]));

test("notes decode by their BOM and save back in the same shape", () => {
  const cases: [Uint8Array, string, string, boolean, string, string, string][] = [
    [utf8("héllo\nworld\n"), "héllo\nworld\n", "utf-8", false, "\n", "héllo\n", "héllo\n"],
    [utf8("a\r\nb\r\nc"), "a\nb\nc", "utf-8", false, "\r\n", "a\nb\r\nc", "a\r\nb\r\nc"],
    [bytes(0xef, 0xbb, 0xbf, ...utf8("hi")), "hi", "utf-8", true, "\n", "hi!", "\uFEFFhi!"],
    [utf16le("Write-Host\r\n"), "Write-Host\n", "utf-16le", true, "\r\n", "x\n", "\uFEFFx\r\n"],
    [bytes(0xfe, 0xff, 0x00, 0x41, 0x00, 0x42), "AB", "utf-16be", true, "\n", "AB", "\uFEFFAB"],
  ];
  for (const [input, text, encoding, bom, eol, edit, saved] of cases) {
    const note = decodeNote(input);
    assert.deepEqual(note, { text, format: { encoding, bom, eol }, readOnlyReason: null });
    assert.equal(encodeNote(edit, note.format), saved);
  }
});

test("a note that would not save back as it was opens read-only", () => {
  assert.equal(decodeNote(bytes(0xff, 0xfe, 0x00, 0xd8)).readOnlyReason, "utf16");
  const legacy = decodeNote(bytes(0x63, 0x61, 0x66, 0xe9));
  assert.ok(legacy.readOnlyReason === "notUtf8" && legacy.text.startsWith("caf"));
  const cut = decodeNote(utf8("ab€").subarray(0, 4), true);
  assert.deepEqual([cut.text, cut.readOnlyReason], ["ab", null]);
});

test("every spelling of this machine counts as local, and nothing else does", () => {
  const local = "localhost LOCALHOST studio.localhost 127.0.0.1 127.1.2.3 [::1] ::1 0.0.0.0 [::]";
  const remote = "192.168.1.20 studio.example.com localhost.example.com 128.0.0.1 [::2]";
  for (const host of [...local.split(" "), "[::ffff:127.0.0.1]", "[::ffff:7f00:1]"]) {
    assert.equal(isLoopbackHost(host), true, host);
  }
  for (const host of remote.split(" ")) assert.equal(isLoopbackHost(host), false, host);
});

table("Reveal is named by the server's file manager, else by its platform", revealLabelFor, [
  [["finder", "mac"], "library.reveal.finder"],
  [["explorer", "windows"], "library.reveal.explorer"],
  [["explorer", "linux"], "library.reveal.explorer"],
  [["files", "linux"], "library.reveal.files"],
  [[null, "linux"], null],
  [[undefined, "mac"], "library.reveal.finder"],
  [[undefined, "windows"], "library.reveal.explorer"],
  [[undefined, "linux"], "library.reveal.files"],
]);

// Thumbnail URLs: one a card still loads or shows is never revoked under it.
const revoked = new Set<string>();
const realRevoke = URL.revokeObjectURL.bind(URL);
URL.revokeObjectURL = (url: string) => {
  revoked.add(url);
  realRevoke(url);
};
const blob = () => Promise.resolve(new Blob(["x"]));
async function fill(prefix: string, count: number) {
  for (let i = 0; i < count; i++) {
    const held = acquireObjectUrl(`${prefix}-${i}`, blob);
    await held.url;
    held.release();
  }
}

test("a held URL survives the count cap, and goes once released", async () => {
  clearCachedObjectUrls();
  let finish!: (value: Blob) => void;
  const pending = acquireObjectUrl("pending", () => new Promise((resolve) => (finish = resolve)));
  await fill("item", 310);
  finish(new Blob(["late"]));
  const url = await pending.url;
  assert.ok(!revoked.has(url) && cachedObjectUrlCount() <= 300);
  pending.release();
  await fill("one-more", 1);
  assert.ok(revoked.has(url));
});

test("a hit is the same URL, and stays valid while anyone holds it", async () => {
  clearCachedObjectUrls();
  const first = acquireObjectUrl("same", blob);
  const second = acquireObjectUrl("same", () => Promise.reject(new Error("not refetched")));
  const url = await first.url;
  assert.equal(await second.url, url);
  first.release();
  await fill("filler", 305);
  assert.ok(!revoked.has(url));
  second.release();
});
