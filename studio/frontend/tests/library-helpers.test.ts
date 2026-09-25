// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Library's pure rules: file naming and typing, note encodings, preview streaming, Reveal, the
// thumbnail URL cache, card times, folder paths and the Library settings.

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
import { formatCardTime } from "../src/features/library/format.ts";
import { decodeNote, encodeNote } from "../src/features/library/note-text.ts";
import {
  acquireObjectUrl,
  cachedObjectUrlCount,
  clearCachedObjectUrls,
} from "../src/features/library/object-url-cache.ts";
import { parentFolder } from "../src/features/library/paths.ts";
import { isLoopbackHost, revealLabelFor } from "../src/features/library/reveal-label.ts";
import {
  DEFAULT_LIBRARY_SETTINGS,
  type LibrarySortKey,
  SORT_STATES,
  compareBySort,
  includedBySettings,
  lastActivity,
  migrateLibrarySettings,
  nextSort,
  sortParam,
} from "../src/features/library/settings-store.ts";
import { readSrc } from "./helpers/kit.ts";

/** One test that checks `run(...args)` against `expected` for each [args, expected] row. */
function table<A extends unknown[], O>(name: string, run: (...args: A) => O, rows: [A, O][]) {
  test(name, () => {
    for (const [args, expected] of rows) assert.deepEqual(run(...args), expected, String(args));
  });
}

const fileName = (name: string, file?: string, textOnly?: boolean) =>
  libraryFileName({ name, fileName: file, textOnly });

table("a renamed item saves under a name every OS takes, with its file's extension", fileName, [
  // A rename that dropped the extension gets it back.
  [["Q3 report", "report.pdf"], "Q3 report.pdf"],
  [["notes.v2", "notes.md"], "notes.v2.md"],
  [["script.PY", "script.py"], "script.PY"],
  // Characters Windows refuses, and controls, become "_"; trailing dots and spaces go.
  [['a<b>c:d"e/f\\g|h?i*j', "x.txt"], "a_b_c_d_e_f_g_h_i_j.txt"],
  [["tab\there\u0007", "x.md"], "tab_here_.md"],
  [["draft. . .", "draft.md"], "draft.md"],
  [["../../etc/passwd", "a.txt"], ".._.._etc_passwd.txt"],
  // Reserved device names are prefixed, whatever follows them.
  [["CON", "con.txt"], "_CON.txt"],
  [["nul.backup", "x.txt"], "_nul.backup.txt"],
  [["com1", "a.py"], "_com1.py"],
  [["console", "a.py"], "console.py"],
  // An empty or dot-only name still saves; a dotfile keeps its name.
  [["", "image.png"], "file.png"],
  [["...", "image.png"], "file.png"],
  [[".env", ".env"], ".env"],
  // Text-only chat uploads say so with .txt.
  [["paper.pdf", undefined, true], "paper.pdf.txt"],
  [["notes.txt", undefined, true], "notes.txt"],
  // A very long name is cut to 200 bytes of UTF-8, keeping its extension and whole characters.
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
  // Attachments live inside their message; images and PDFs are small enough to buffer.
  for (const [id, body] of [
    ["attachment:m:a", "video"], ["upload:abc", "image"], ["upload:abc", "pdf"], ["upload:abc", null],
    ["model:training:/x", "video"], ["upload", "video"], [":upload", "video"],
  ] as const) {
    assert.equal(streamsPreview(id, body), false, `${id} ${body}`);
  }
});

const utf8 = (text: string) => new TextEncoder().encode(text);
const bytes = (...values: number[]) => new Uint8Array(values);
// UTF-16LE with its BOM, as Notepad and PowerShell write it.
const utf16le = (text: string) =>
  bytes(0xff, 0xfe, ...[...text].flatMap((c) => [c.charCodeAt(0) & 0xff, c.charCodeAt(0) >> 8]));

test("notes decode by their BOM and save back in the same shape", () => {
  // [bytes, text shown, encoding, bom, eol, an edit, what saving it writes]
  const cases: [Uint8Array, string, string, boolean, string, string, string][] = [
    [utf8("héllo\nworld\n"), "héllo\nworld\n", "utf-8", false, "\n", "héllo\n", "héllo\n"],
    // CRLF shows as LF and goes back as CRLF; a pasted CRLF is not doubled.
    [utf8("a\r\nb\r\nc"), "a\nb\nc", "utf-8", false, "\r\n", "a\nb\r\nc", "a\r\nb\r\nc"],
    [bytes(0xef, 0xbb, 0xbf, ...utf8("hi")), "hi", "utf-8", true, "\n", "hi!", "\uFEFFhi!"],
    // The server writes the leading U+FEFF in the note's own encoding.
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
  // A BOM, then a lone high surrogate; "café" in windows-1252.
  assert.equal(decodeNote(bytes(0xff, 0xfe, 0x00, 0xd8)).readOnlyReason, "utf16");
  const legacy = decodeNote(bytes(0x63, 0x61, 0x66, 0xe9));
  assert.ok(legacy.readOnlyReason === "notUtf8" && legacy.text.startsWith("caf"));
  // A prefix cut inside a character is not a bad encoding.
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
  // WSL reports linux, but reveals in the Windows host's Explorer.
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
/** Acquires and releases `count` fresh entries, pushing older ones toward eviction. */
async function fill(prefix: string, count: number) {
  for (let i = 0; i < count; i++) {
    const held = acquireObjectUrl(`${prefix}-${i}`, blob);
    await held.url;
    held.release();
  }
}

test("a held URL survives the count cap, and goes once released", async () => {
  clearCachedObjectUrls();
  // Held while still loading, as a card waiting on its thumbnail is.
  let finish!: (value: Blob) => void;
  const pending = acquireObjectUrl("pending", () => new Promise((resolve) => (finish = resolve)));
  await fill("item", 310);
  finish(new Blob(["late"]));
  const url = await pending.url;
  assert.ok(!revoked.has(url) && cachedObjectUrlCount() <= 300);
  pending.release();
  // Released and now the oldest, so the next miss past the cap takes it.
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

// A Wednesday afternoon, local time. ICU may put a narrow no-break space before AM.
const now = new Date(2026, 8, 23, 15, 0).getTime();
const at = (month: number, day: number, year = 2026) => new Date(year, month, day, 10, 30).getTime();
const cardTime = (ts: number, locale: Parameters<typeof formatCardTime>[1]) =>
  formatCardTime(ts, locale, now).replace(/\s/g, " ");

table("card times read in the app's language; a time ahead of this clock shows its date", cardTime, [
  [[at(8, 22), "en"], "Yesterday"],
  [[at(8, 22), "de"], "Gestern"],
  [[at(8, 22), "fr"], "Hier"],
  [[at(8, 22), "ja"], "昨日"],
  [[at(8, 20), "en"], "Sunday"],
  [[at(8, 20), "es"], "domingo"],
  [[at(8, 23), "en"], "10:30 AM"],
  [[at(7, 1), "en"], "Aug 1"],
  [[at(7, 1, 2025), "en"], "Aug 1, 2025"],
  [[at(8, 25), "en"], "Sep 25"],
  [[at(0, 2, 2027), "en"], "Jan 2, 2027"],
  [[Number.NaN, "en"], ""],
]);

table("the folder picker starts in the folder above, drive and share roots included", parentFolder, [
  [["/Users/me/Pictures/Unsloth Images"], "/Users/me/Pictures"],
  [["/Unsloth Images"], "/"],
  [["/Users/me/Pictures/"], "/Users/me"],
  // `D:` alone is drive D's current folder, not its root.
  [["D:\\Unsloth Images"], "D:\\"],
  [["D:/Unsloth Images"], "D:/"],
  [["C:\\Users\\me\\Unsloth"], "C:\\Users\\me"],
  [["\\\\server\\share\\Unsloth Images"], "\\\\server\\share\\"],
  [["\\\\server\\share\\a\\b"], "\\\\server\\share\\a"],
  // A root has nothing above it to start from.
  [["/"], undefined],
  [["D:\\"], undefined],
  [["\\\\server\\share"], undefined],
  [["relative"], undefined],
  [[""], undefined],
]);

const hidden = { ...DEFAULT_LIBRARY_SETTINGS, showChatAttachments: false, showFineTunes: false };
const included = (id: string, more?: Partial<typeof hidden>) => includedBySettings(id, { ...hidden, ...more });

table("sources hidden in settings drop out, Library uploads always stay", included, [
  [["upload:abc"], true],
  [["attachment:m:a"], false],
  [["model:training:/runs/x"], false],
  [["sandbox:t:out.csv"], true],
  [["image:1", { showGeneratedMedia: false }], false],
]);

const sortable = [
  { name: "b 10", updatedAt: 2, sizeBytes: 5 },
  { name: "b 9", updatedAt: 3, sizeBytes: null },
  { name: "a", updatedAt: 1, sizeBytes: 50 },
];
const sortedNames = (sort: keyof typeof SORT_STATES) =>
  [...sortable].sort(compareBySort(SORT_STATES[sort])).map((item) => item.name);

table("sort orders by recency, name and size", sortedNames, [
  [["recent"], ["b 9", "b 10", "a"]],
  [["oldest"], ["a", "b 10", "b 9"]],
  [["name"], ["a", "b 9", "b 10"]],
  [["size"], ["a", "b 10", "b 9"]],
]);

table("a column click flips its direction or starts a new column naturally", (key: LibrarySortKey) => nextSort(SORT_STATES.size, key), [
  [["size"], { key: "size", desc: false }],
  [["name"], { key: "name", desc: false }],
  [["modified"], { key: "modified", desc: true }],
]);

table("last activity is the later of modified and opened", lastActivity, [
  [[{ updatedAt: 5, openedAt: null }], 5],
  [[{ updatedAt: 5, openedAt: 9 }], 9],
  [[{ updatedAt: 9, openedAt: 5 }], 9],
]);

test("every column order has a ?sort value that reads back the same", () => {
  for (const key of ["name", "modified", "size"] as const) {
    for (const desc of [true, false]) assert.deepEqual(SORT_STATES[sortParam({ key, desc })], { key, desc });
  }
});

test("the v1 media switch becomes one setting per tab", () => {
  const migrated = migrateLibrarySettings({ mediaTabs: "always", sort: "name" }, 1);
  assert.equal(migrated.mediaTabs, undefined);
  assert.equal(migrated.sort, "name");
  const media = { images: "always", videos: "always", audio: "always" };
  assert.deepEqual(migrated.tabs, { ...DEFAULT_LIBRARY_SETTINGS.tabs, ...media });
});

test("a card's date shows on hover, on keyboard focus and always on touch", () => {
  const date = /"([^"]*)",\s*\)}\s*>\s*\{formatCardTime/.exec(readSrc("features/library/components/library-cards.tsx"))?.[1].split(" ");
  for (const reveal of [
    "group-hover/library-card:opacity-100",
    "group-has-[:focus-visible]/library-card:opacity-100",
    "pointer-coarse:opacity-100",
  ]) {
    assert.ok(date?.includes(reveal), reveal);
  }
});

test("a moved folder's files are read again by the galleries and the Library", () => {
  const move = /async function move\([\s\S]*?\n {2}\}\n/.exec(readSrc("features/settings/tabs/library-tab.tsx"))?.[0] ?? "";
  assert.match(move, /notifyGalleryChanged\(location\.key\)/);
  assert.match(move, /useLibraryStore\.getState\(\)[\s\S]*\.refresh\(\)/);
});
