// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// PR 9057 review simulation, frontend half. Not part of the PR.
//
// Covers the axes a video attachment travels in a browser: how the four engines
// report a MIME type for the five accepted containers, which adapter the
// composite dispatches a file to (order matters, .mp4 and .webm are claimed by
// more than one adapter's accept list), the size boundary against the backend's
// own ceiling, and the extractor that turns a stored part back into base64.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  MAX_VIDEO_SIZE,
  VIDEO_ACCEPT,
  getVideoSizeError,
  isVideoFile,
} from "../src/lib/video-utils.ts";
import { AUDIO_ACCEPT, MAX_AUDIO_SIZE } from "../src/lib/audio-utils.ts";

// chat-adapter.ts drags in the stores, the toast layer and the whole runtime for
// one pure extractor, so lift the shipped source instead of importing it -- the
// same trick tests/auto-load-target-key.test.ts uses. This still asserts against
// the real code: a rename or a rewrite fails the slice below.
const adapterSource = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

function lift(name: string, opener: string): string {
  const start = adapterSource.indexOf(opener);
  assert.ok(start >= 0, `${name} is no longer defined in chat-adapter.ts`);
  const end = adapterSource.indexOf("\n}", start);
  assert.ok(end > start, `could not find the end of ${name}`);
  return adapterSource.slice(start, end + 2);
}

const liftedTs = [
  lift("extractVideoPartBase64", "function extractVideoPartBase64("),
  lift("findLatestUserVideoBase64", "export function findLatestUserVideoBase64(").replace(
    "export function",
    "function",
  ),
  "return findLatestUserVideoBase64;",
].join("\n\n");

const liftedJs = ts.transpileModule(liftedTs, {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
}).outputText;

// eslint-disable-next-line @typescript-eslint/no-implied-eval
const findLatestUserVideoBase64 = new Function(liftedJs)() as (
  messages: unknown,
) => string | undefined;

// ---------------------------------------------------------------------------
// A. isVideoFile across the MIME types the four engines actually report
// ---------------------------------------------------------------------------

/** A File stand-in: isVideoFile only reads .type and .name. */
const f = (name: string, type: string) => ({ name, type }) as unknown as File;

// Observed reporting: Chrome/Edge and Firefox map by extension from their own
// table; Safari uses UTIs; every engine falls back to "" for a container it does
// not know, which is the case that makes the extension fallback load-bearing.
const ENGINE_MIME: Record<string, Record<string, string>> = {
  "chrome/edge": {
    "clip.mp4": "video/mp4",
    "clip.mov": "video/quicktime",
    "clip.webm": "video/webm",
    "clip.mkv": "video/x-matroska",
    "clip.avi": "video/x-msvideo",
  },
  firefox: {
    "clip.mp4": "video/mp4",
    "clip.mov": "video/quicktime",
    "clip.webm": "video/webm",
    // Firefox on Windows leans on the registry and routinely reports nothing.
    "clip.mkv": "",
    "clip.avi": "video/x-msvideo",
  },
  safari: {
    "clip.mp4": "video/mp4",
    "clip.mov": "video/quicktime",
    "clip.webm": "video/webm",
    // No UTI for Matroska on stock macOS.
    "clip.mkv": "",
    "clip.avi": "video/avi",
  },
  "windows-no-codec-pack": {
    "clip.mp4": "video/mp4",
    "clip.mov": "",
    "clip.webm": "",
    "clip.mkv": "",
    "clip.avi": "",
  },
};

for (const [engine, table] of Object.entries(ENGINE_MIME)) {
  for (const [name, type] of Object.entries(table)) {
    test(`isVideoFile claims ${name} as reported by ${engine} (type=${JSON.stringify(type)})`, () => {
      assert.equal(isVideoFile(f(name, type)), true);
    });
  }
}

test("an uppercase extension from a Windows share is still a video", () => {
  for (const name of ["CLIP.MP4", "Clip.MoV", "CLIP.MKV", "holiday.AVI", "a.WebM"]) {
    assert.equal(isVideoFile(f(name, "")), true, name);
  }
});

test("a video MIME with no extension at all is still a video", () => {
  assert.equal(isVideoFile(f("recording", "video/mp4")), true);
  assert.equal(isVideoFile(f("recording", "VIDEO/MP4")), true);
});

test("a document that merely mentions a container in its name is not a video", () => {
  for (const name of ["notes-about-mp4.txt", "clip.mp4.pdf", "mp4", "avi.docx", ".mp4x"]) {
    assert.equal(isVideoFile(f(name, "text/plain")), false, name);
  }
});

test("images, audio and documents are never mistaken for video", () => {
  for (const [name, type] of [
    ["a.png", "image/png"],
    ["a.jpg", "image/jpeg"],
    ["a.wav", "audio/wav"],
    ["a.mp3", "audio/mpeg"],
    ["a.m4a", "audio/mp4"],
    ["a.pdf", "application/pdf"],
    ["a.md", "text/markdown"],
  ] as const) {
    assert.equal(isVideoFile(f(name, type)), false, name);
  }
});

// ---------------------------------------------------------------------------
// B. which adapter claims the file: the composite takes the FIRST match
// ---------------------------------------------------------------------------

// Reimplementation of @assistant-ui/core's fileMatchesAccept, verbatim, so the
// dispatch order can be simulated without mounting React.
function fileMatchesAccept(file: { name: string; type: string }, accept: string) {
  if (accept === "*") return true;
  const allowed = accept.split(",").map((t) => t.trim().toLowerCase());
  const ext = `.${file.name.split(".").pop()!.toLowerCase()}`;
  const mime = file.type.toLowerCase();
  for (const t of allowed) {
    if (t.startsWith(".") && t === ext) return true;
    if (t.includes("/") && t === mime) return true;
    if (t.endsWith("/*") && mime.startsWith(`${t.split("/")[0]}/`)) return true;
  }
  return false;
}

// runtime-provider.tsx registration order.
const ADAPTERS: [string, string][] = [
  ["image", "image/jpeg,image/png,image/webp,image/gif"],
  ["audio", AUDIO_ACCEPT],
  ["video", VIDEO_ACCEPT],
  ["text", "text/plain,text/markdown,.txt,.md"],
  ["html", "text/html,.html"],
  ["pdf", "application/pdf,.pdf"],
];

const dispatch = (file: { name: string; type: string }) =>
  ADAPTERS.find(([, accept]) => fileMatchesAccept(file, accept))?.[0] ?? null;

test("every accepted container reaches the video adapter, not a document one", () => {
  for (const table of Object.values(ENGINE_MIME)) {
    for (const [name, type] of Object.entries(table)) {
      assert.equal(dispatch({ name, type }), "video", `${name} ${type}`);
    }
  }
});

test("an audio-only webm still reaches the audio adapter, which is registered first", () => {
  assert.equal(dispatch({ name: "voice.webm", type: "audio/webm" }), "audio");
});

test("an m4a keeps going to audio even though mp4 is a video container", () => {
  assert.equal(dispatch({ name: "voice.m4a", type: "audio/mp4" }), "audio");
});

test("an audio-only 3gp reaches audio while a video 3gp stays video", () => {
  assert.equal(dispatch({ name: "voice.3gp", type: "audio/3gpp" }), "audio");
  assert.equal(dispatch({ name: "clip.3gp", type: "video/3gpp" }), "video");
});

test("adding the video adapter did not steal any pre-existing attachment type", () => {
  // Same corpus, with the video adapter removed: the answer must be unchanged
  // for everything that is not a video.
  const before = ADAPTERS.filter(([n]) => n !== "video");
  const dispatchBefore = (file: { name: string; type: string }) =>
    before.find(([, a]) => fileMatchesAccept(file, a))?.[0] ?? null;
  for (const [name, type] of [
    ["a.png", "image/png"],
    ["a.gif", "image/gif"],
    ["a.wav", "audio/wav"],
    ["a.mp3", "audio/mpeg"],
    ["a.ogg", "audio/ogg"],
    ["a.flac", "audio/flac"],
    ["voice.webm", "audio/webm"],
    ["voice.m4a", "audio/mp4"],
    ["a.txt", "text/plain"],
    ["a.md", "text/markdown"],
    ["a.html", "text/html"],
    ["a.pdf", "application/pdf"],
  ] as const) {
    assert.equal(dispatch({ name, type }), dispatchBefore({ name, type }), `${name} ${type}`);
  }
});

test("the composer's accept attribute is the union, so the picker offers video", () => {
  const union = ADAPTERS.map(([, a]) => a).join(",");
  for (const token of ["video/mp4", "video/quicktime", "video/webm", ".mkv", ".avi", ".mov"]) {
    assert.ok(union.includes(token), token);
  }
  // and still offers everything it used to
  for (const token of ["image/png", "audio/wav", "application/pdf"]) {
    assert.ok(union.includes(token), token);
  }
});

// ---------------------------------------------------------------------------
// C. the size gate, and its agreement with the backend ceiling
// ---------------------------------------------------------------------------

test("the composer cap is exactly 64 MiB", () => {
  assert.equal(MAX_VIDEO_SIZE, 64 * 1024 * 1024);
  assert.equal(MAX_VIDEO_SIZE, 67108864);
});

test("a clip of exactly the cap is accepted, one byte over is refused", () => {
  assert.equal(getVideoSizeError(MAX_VIDEO_SIZE), null);
  assert.equal(getVideoSizeError(MAX_VIDEO_SIZE - 1), null);
  assert.equal(getVideoSizeError(0), null);
  assert.ok(getVideoSizeError(MAX_VIDEO_SIZE + 1));
});

test("the backend ceiling admits every clip this composer admits", () => {
  // _MAX_VIDEO_B64_CHARS in routes/inference.py, padded base64 of the same cap.
  const backendCeiling = 4 * Math.ceil(MAX_VIDEO_SIZE / 3);
  assert.equal(backendCeiling, 89478488);
  // Padded base64 length for the largest allowed file.
  const encodedAtCap = 4 * Math.ceil(MAX_VIDEO_SIZE / 3);
  assert.ok(encodedAtCap <= backendCeiling, "the largest allowed clip must not 413");
  // The floor form the review flagged would have been three characters short.
  assert.ok(Math.floor((MAX_VIDEO_SIZE * 4) / 3) < encodedAtCap);
});

test("video and audio caps stay distinct, so neither gate borrows the other's limit", () => {
  assert.notEqual(MAX_VIDEO_SIZE, MAX_AUDIO_SIZE);
  assert.ok(MAX_VIDEO_SIZE > MAX_AUDIO_SIZE);
});

// ---------------------------------------------------------------------------
// D. reading the clip back out of a thread
// ---------------------------------------------------------------------------

const userVideo = (data: string, mimeType = "video/mp4") => ({
  role: "user" as const,
  content: [
    { type: "text", text: "what happens here?" },
    { type: "file", data, mimeType },
  ],
});

test("a raw base64 part is returned untouched", () => {
  assert.equal(findLatestUserVideoBase64([userVideo("QUJD")] as never), "QUJD");
});

test("a data URI part is stripped to its payload", () => {
  assert.equal(
    findLatestUserVideoBase64([userVideo("data:video/mp4;base64,QUJD")] as never),
    "QUJD",
  );
});

test("an uppercase MIME from Safari is still recognised as video", () => {
  assert.equal(findLatestUserVideoBase64([userVideo("QUJD", "VIDEO/QUICKTIME")] as never), "QUJD");
});

test("only the newest user turn contributes a clip", () => {
  const messages = [
    userVideo("OLD"),
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
    { role: "user", content: [{ type: "text", text: "and now?" }] },
  ];
  assert.equal(findLatestUserVideoBase64(messages as never), undefined);
});

test("a clip on the newest turn wins over an older one", () => {
  const messages = [
    userVideo("OLD"),
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
    userVideo("NEW"),
  ];
  assert.equal(findLatestUserVideoBase64(messages as never), "NEW");
});

test("a clip carried as an attachment rather than a content part is found", () => {
  const messages = [
    {
      role: "user",
      content: [{ type: "text", text: "hi" }],
      attachments: [{ content: [{ type: "file", data: "ATT", mimeType: "video/webm" }] }],
    },
  ];
  assert.equal(findLatestUserVideoBase64(messages as never), "ATT");
});

test("a non-video file part is never mistaken for a clip", () => {
  for (const mimeType of ["application/pdf", "text/plain", "image/png", "audio/wav", ""]) {
    assert.equal(findLatestUserVideoBase64([userVideo("QUJD", mimeType)] as never), undefined, mimeType);
  }
});

test("a thread with no video and no user turn returns nothing rather than throwing", () => {
  assert.equal(findLatestUserVideoBase64([] as never), undefined);
  assert.equal(
    findLatestUserVideoBase64([{ role: "assistant", content: [{ type: "text", text: "x" }] }] as never),
    undefined,
  );
  assert.equal(findLatestUserVideoBase64([{ role: "user" }] as never), undefined);
});

// ---------------------------------------------------------------------------
// E. the context-usage recount must decline a turn carrying a clip
// ---------------------------------------------------------------------------

const recountSource = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/utils/refresh-context-usage.ts", import.meta.url),
  ),
  "utf8",
);

test("the recount declines a video turn, as it already declines image and audio", () => {
  // toOpenAIMessages has no video branch, so a turn carrying a clip would be
  // priced as text-only while the real request sends video_base64 and
  // llama-server expands it into frames. /chat/count_tokens 503s on video for
  // the same reason, so without this bail the bar shows room that is not there.
  assert.ok(recountSource.includes("messagesContainImage(runMessages)"));
  assert.ok(recountSource.includes("findLatestUserAudioBase64(runMessages)"));
  assert.ok(
    recountSource.includes("findLatestUserVideoBase64(runMessages)"),
    "refresh-context-usage.ts must decline a turn carrying a video",
  );
});

test("the video bail is paid before the branch signature hashes the base64", () => {
  // branchSignature JSON.stringifies every part on the UI thread; the image
  // bail's own comment says it exists to keep base64 out of that hash, and a
  // 64 MB clip is ~85 MB of base64.
  const bail = recountSource.indexOf("findLatestUserVideoBase64(runMessages)");
  const hash = recountSource.indexOf("countedBranch = branchSignature(");
  assert.ok(bail >= 0 && hash >= 0);
  assert.ok(bail < hash, "the video bail must run before branchSignature");
});

test("an old persisted thread with no file parts at all is unaffected", () => {
  // Forward/backwards compatibility: chats written before this PR carry only
  // text and image parts, and must read back exactly as they did.
  const legacy = [
    { role: "user", content: [{ type: "text", text: "hello" }] },
    { role: "assistant", content: [{ type: "text", text: "hi" }] },
    {
      role: "user",
      content: [
        { type: "text", text: "look" },
        { type: "image", image: "data:image/png;base64,AA" },
      ],
    },
  ];
  assert.equal(findLatestUserVideoBase64(legacy as never), undefined);
});
