// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  OPEN_DOCUMENT_ATTACHMENT_ACCEPT,
  OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS,
} from "../src/features/chat/open-document-accept.ts";
import {
  TEXT_ATTACHMENT_ACCEPT,
  TEXT_ATTACHMENT_EXTENSIONS,
} from "../src/features/chat/text-attachment-accept.ts";
import {
  dequeueNativeAttachments,
  enqueueNativeAttachments,
} from "../src/features/native-intents/attachment-queue.ts";
import {
  CHAT_AUDIO_DROP_ACCEPT,
  CHAT_IMAGE_DROP_ACCEPT,
  CHAT_VIDEO_DROP_ACCEPT,
  SUPPORTED_DROP_HINT,
  classifyDropPaths,
  isComposerAttachmentName,
} from "../src/features/native-intents/drop-paths.ts";
import type { NativeIntent } from "../src/features/native-intents/types.ts";
import { RAG_UPLOAD_ACCEPT } from "../src/features/rag/types/rag.ts";
import { MAX_REFERENCE_BYTES } from "../src/features/video/reference-budget.ts";
import { AUDIO_ACCEPT } from "../src/lib/audio-utils.ts";
import { VIDEO_ACCEPT } from "../src/lib/video-utils.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useNativeIntentStore } = await import(
  "../src/features/native-intents/store.ts"
);

const BACKEND_UPLOAD_EXTS_RE = /UPLOAD_EXTS\s*=\s*\{([^}]+)\}/s;
const RUST_ATTACHMENT_EXTS_RE = /ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_OPEN_DOCUMENT_ATTACHMENT_EXTS_RE =
  /OPEN_DOCUMENT_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_IMAGE_ATTACHMENT_EXTS_RE =
  /IMAGE_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_AUDIO_ATTACHMENT_EXTS_RE =
  /AUDIO_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_AUDIO_MIME_RE = /Some\("(audio\/[^"]+)"\)/g;
const RUST_VIDEO_ATTACHMENT_EXTS_RE =
  /VIDEO_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_TEXT_ATTACHMENT_EXTS_RE =
  /TEXT_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_VIDEO_MIME_RE = /Some\("(video\/[^"]+)"\)/g;
const DOTTED_EXTENSION_RE = /"(\.[^"]+)"/g;
const RUST_EXTENSION_RE = /"([^"]+)"/g;
const RUST_MIME_ARM_RE = /Some\("(image\/[^"]+)"\)/g;
const MIME_MATCH_BODY_RE =
  /fn attachment_mime_type[\s\S]*?match ext\.as_str\(\) \{([\s\S]*?)\n {4}\}/;
const MIME_ARM_EXTENSION_RE =
  /^\s*((?:"[^"]+"\s*\|?\s*)+)=>\s*Some\("image\//gm;
const COMPOSER_IMAGE_ACCEPT_RE = /const IMAGE_ACCEPT\s*=\s*"([^"]+)"/;
const VISION_ADAPTER_ACCEPT_RE =
  /class VisionImageAdapter[^{]*\{\s*accept\s*=\s*"([^"]+)"/s;
const OPEN_DOCUMENT_EXTENSION_RE = /\.ods/;
const OPEN_DOCUMENT_ADAPTER_ACCEPT_RE =
  /class OpenDocumentAttachmentAdapter[^{]*\{[\s\S]*?accept = OPEN_DOCUMENT_ATTACHMENT_ACCEPT;/;
const OPEN_DOCUMENT_DROP_TO_COMPOSER_RE =
  /const composerAttachments = \[[\s\S]*?\.\.\.registered\.composerDocuments,[\s\S]*?\.\.\.registered\.images,[\s\S]*?await attachOptions\.onAttachOpenDocuments\?\.\([\s\S]*?composerAttachments/;
const OPEN_DOCUMENT_REGISTRATION_CLASS_RE =
  /const composerDocumentPaths = docPaths\.filter\(isComposerAttachmentName\);[\s\S]*?const ragDocumentPaths = docPaths\.filter[\s\S]*?registerEach\(ragDocumentPaths\),[\s\S]*?registerEach\(composerDocumentPaths\),[\s\S]*?composerDocuments: composerDocuments\.intents/;
const OPEN_DOCUMENT_IMAGE_REGISTRATION_RE =
  /const needsComposerDocuments = composerDocumentPaths\.length > 0;[\s\S]*?const needsComposerAttachments =\s*needsImages \|\| needsComposerDocuments;[\s\S]*?if \(needsComposerAttachments\) store\.beginImageDropRegistration\(\);[\s\S]*?finally \{[\s\S]*?if \(needsComposerAttachments\) store\.endImageDropRegistration\(\)/;
const OPEN_DOCUMENT_BACKEND_GATE_RE =
  /function canAttachDocumentPaths[\s\S]*?isComposerAttachmentName\(path\)[\s\S]*?\? canAttachOpenDocuments\(options\)[\s\S]*?: canAttachDocs\(options\)[\s\S]*?const needsRagDocuments = documentPaths\.some[\s\S]*?if \(needsRagDocuments && !canAttachDocs\(currentOptions\)\)/;
const OPEN_DOCUMENT_CHAT_QUEUE_RE =
  /const handleNativeOpenDocumentDrop = useCallback\([\s\S]*?addOpenDocumentAttachments\(artifactViewKey, intents\)[\s\S]*?onAttachOpenDocuments: handleNativeOpenDocumentDrop/;
const OPEN_DOCUMENT_DRAIN_RE =
  /takeOpenDocumentAttachments\(targetKey\)[\s\S]*?nativeAttachmentIntentToFile\(intent\)[\s\S]*?await aui\.composer\(\)\.addAttachment\(file\)/;

function attachmentIntent(id: string): NativeIntent {
  return {
    id,
    kind: "attachment",
    sourceKind: "drop",
    displayLabel: `${id}.txt`,
    path: {
      token: `token-${id}`,
      kind: "attachment",
      displayLabel: `${id}.txt`,
      allowedOperations: ["attach", "reveal"],
      expiresAtMs: Date.now() + 60_000,
    },
  };
}

test("a lone gguf is a model drop", () => {
  assert.deepEqual(classifyDropPaths(["C:/models/qwen.gguf"]), {
    kind: "model",
    path: "C:/models/qwen.gguf",
  });
  // Extension casing comes from the filesystem, not from us.
  assert.equal(classifyDropPaths(["/models/Qwen.GGUF"]).kind, "model");
});

test("documents are an attachment drop, one or many", () => {
  const dropped = classifyDropPaths([
    "/docs/a.pdf",
    "/docs/b.MD",
    "/docs/c.docx",
  ]);
  assert.equal(dropped.kind, "docs");
  assert.equal(dropped.kind === "docs" ? dropped.paths.length : 0, 3);
});

test("OpenDocument picker types are accepted by native drops", () => {
  assert.match(OPEN_DOCUMENT_ATTACHMENT_ACCEPT, OPEN_DOCUMENT_EXTENSION_RE);
  for (const extension of OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS.split(",")) {
    const path = `/docs/file${extension.toUpperCase()}`;
    assert.equal(classifyDropPaths([path]).kind, "docs", path);
    assert.ok(SUPPORTED_DROP_HINT.includes(extension));
  }

  const providerSource = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.match(providerSource, OPEN_DOCUMENT_ADAPTER_ACCEPT_RE);

  const nativeDropSource = readFileSync(
    new URL(
      "../src/features/native-intents/use-native-drop.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(nativeDropSource, OPEN_DOCUMENT_DROP_TO_COMPOSER_RE);
  assert.match(nativeDropSource, OPEN_DOCUMENT_REGISTRATION_CLASS_RE);
  assert.match(nativeDropSource, OPEN_DOCUMENT_IMAGE_REGISTRATION_RE);
  assert.match(nativeDropSource, OPEN_DOCUMENT_BACKEND_GATE_RE);

  const chatPageSource = readFileSync(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(chatPageSource, OPEN_DOCUMENT_CHAT_QUEUE_RE);

  const threadSource = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(threadSource, OPEN_DOCUMENT_DRAIN_RE);

  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_OPEN_DOCUMENT_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();
  const frontend = OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS.split(",").sort();
  assert.deepEqual(rust, frontend);
});

test("images route to chat vision attachments, one or many", () => {
  const dropped = classifyDropPaths([
    "/photos/cat.PNG",
    "/photos/dog.jpeg",
    "/photos/icon.webp",
  ]);
  assert.equal(dropped.kind, "images");
  assert.equal(dropped.kind === "images" ? dropped.paths.length : 0, 3);
});

test("documents and images can be dropped together", () => {
  const dropped = classifyDropPaths(["/docs/a.pdf", "/photos/cat.png"]);
  assert.equal(dropped.kind, "attach");
  if (dropped.kind === "attach") {
    assert.deepEqual(dropped.docs, ["/docs/a.pdf"]);
    assert.deepEqual(dropped.images, ["/photos/cat.png"]);
  }
});

test("a mixed or unsupported drop is rejected", () => {
  // The regression in #7661: these used to be reported as "GGUF models only".
  assert.equal(
    classifyDropPaths(["/docs/a.pdf", "/models/q.gguf"]).kind,
    "unsupported",
  );
  assert.equal(
    classifyDropPaths(["/models/a.gguf", "/models/b.gguf"]).kind,
    "unsupported",
  );
  assert.equal(
    classifyDropPaths(["/docs/a.pdf", "/docs/b.zip"]).kind,
    "unsupported",
  );
  assert.equal(classifyDropPaths(["/docs/notes.zip"]).kind, "unsupported");
});

test("an empty payload is not a drop target", () => {
  assert.equal(classifyDropPaths([]).kind, "none");
});

test("the rejection hint mentions images without widening RAG upload accept", () => {
  assert.match(SUPPORTED_DROP_HINT, /\.png/);
  assert.doesNotMatch(RAG_UPLOAD_ACCEPT, /\.png/);
});

test("attachment batches stay bound to the chat that received the drop", () => {
  const queued = enqueueNativeAttachments({}, "single:thread-a", [
    attachmentIntent("doc"),
  ]);

  const [wrongThread, afterWrongThread] = dequeueNativeAttachments(
    queued,
    "single:thread-b",
  );
  assert.deepEqual(wrongThread, []);
  assert.equal(afterWrongThread, queued);

  const [rightThread, remaining] = dequeueNativeAttachments(
    afterWrongThread,
    "single:thread-a",
  );
  assert.deepEqual(
    rightThread.map((intent) => intent.id),
    ["doc"],
  );
  assert.deepEqual(remaining, {});
});

// Registration crosses into Rust before the queue exists, and a send in that gap
// would go out without the image.
test("registering image drops hold the gate before the queue can", () => {
  const store = useNativeIntentStore.getState();
  assert.equal(store.registeringImageDrops, 0);

  store.beginImageDropRegistration();
  assert.equal(useNativeIntentStore.getState().registeringImageDrops, 1);

  // Two drops can be in flight at once; the first to finish can't open the gate.
  store.beginImageDropRegistration();
  store.endImageDropRegistration();
  assert.equal(useNativeIntentStore.getState().registeringImageDrops, 1);

  store.endImageDropRegistration();
  assert.equal(useNativeIntentStore.getState().registeringImageDrops, 0);

  // A stray end can't drive it negative and wedge the gate shut.
  store.endImageDropRegistration();
  assert.equal(useNativeIntentStore.getState().registeringImageDrops, 0);
});

test("frontend, backend, and Rust accept the same document extensions", () => {
  const frontend = RAG_UPLOAD_ACCEPT.split(",").sort();
  const backendSource = readFileSync(
    new URL("../../backend/core/rag/config.py", import.meta.url),
    "utf8",
  );
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const backend = [
    ...(backendSource
      .match(BACKEND_UPLOAD_EXTS_RE)?.[1]
      .matchAll(DOTTED_EXTENSION_RE) ?? []),
  ]
    .map((match) => match[1])
    .sort();
  const rust = [
    ...(rustSource
      .match(RUST_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual(backend, frontend);
  assert.deepEqual(rust, frontend);
});

test("frontend and Rust accept the same chat image extensions", () => {
  const frontend = CHAT_IMAGE_DROP_ACCEPT.split(",")
    .map((ext) => ext.trim().toLowerCase())
    .sort();
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_IMAGE_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual(rust, frontend);
});

// One more seam of #7963's shape: Rust stamps the File's MIME type and the
// composer routes by MIME, so a type VisionImageAdapter does not claim lands on
// the wrong adapter or nowhere.
test("every MIME type Rust stamps is one the vision adapter claims", () => {
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const stamped = [
    ...new Set(
      [...rustSource.matchAll(RUST_MIME_ARM_RE)].map((match) => match[1]),
    ),
  ].sort();

  const providerSource = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const accepted = providerSource
    .match(VISION_ADAPTER_ACCEPT_RE)?.[1]
    .split(",")
    .map((type) => type.trim())
    .sort();

  assert.ok(
    stamped.length > 0,
    "no image MIME arms found in native_intents.rs",
  );
  assert.deepEqual(stamped, accepted);
});

// The join between the two tests above: without it an extension can reach both
// allow-lists with no MIME arm, and the reader refuses it after the drop.
test("every accepted image extension has a Rust MIME arm", () => {
  const policySource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const accepted = [
    ...(policySource
      .match(RUST_IMAGE_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => match[1])
    .sort();

  const intentsSource = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const body = intentsSource.match(MIME_MATCH_BODY_RE)?.[1];
  assert.ok(body, "attachment_mime_type match block not found");
  const mapped = [...body.matchAll(MIME_ARM_EXTENSION_RE)]
    .flatMap((match) =>
      [...match[1].matchAll(/"([^"]+)"/g)].map((ext) => ext[1]),
    )
    .sort();

  assert.ok(accepted.length > 0, "no IMAGE_ATTACHMENT_EXTS found");
  assert.deepEqual(mapped, accepted);
});

// The one constant drop-paths.ts names in its own "keep in sync" comment.
test("the drop image list matches the composer's file picker", () => {
  const composerSource = readFileSync(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );
  const picker = composerSource
    .match(COMPOSER_IMAGE_ACCEPT_RE)?.[1]
    .split(",")
    .map((type) => type.trim().replace("image/", ""))
    .sort();
  const dropped = CHAT_IMAGE_DROP_ACCEPT.split(",")
    .map((ext) => ext.trim().toLowerCase().replace(".", ""))
    .map((ext) => (ext === "jpg" ? "jpeg" : ext))
    .sort();

  assert.ok(picker, "IMAGE_ACCEPT not found in shared-composer.tsx");
  assert.deepEqual([...new Set(dropped)], picker);
});

// A remount means the instance that queued the batch cannot hand it over.
test("a remounted composer claims image and OpenDocument batches", () => {
  const store = useNativeIntentStore.getState();
  const intent = {
    id: "i1",
    kind: "attachment",
    path: {
      token: "t1",
      kind: "attachment",
      displayLabel: "a.png",
      allowedOperations: ["attach"],
      expiresAtMs: Date.now() + 60_000,
    },
  } as unknown as NativeIntent;
  const openDocument = attachmentIntent("remounted-sheet");

  store.addImageAttachments("single:new", [intent]);
  store.addOpenDocumentAttachments("single:new", [openDocument]);
  store.noteImageDropOwner("single:new", "composer-1");

  // A different composer must not take it.
  store.claimImageAttachments("composer-2", "single:other");
  assert.equal(
    useNativeIntentStore.getState().pendingImageAttachments["single:new"]
      ?.length,
    1,
  );

  store.claimImageAttachments("composer-1", "single:thread-7");
  const after = useNativeIntentStore.getState();
  assert.equal(after.pendingImageAttachments["single:new"], undefined);
  assert.deepEqual(after.pendingImageAttachments["single:thread-7"], [intent]);
  assert.equal(after.pendingOpenDocumentAttachments["single:new"], undefined);
  assert.deepEqual(after.pendingOpenDocumentAttachments["single:thread-7"], [
    openDocument,
  ]);
  assert.deepEqual(after.imageDropOwners, {});
});

// The predecessor's requeue can land after the replacement has claimed once.
test("ownership recorded after a claim is still picked up", () => {
  const store = useNativeIntentStore.getState();
  const intent = {
    id: "i2",
    kind: "attachment",
    path: {
      token: "t2",
      kind: "attachment",
      displayLabel: "late.png",
      allowedOperations: ["attach"],
      expiresAtMs: Date.now() + 60_000,
    },
  } as unknown as NativeIntent;

  // The replacement composer claims first, finding nothing.
  store.claimImageAttachments("composer-3", "single:thread-9");
  // Only then does the outgoing drain put its batch back and tag it.
  store.addImageAttachments("single:new", [intent]);
  store.noteImageDropOwner("single:new", "composer-3");

  const owners = useNativeIntentStore.getState().imageDropOwners;
  assert.equal(
    owners["single:new"],
    "composer-3",
    "the note survives for a later claim",
  );

  store.claimImageAttachments("composer-3", "single:thread-9");
  const after = useNativeIntentStore.getState();
  assert.equal(after.pendingImageAttachments["single:new"], undefined);
  assert.deepEqual(after.pendingImageAttachments["single:thread-9"], [intent]);
});

test("image failures cannot consume queued OpenDocuments", () => {
  const store = useNativeIntentStore.getState();
  const image = attachmentIntent("mixed-image");
  const openDocument = attachmentIntent("mixed-sheet");

  store.addImageAttachments("single:mixed", [image]);
  store.addOpenDocumentAttachments("single:mixed", [openDocument]);

  assert.deepEqual(store.takeImageAttachments("single:mixed"), [image]);
  assert.deepEqual(store.takeOpenDocumentAttachments("single:mixed"), [
    openDocument,
  ]);
});

test("document, image and audio drop extensions stay disjoint", () => {
  // classifyDropPaths sums the three filters; an overlap double-counts and
  // turns a good drop into "unsupported".
  const exts = [
    RAG_UPLOAD_ACCEPT,
    CHAT_IMAGE_DROP_ACCEPT,
    CHAT_AUDIO_DROP_ACCEPT,
  ]
    .flatMap((accept) => accept.split(","))
    .map((ext) => ext.trim().toLowerCase());
  assert.deepEqual(
    exts.filter((ext, index) => exts.indexOf(ext) !== index),
    [],
  );
});

// A dropped clip has to reach the same adapter an upload does.
test("a single audio file routes to chat audio attachments", () => {
  const dropped = classifyDropPaths(["/clips/take.WAV"]);
  assert.equal(dropped.kind, "audio");
  assert.deepEqual(dropped.kind === "audio" ? dropped.paths : [], [
    "/clips/take.WAV",
  ]);
});

// One clip per message, so a larger batch is turned away before it is read.
test("multi-audio drops are rejected before they are routed", () => {
  const dropped = classifyDropPaths([
    "/clips/take.WAV",
    "/clips/note.mp3",
    "/clips/voice.flac",
  ]);
  assert.equal(dropped.kind, "unsupported");
});

test("a second clip alongside other attachments is rejected too", () => {
  const dropped = classifyDropPaths([
    "/docs/a.pdf",
    "/clips/note.mp3",
    "/clips/voice.flac",
  ]);
  assert.equal(dropped.kind, "unsupported");
});

test("documents, images and audio can be dropped together", () => {
  const dropped = classifyDropPaths([
    "/docs/a.pdf",
    "/photos/cat.png",
    "/clips/note.mp3",
  ]);
  assert.equal(dropped.kind, "attach");
  if (dropped.kind === "attach") {
    assert.deepEqual(dropped.docs, ["/docs/a.pdf"]);
    assert.deepEqual(dropped.images, ["/photos/cat.png"]);
    assert.deepEqual(dropped.audio, ["/clips/note.mp3"]);
  }
});

test("a video routes to chat video attachments", () => {
  const dropped = classifyDropPaths(["/clips/demo.mp4"]);
  assert.equal(dropped.kind, "video");
  if (dropped.kind === "video") {
    assert.deepEqual(dropped.paths, ["/clips/demo.mp4"]);
  }
});

// llama-server expands one clip into a run of frames, so a batch would spend
// the whole context before the model saw any of it.
test("more than one video is not a drop target", () => {
  assert.equal(
    classifyDropPaths(["/clips/a.mp4", "/clips/b.mov"]).kind,
    "unsupported",
  );
});

test("video rides along in a mixed attachment drop", () => {
  const dropped = classifyDropPaths([
    "/docs/a.pdf",
    "/photos/cat.png",
    "/clips/demo.webm",
  ]);
  assert.equal(dropped.kind, "attach");
  if (dropped.kind === "attach") {
    assert.deepEqual(dropped.docs, ["/docs/a.pdf"]);
    assert.deepEqual(dropped.images, ["/photos/cat.png"]);
    assert.deepEqual(dropped.video, ["/clips/demo.webm"]);
    assert.deepEqual(dropped.audio, []);
  }
});

test("frontend and Rust accept the same chat video extensions", () => {
  const frontend = CHAT_VIDEO_DROP_ACCEPT.split(",")
    .map((ext) => ext.trim().toLowerCase())
    .sort();
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_VIDEO_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual(rust, frontend);
});

// Same seam as the vision and audio checks: a video MIME the adapter does not
// claim would be read off disk and then refused by the composer.
test("every video MIME Rust stamps is one the video adapter claims", () => {
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const claimed = new Set(
    VIDEO_ACCEPT.split(",").map((token) => token.trim().toLowerCase()),
  );
  const stamped = [
    ...(rustSource.match(MIME_MATCH_BODY_RE)?.[1] ?? "").matchAll(
      RUST_VIDEO_MIME_RE,
    ),
  ].map((match) => match[1]);

  assert.ok(stamped.length > 0, "Rust stamps no video MIME types");
  for (const mime of stamped) {
    assert.ok(claimed.has(mime), `the video adapter does not claim ${mime}`);
  }
});

test("the rejection hint names video too", () => {
  assert.ok(SUPPORTED_DROP_HINT.includes(CHAT_VIDEO_DROP_ACCEPT));
});

test("frontend and Rust accept the same chat audio extensions", () => {
  const frontend = CHAT_AUDIO_DROP_ACCEPT.split(",")
    .map((ext) => ext.trim().toLowerCase())
    .sort();
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_AUDIO_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual(rust, frontend);
});

// Same seam as the vision check: an audio MIME the adapter does not claim
// lands nowhere.
test("every audio MIME Rust stamps is one the audio adapter claims", () => {
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const claimed = new Set(
    AUDIO_ACCEPT.split(",").map((token) => token.trim().toLowerCase()),
  );
  const stamped = [
    ...(rustSource.match(MIME_MATCH_BODY_RE)?.[1] ?? "").matchAll(
      RUST_AUDIO_MIME_RE,
    ),
  ].map((match) => match[1]);

  assert.ok(stamped.length > 0, "Rust stamps no audio MIME types");
  for (const mime of stamped) {
    assert.ok(claimed.has(mime), `the audio adapter does not claim ${mime}`);
  }
});

// The native reader's video cap bounds the FILE; the reference picker's cap
// bounds the data URL it builds from it. Set to the base64 figure, Rust reads
// and encodes 96 MiB (128 MiB over the bridge) for a clip the picker rejects.
test("the native video cap is the raw limit the reference picker enforces", () => {
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const rustCap = Number(
    rustSource
      .match(/const MAX_NATIVE_VIDEO_BYTES: u64 = ([0-9_]+);/)?.[1]
      .replaceAll("_", ""),
  );

  assert.ok(Number.isFinite(rustCap), "MAX_NATIVE_VIDEO_BYTES not found");
  assert.equal(rustCap, MAX_REFERENCE_BYTES.video);
  // The thing that made this wrong: the two differ by a third.
  assert.ok(rustCap < 96 * 1024 * 1024);
});

test("a dropped source file attaches instead of being refused", () => {
  for (const path of ["/src/Program.cs", "/src/index.php", "/app/main.js"]) {
    assert.equal(classifyDropPaths([path]).kind, "docs", path);
    assert.equal(isComposerAttachmentName(path), true, path);
  }
  const mixed = classifyDropPaths(["/src/a.cs", "/notes/report.pdf"]);
  assert.equal(mixed.kind, "docs");
});

test("RAG documents keep their existing route", () => {
  for (const path of ["/docs/notes.txt", "/docs/notes.md", "/docs/paper.pdf"]) {
    assert.equal(classifyDropPaths([path]).kind, "docs", path);
    assert.equal(isComposerAttachmentName(path), false, path);
  }
});

test("a dotfile is not mistaken for a source file", () => {
  // Rust classifies on Path::extension, which ".env" does not have.
  for (const path of ["/p/.env", "/p/.properties", "/p/.log"]) {
    assert.equal(isComposerAttachmentName(path), false, path);
    assert.equal(classifyDropPaths([path]).kind, "unsupported", path);
  }
  assert.equal(isComposerAttachmentName("/p/app.env"), true);
});

test("an unreadable type is still refused", () => {
  assert.equal(classifyDropPaths(["/docs/archive.zip"]).kind, "unsupported");
  assert.equal(classifyDropPaths(["/bin/tool.exe"]).kind, "unsupported");
});

test("frontend and Rust accept the same dropped text extensions", () => {
  const docs = RAG_UPLOAD_ACCEPT.split(",").map((ext) =>
    ext.trim().toLowerCase(),
  );
  const frontend = TEXT_ATTACHMENT_EXTENSIONS.map((ext) => ext.toLowerCase())
    .filter((ext) => !docs.includes(ext))
    .sort();
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_TEXT_ATTACHMENT_EXTS_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => `.${match[1]}`)
    .sort();

  assert.deepEqual(rust, frontend);
});

test("the composer adapter reads the shared text accept list", () => {
  const src = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /class TextAttachmentAdapter implements AttachmentAdapter \{[\s\S]*?accept = TEXT_ATTACHMENT_ACCEPT;/,
  );
  for (const ext of [".cs", ".php", ".js"]) {
    assert.ok(TEXT_ATTACHMENT_ACCEPT.includes(ext), ext);
  }

  const attachmentContentSource = readFileSync(
    new URL("../src/features/chat/attachment-content.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    attachmentContentSource,
    /import \{ TEXT_ATTACHMENT_ACCEPT \} from "\.\/text-attachment-accept";/,
  );
});
