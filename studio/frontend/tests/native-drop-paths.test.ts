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
  TEXT_ATTACHMENT_BASENAMES,
  TEXT_ATTACHMENT_EXTENSIONS,
  isBinaryPropertyList,
  isBinaryTrackerModule,
  MAX_TEXT_ATTACHMENT_BYTES,
  decodeTextAttachmentBytes,
  isBinaryOfficeTemplate,
  isBinaryVobSubSubtitle,
  isCompiledFortranModule,
  pickerAcceptForTextBasenames,
  readTextAttachment,
  readTextAttachmentOnce,
  UndecodableTextError,
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
const RUST_TEXT_ATTACHMENT_NAMES_RE =
  /TEXT_ATTACHMENT_NAMES[^=]*=\s*&\[([^\]]+)\]/s;
const PICKER_BASENAME_CALL_RE = /pickerAcceptForTextBasenames\(enabledAccept\)/;
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

test("the conventional extensionless Containerfile attaches as text", () => {
  for (const path of ["/project/Containerfile", "C:\\project\\CONTAINERFILE"]) {
    assert.equal(classifyDropPaths([path]).kind, "docs", path);
    assert.equal(isComposerAttachmentName(path), true, path);
  }
});

test("the picker remains able to show extensionless text basenames", () => {
  assert.equal(pickerAcceptForTextBasenames(TEXT_ATTACHMENT_ACCEPT), "*");
  assert.equal(
    pickerAcceptForTextBasenames(".txt,text/plain"),
    ".txt,text/plain",
  );

  const threadSource = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(threadSource, PICKER_BASENAME_CALL_RE);
});

test("frontend and Rust accept the same extensionless text names", () => {
  const rustSource = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const rust = [
    ...(rustSource
      .match(RUST_TEXT_ATTACHMENT_NAMES_RE)?.[1]
      .matchAll(RUST_EXTENSION_RE) ?? []),
  ]
    .map((match) => match[1])
    .sort();
  assert.deepEqual(rust, [...TEXT_ATTACHMENT_BASENAMES].sort());
});

test("binary property-list files are distinguished from text forms", async () => {
  assert.equal(
    await isBinaryPropertyList(new File(["bplist00payload"], "settings.plist")),
    true,
  );
  assert.equal(
    await isBinaryPropertyList(
      new File(['<?xml version="1.0"?><plist/>'], "settings.plist"),
    ),
    false,
  );
  assert.equal(
    await isBinaryPropertyList(new File(["bplist00payload"], "settings.txt")),
    false,
  );
  assert.equal(
    await isBinaryPropertyList(
      new File(["bplist00payload"], "Localizable.strings"),
    ),
    true,
  );
  assert.equal(
    await isBinaryPropertyList(
      new File(['"hello" = "world";'], "Localizable.strings"),
    ),
    false,
  );
});

test("binary VobSub files are distinguished from text subtitles", async () => {
  assert.equal(
    await isBinaryVobSubSubtitle(
      new File([new Uint8Array([0x00, 0x00, 0x01, 0xba, 0x44])], "movie.sub"),
    ),
    true,
  );
  assert.equal(
    await isBinaryVobSubSubtitle(new File(["{1}{25}Hello|world"], "movie.sub")),
    false,
  );
  assert.equal(
    await isBinaryVobSubSubtitle(
      new File([new Uint8Array([0x00, 0x00, 0x01, 0xba])], "movie.bin"),
    ),
    false,
  );
});

test("tracker MOD binaries are distinguished from text module files", async () => {
  const tracker = new Uint8Array(1084);
  tracker.set(new TextEncoder().encode("M.K."), 1080);
  assert.equal(
    await isBinaryTrackerModule(new File([tracker], "track.mod")),
    true,
  );
  const soundtracker = new Uint8Array(600 + 1024 + 8);
  soundtracker[43] = 4; // Four big-endian words of sample data.
  soundtracker[45] = 64;
  soundtracker[470] = 1;
  soundtracker[471] = 120;
  assert.equal(
    await isBinaryTrackerModule(new File([soundtracker], "classic.mod")),
    true,
  );
  assert.equal(
    await isBinaryTrackerModule(
      new File(["module example.com/project\n".padEnd(1200, " ")], "go.mod"),
    ),
    false,
  );
  assert.equal(
    await isBinaryTrackerModule(new File([tracker], "track.bin")),
    false,
  );
});

test("compiled Fortran modules are rejected, text .mod files are not", async () => {
  // gfortran gzips its modules, so the tracker-header check never sees them.
  const compiled = new Uint8Array([0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00]);
  assert.equal(
    await isCompiledFortranModule(new File([compiled], "kinds.mod")),
    true,
  );
  assert.equal(
    await isCompiledFortranModule(
      new File(["module example.com/project\n"], "go.mod"),
    ),
    false,
  );
  // An uncompressed gfortran module is readable text and stays accepted.
  assert.equal(
    await isCompiledFortranModule(
      new File(["GFORTRAN module version '15'\n"], "kinds.mod"),
    ),
    false,
  );
  assert.equal(
    await isCompiledFortranModule(new File([compiled], "archive.gz")),
    false,
  );
});

test("a legacy code page subtitle is named, not guessed at", async () => {
  // The same byte is a different letter in windows-1252, windows-1251 and
  // Shift-JIS, so decoding it as any one of them sends confident mojibake.
  const srt = new Uint8Array([
    ...new TextEncoder().encode("1\n00:00:01,000 --> 00:00:02,000\nCaf"),
    0xe9, // valid in several code pages, invalid on its own in UTF-8
    0x0a,
  ]);
  await assert.rejects(
    readTextAttachment(new File([srt], "movie.srt")),
    (error: Error) => {
      assert.ok(error instanceof UndecodableTextError);
      assert.match(error.message, /movie\.srt is not UTF-8 text/);
      return true;
    },
  );
});

test("valid UTF-8 keeps its own decoding", async () => {
  const text = await readTextAttachment(
    new File([new TextEncoder().encode("Caf\u00e9 \u2014 na\u00efve")], "notes.srt"),
  );
  assert.equal(text, "Caf\u00e9 \u2014 na\u00efve");
});

test("a preview cut mid-character stays UTF-8", () => {
  // The bounded preview slices at a byte offset, so the last character can be
  // half-read. That is a truncation, not a legacy encoding.
  const full = new TextEncoder().encode("caf\u00e9");
  const cut = full.subarray(0, full.length - 1);
  assert.equal(decodeTextAttachmentBytes(cut, "notes.txt", true), "caf");
  // A whole file gets no such licence: a dangling lead byte is a bad encoding,
  // not a cut, and dropping it silently would lose the character.
  assert.throws(
    () => decodeTextAttachmentBytes(cut, "notes.txt"),
    (error: Error) => error instanceof UndecodableTextError,
  );
});

test("legacy Office templates are rejected, text templates are not", async () => {
  const ole = new Uint8Array([
    0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0x00, 0x00,
  ]);
  assert.equal(
    await isBinaryOfficeTemplate(new File([ole], "report.dot")),
    true,
  );
  assert.equal(await isBinaryOfficeTemplate(new File([ole], "deck.pot")), true);
  // Graphviz and gettext keep the same extensions and stay accepted.
  assert.equal(
    await isBinaryOfficeTemplate(
      new File(["digraph G { a -> b; }"], "graph.dot"),
    ),
    false,
  );
  assert.equal(
    await isBinaryOfficeTemplate(
      new File(['msgid ""\nmsgstr ""\n'], "messages.pot"),
    ),
    false,
  );
  assert.equal(await isBinaryOfficeTemplate(new File([ole], "deck.ppt")), false);
});

test("an 8-bit email is decoded with the charset it declares", async () => {
  // A standards-valid ISO-8859-1 message is not a guess: the file says so.
  const eml = new Uint8Array([
    ...new TextEncoder().encode(
      "From: a@example.com\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n" +
        "Content-Transfer-Encoding: 8bit\r\n\r\nCaf",
    ),
    0xe9,
    0x0a,
  ]);
  const text = await readTextAttachment(new File([eml], "message.eml"));
  assert.match(text, /Caf\u00e9/);
  assert.equal(text.includes("\uFFFD"), false);
});

test("a folded Content-Type header still yields its charset", async () => {
  const eml = new Uint8Array([
    ...new TextEncoder().encode(
      "Content-Type: text/plain;\r\n\tcharset=\"ISO-8859-1\"\r\n\r\nCaf",
    ),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([eml], "archive.mbox")),
    /Caf\u00e9/,
  );
});

test("a UTF-8 email is not remapped by a stale declaration", async () => {
  // The charset is a fallback for bytes UTF-8 rejects, never a rewrite of text
  // that already decoded.
  const eml = new TextEncoder().encode(
    "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf\u00e9 \u2014 na\u00efve",
  );
  assert.equal(
    await readTextAttachment(new File([eml], "message.eml")),
    "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf\u00e9 \u2014 na\u00efve",
  );
});

test("an email with no declared charset is still named, not guessed", async () => {
  const eml = new Uint8Array([
    ...new TextEncoder().encode("Subject: hi\r\n\r\nCaf"),
    0xe9,
  ]);
  await assert.rejects(
    readTextAttachment(new File([eml], "message.eml")),
    (error: Error) => error instanceof UndecodableTextError,
  );
});

test("the browser text cap matches the native one", () => {
  // Reading happens while attaching now, so an unbounded .mbox would decode
  // gigabytes into the webview before the user could send it.
  assert.equal(MAX_TEXT_ATTACHMENT_BYTES, 20 * 1024 * 1024);
  const rust = readFileSync(
    new URL("../../src-tauri/src/native_intents.rs", import.meta.url),
    "utf8",
  );
  const native = rust.match(
    /const MAX_NATIVE_TEXT_BYTES: u64 = (\d+) \* 1024 \* 1024;/,
  )?.[1];
  assert.ok(native, "native text cap not found");
  assert.equal(
    Number(native) * 1024 * 1024,
    MAX_TEXT_ATTACHMENT_BYTES,
  );
});

test("a text attachment is read once, not once per stage", async () => {
  // The composer decodes while attaching so a bad encoding is reported there;
  // sending must reuse that answer rather than read the whole file again.
  const file = new File(["hello"], "notes.txt");
  let reads = 0;
  const real = file.arrayBuffer.bind(file);
  Object.defineProperty(file, "arrayBuffer", {
    value: () => {
      reads += 1;
      return real();
    },
  });
  assert.equal(await readTextAttachmentOnce(file), "hello");
  assert.equal(await readTextAttachmentOnce(file), "hello");
  assert.equal(reads, 1);
  // A different file is its own entry, not a stale hit.
  assert.equal(await readTextAttachmentOnce(new File(["bye"], "b.txt")), "bye");
});

test("both attachment stages go through the decode-once path", () => {
  const provider = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const adapter = provider.slice(
    provider.indexOf("class TextAttachmentAdapter"),
    provider.indexOf("class HtmlAttachmentAdapter"),
  );
  assert.ok(adapter.length > 0, "text adapter not found");
  assert.equal(adapter.includes("readTextAttachmentOnce"), true);
  // The unbounded read stays out of both stages.
  assert.equal(/await readTextAttachment\(/.test(adapter), false);
  // The size cap comes before any read, including the signature slices.
  assert.ok(
    adapter.indexOf("MAX_TEXT_ATTACHMENT_BYTES") <
      adapter.indexOf("isBinaryPropertyList"),
  );
});

test("legacy M3U playlists are not advertised as UTF-8 text", () => {
  assert.equal(TEXT_ATTACHMENT_EXTENSIONS.includes(".m3u"), false);
  assert.equal(TEXT_ATTACHMENT_EXTENSIONS.includes(".m3u8"), true);
});

test("UTF-16 registry exports are decoded before attachment", async () => {
  const text =
    "Windows Registry Editor Version 5.00\r\n\r\n[HKEY_CURRENT_USER\\Software\\Test]";
  const utf16le = new Uint8Array(2 + text.length * 2);
  utf16le.set([0xff, 0xfe]);
  for (let index = 0; index < text.length; index += 1) {
    const codeUnit = text.charCodeAt(index);
    utf16le[2 + index * 2] = codeUnit & 0xff;
    utf16le[3 + index * 2] = codeUnit >>> 8;
  }

  assert.equal(
    await readTextAttachment(new File([utf16le], "export.reg")),
    text,
  );
  assert.equal(
    await readTextAttachment(new File(["plain UTF-8"], "notes.txt")),
    "plain UTF-8",
  );
});

test("gettext catalogs use the charset declared by their header", async () => {
  const before =
    'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=ISO-8859-1\\n"\n\nmsgid "coffee"\nmsgstr "caf';
  const after = '"\n';
  const prefix = new TextEncoder().encode(before);
  const suffix = new TextEncoder().encode(after);
  const encoded = new Uint8Array(prefix.length + 1 + suffix.length);
  encoded.set(prefix);
  encoded[prefix.length] = 0xe9;
  encoded.set(suffix, prefix.length + 1);

  assert.equal(
    await readTextAttachment(new File([encoded], "messages.po")),
    `${before}é${after}`,
  );
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
  assert.match(src, /if \(await isBinaryTrackerModule\(file\)\)/);
  for (const ext of [".cs", ".php", ".js"]) {
    assert.ok(TEXT_ATTACHMENT_ACCEPT.includes(ext), ext);
  }

  const attachmentContentSource = readFileSync(
    new URL("../src/features/chat/attachment-content.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    attachmentContentSource,
    /import \{[\s\S]*?TEXT_ATTACHMENT_ACCEPT[\s\S]*?decodeTextAttachmentBytes[\s\S]*?\} from "\.\/text-attachment-accept";/,
  );
});
