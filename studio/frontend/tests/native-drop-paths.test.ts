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
  isTextAttachmentName,
} from "../src/features/chat/text-attachment-accept.ts";
import {
  AUDIO_ATTACHMENT_ACCEPT,
  AUDIO_PICKER_ACCEPT,
  isAudioAttachmentFile,
} from "../src/lib/audio-utils.ts";
import {
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
import {
  VIDEO_ACCEPT,
  classifiedAttachmentFile,
  classifiedAttachmentFiles,
  isAudioOnly3gpBytes,
  isVideoFile,
} from "../src/lib/video-utils.ts";
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

test("every basename the picker claims is one the drop paths accept", () => {
  // assistant-ui derives the extension as `.${name.split(".").pop()}`, so an
  // extensionless "Dockerfile" matches the ".dockerfile" token and the picker
  // takes it. A drop of the same file has to agree, or the two disagree on one
  // conventional build file.
  for (const name of ["Dockerfile", "Makefile", "Containerfile"]) {
    const derived = `.${name.split(".").pop()!.toLowerCase()}`;
    assert.ok(
      TEXT_ATTACHMENT_EXTENSIONS.includes(derived),
      `${name} is claimed through ${derived}`,
    );
    assert.equal(isTextAttachmentName(name), true, `${name} is droppable`);
  }
  // A name with no matching token stays out of both, rather than being widened.
  assert.equal(TEXT_ATTACHMENT_EXTENSIONS.includes(".gnumakefile"), false);
  assert.equal(isTextAttachmentName("GNUmakefile"), false);
});

test("a container declaring two charsets is refused, not part-decoded", async () => {
  // Decoding the whole file with the first declaration would corrupt every
  // later part, which is the failure mode the strict decoder exists to avoid.
  const mixed = new Uint8Array([
    ...new TextEncoder().encode(
      "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...new TextEncoder().encode(
      "\r\n--b\r\nContent-Type: text/plain; charset=windows-1251\r\n\r\n",
    ),
    0xcf,
    0xf0,
  ]);
  await assert.rejects(
    readTextAttachment(new File([mixed], "thread.mbox")),
    (error: Error) => {
      assert.ok(error instanceof UndecodableTextError);
      assert.match(error.message, /declares more than one charset/);
      return true;
    },
  );
});

test("a multipart message whose parts agree is decoded with that charset", async () => {
  // The outer header carries no charset, so reading only the first Content-Type
  // rejected a file that says plainly what it is.
  const eml = new Uint8Array([
    ...new TextEncoder().encode(
      "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([eml], "message.eml")),
    /Caf\u00e9/,
  );
});

test("a vCard property charset is honoured like an email's", async () => {
  // vCard 2.1 puts the encoding on the property rather than in a header.
  const vcf = new Uint8Array([
    ...new TextEncoder().encode(
      "BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf",
    ),
    0xe9,
    ...new TextEncoder().encode("\r\nEND:VCARD\r\n"),
  ]);
  assert.match(
    await readTextAttachment(new File([vcf], "contact.vcf")),
    /Caf\u00e9/,
  );
});

test("compare mode takes the same audio files the chat composer does", () => {
  // Browsers leave file.type empty for several of these containers, so a
  // MIME-only check dropped them silently.
  for (const name of ["clip.wma", "voice.amr", "note.caf", "take.aiff"]) {
    assert.equal(isAudioAttachmentFile(new File([], name, { type: "" })), true);
  }
  assert.equal(
    isAudioAttachmentFile(new File([], "song.mp3", { type: "audio/mpeg" })),
    true,
  );
  assert.equal(isAudioAttachmentFile(new File([], "notes.txt", { type: "" })), false);
  // The compare composer classifies through the shared helper, not file.type.
  const composer = readFileSync(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );
  assert.equal(/file\.type\.match\(\/\^audio/.test(composer), false);
  assert.match(composer, /isAudioAttachmentFile\(file\)/);
  assert.match(composer, /accept=\{AUDIO_PICKER_ACCEPT\}/);
});

test("a declaration past the first pages is not missed", async () => {
  // A prefix-scan cutoff decoded the whole archive as the first message's
  // charset, so a later message in another one came out as mojibake. Any cutoff
  // has that failure, which is why the scan reads the file the cap already
  // bounds. Padded well past the 64 KiB the scan used to stop at.
  const enc = new TextEncoder();
  const mbox = new Uint8Array([
    ...enc.encode(
      "From a@example.com Mon Jan  1 00:00:00 2024\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\n" +
        "x".repeat(70 * 1024) +
        "\r\n\r\nFrom b@example.com Mon Jan  1 00:00:00 2024\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n\r\n",
    ),
    0xcf,
    0xf0,
    0xe8,
    0xe2,
    0xe5,
    0xf2,
  ]);
  await assert.rejects(
    readTextAttachment(new File([mbox], "inbox.mbox")),
    (error: Error) => {
      assert.ok(error instanceof UndecodableTextError);
      assert.match(error.message, /ISO-8859-1, windows-1251/);
      return true;
    },
  );
  // The source carries no byte ceiling on the scan for a cutoff to creep back in.
  const source = readFileSync(
    new URL(
      "../src/features/chat/text-attachment-accept.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(source.includes("DECLARATION_SCAN_BYTES"), false);
});

test("a header-shaped line in the body is not a declaration", async () => {
  // Quoted mail and config snippets put "Content-Type:" at the start of a body
  // line. Counting those as declarations refused files that say one thing.
  const eml = new Uint8Array([
    ...new TextEncoder().encode(
      "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\n" +
        "Here is the config we discussed:\r\n" +
        "Content-Type: text/html; charset=windows-1251\r\n" +
        "and that is the end of it. Caf",
    ),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([eml], "message.eml")),
    /Caf\u00e9/,
  );
});

test("parts after a boundary are still read as headers", async () => {
  // The narrowing must not lose real part headers, which is what made the
  // multipart case work in the first place.
  const enc = new TextEncoder();
  const mixed = new Uint8Array([
    ...enc.encode(
      "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode("\r\n--b\r\nContent-Type: text/plain; charset=windows-1251\r\n\r\n"),
    0xcf,
    0xf0,
  ]);
  await assert.rejects(
    readTextAttachment(new File([mixed], "message.eml")),
    (error: Error) => {
      assert.match(error.message, /ISO-8859-1, windows-1251/);
      return true;
    },
  );
});

test("a signature marker is not a MIME boundary", async () => {
  // "-- " opens a signature block in most mail clients. Treating every line
  // starting with "--" as a boundary put the body back into header mode, so a
  // quoted header below it became a second declaration and refused the file.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode("Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf"),
    0xe9,
    ...enc.encode(
      "\r\n-- \r\nSent from a phone\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n",
    ),
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Caf\u00e9/);
});

test("a closing delimiter ends its part instead of starting one", async () => {
  // "--b--" closes the multipart; the epilogue after it is body text, so a
  // header-shaped line there declares nothing.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode(
      "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode("\r\n--b--\r\nContent-Type: text/plain; charset=windows-1251\r\n"),
  ]);
  assert.match(await readTextAttachment(new File([eml], "thread.mbox")), /Caf\u00e9/);
});

test("a vCard charset in a value is not a property parameter", async () => {
  // The escaped semicolon is part of the NOTE text. Matching it added a second
  // declaration and refused a card that names exactly one charset.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nNOTE:Write \\;CHARSET=windows-1251 to pin it\r\nEND:VCARD\r\n"),
  ]);
  assert.match(await readTextAttachment(new File([vcf], "contact.vcf")), /Caf\u00e9/);
});

test("each vCard property is read under its own declared charset", async () => {
  // CHARSET is a property parameter, so a card naming two of them says plainly
  // what each value holds. Reading it as one unit corrupted all but one.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nORG;CHARSET=windows-1251:"),
    0xcf,
    0xf0,
    ...enc.encode("\r\nTEL:+15551234\r\nEND:VCARD\r\n"),
  ]);
  const text = await readTextAttachment(new File([vcf], "contact.vcf"));
  assert.match(text, /FN;CHARSET=windows-1252:Café/);
  assert.match(text, /ORG;CHARSET=windows-1251:Пр/);
  // Names, parameters, undeclared properties and line endings survive intact.
  assert.match(text, /TEL:\+15551234/);
  assert.ok(text.startsWith("BEGIN:VCARD\r\nVERSION:2.1\r\n"));
  assert.ok(text.endsWith("END:VCARD\r\n"));
});

test("a folded value keeps the charset of the property above it", async () => {
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nNOTE;CHARSET=windows-1251:"),
    0xcf,
    ...enc.encode("\r\n "),
    0xf0,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  const text = await readTextAttachment(new File([vcf], "contact.vcf"));
  assert.match(text, /NOTE;CHARSET=windows-1251:П\r\n р/);
});

test("a vCard naming a charset it does not hold is still refused", async () => {
  // The per-property read only replaces the refusal when it can account for
  // every value; a property whose bytes break its own declaration is not that.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=utf-8:Caf"),
    0xe9,
    ...enc.encode("\r\nORG;CHARSET=windows-1251:"),
    0xcf,
    0xf0,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  await assert.rejects(
    readTextAttachment(new File([vcf], "contact.vcf")),
    (error: Error) => {
      // The property that broke its declaration, not a list of every charset
      // the card names: only one of them is the reason it cannot be read.
      assert.match(
        error.message,
        /declares charset "utf-8" but does not hold valid utf-8 text/,
      );
      return true;
    },
  );
});

test("a multipart mail declaring two charsets is still refused", async () => {
  // Only the vCard reading changed: a MIME container needs a parser this is not.
  const enc = new TextEncoder();
  const mixed = new Uint8Array([
    ...enc.encode(
      "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode("\r\n--b\r\nContent-Type: text/plain; charset=windows-1251\r\n\r\n"),
    0xcf,
    0xf0,
  ]);
  await assert.rejects(
    readTextAttachment(new File([mixed], "message.eml")),
    (error: Error) => {
      assert.match(error.message, /declares more than one charset/);
      return true;
    },
  );
});

test("an XML prolog encoding is honoured for every XML dialect", async () => {
  // .resx, .xliff and friends are XML documents that state their encoding, so
  // refusing them for carrying the bytes they describe was wrong.
  const enc = new TextEncoder();
  for (const name of ["Strings.resx", "ui.xliff", "app.xlf", "icon.svg"]) {
    const bytes = new Uint8Array([
      ...enc.encode('<?xml version="1.0" encoding="windows-1252"?>\n<root>Caf'),
      0xe9,
      ...enc.encode("</root>\n"),
    ]);
    assert.match(
      await readTextAttachment(new File([bytes], name)),
      /Caf\u00e9/,
      `${name} decodes under its declared encoding`,
    );
  }
  // A prolog only counts at the start of the document, where the spec puts it.
  const late = new Uint8Array([
    ...enc.encode('# notes\n<?xml version="1.0" encoding="windows-1252"?>\nCaf'),
    0xe9,
  ]);
  await assert.rejects(
    readTextAttachment(new File([late], "notes.txt")),
    (error: Error) => error instanceof UndecodableTextError,
  );
});

test("the tracker magic tables agree, and cover ProTracker's second marker", async () => {
  // The two lists are hand-mirrored, so drift is only visible if something
  // compares them. !PM! is the marker that was missing from both: file(1) lists
  // it at 1080 beside M.K., and a 31-sample module puts byte 470 inside a
  // sample name rather than the order table, so the Soundtracker fallback does
  // not catch one either and the module was read as UTF-8 text.
  const rust = readFileSync(
    new URL("../../src-tauri/src/native_path_policy.rs", import.meta.url),
    "utf8",
  );
  const table = rust.match(
    /const TRACKER_MOD_MAGICS: &\[&\[u8; 4\]\] = &\[([\s\S]*?)\];/,
  );
  assert.ok(table, "TRACKER_MOD_MAGICS not found in native_path_policy.rs");
  const rustMagics = [...table[1]!.matchAll(/b"((?:[^"\\]|\\.){1,8})"/g)].map((m) =>
    m[1]!.replace(/\\0/g, "\0"),
  );
  assert.ok(rustMagics.includes("!PM!"), "Rust table is missing !PM!");

  const module = await import("../src/features/chat/text-attachment-accept.ts");
  const source = readFileSync(
    new URL("../src/features/chat/text-attachment-accept.ts", import.meta.url),
    "utf8",
  );
  const tsTable = source.match(/const TRACKER_MOD_MAGICS = new Set\(\[([\s\S]*?)\]\)/);
  assert.ok(tsTable, "TRACKER_MOD_MAGICS not found in text-attachment-accept.ts");
  const tsMagics = [...tsTable[1]!.matchAll(/"((?:[^"\\]|\\.){1,8})"/g)].map((m) =>
    m[1]!.replace(/\\0/g, "\0"),
  );
  assert.deepEqual([...tsMagics].sort(), [...rustMagics].sort());
  assert.ok(module.isBinaryTrackerModule, "the sniffer is still exported");

  // And the marker is actually acted on, not merely listed.
  const bytes = new Uint8Array(1084 + 4);
  bytes.set(new TextEncoder().encode("!PM!"), 1080);
  assert.equal(
    await module.isBinaryTrackerModule(new File([bytes], "tune.mod")),
    true,
  );
  // A go.mod of the same length is still text.
  const text = new Uint8Array(1084 + 4).fill(0x20);
  assert.equal(
    await module.isBinaryTrackerModule(new File([text], "go.mod")),
    false,
  );
});

test("a charset with no decoder here is reported, not swallowed", async () => {
  // The composer only toasts an UndecodableTextError; a bare Error failed that
  // check and the attachment vanished with no message at all. These are not
  // exotic labels: the Encoding Standard maps the ISO-2022-KR, ISO-2022-CN and
  // HZ-GB-2312 families to "replacement", which TextDecoder is required to
  // refuse, so a Korean or Chinese card declaring its own charset went out
  // silently on every engine.
  for (const charset of ["ISO-2022-KR", "HZ-GB-2312", "ISO-2022-CN", "cp437"]) {
    const card = new File(
      [`BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=${charset}:name\r\nEND:VCARD\r\n`],
      "contact.vcf",
    );
    await assert.rejects(
      readTextAttachment(card),
      (error: Error) => {
        assert.ok(
          error instanceof UndecodableTextError,
          `${charset} produced a ${error.name}, which the composer does not toast`,
        );
        assert.match(error.message, new RegExp(charset, "i"));
        return true;
      },
      charset,
    );
  }
});

test("a declared charset is decoded as strictly as an undeclared one", async () => {
  // The declared-charset paths used a lenient decoder, so a file claiming UTF-8
  // and holding broken bytes came through as replacement characters, which is
  // the corruption the strict default was added to stop.
  const enc = new TextEncoder();
  const broken = [0xc3, 0x28]; // a lead byte followed by an invalid continuation
  for (const [name, head] of [
    [
      "messages.po",
      'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n\nmsgstr "Caf',
    ],
    ["strings.resx", '<?xml version="1.0" encoding="UTF-8"?><r>Caf'],
  ] as const) {
    await assert.rejects(
      readTextAttachment(new File([new Uint8Array([...enc.encode(head), ...broken])], name)),
      (error: Error) => {
        assert.ok(error instanceof UndecodableTextError);
        assert.match(error.message, /declares charset "UTF-8"/);
        return true;
      },
      `${name} is refused rather than corrupted`,
    );
  }
  // A single-byte charset maps every byte, so those files are unaffected.
  const latin = new Uint8Array([
    ...enc.encode('<?xml version="1.0" encoding="windows-1252"?><r>Caf'),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([latin], "strings.resx")),
    /Caf\u00e9/,
  );
  // A bounded preview keeps its allowance for the character the slice cut.
  const cut = new Uint8Array([
    ...enc.encode('<?xml version="1.0" encoding="UTF-8"?><r>caf'),
    0xc3,
  ]);
  assert.match(
    decodeTextAttachmentBytes(cut, "strings.resx", true),
    /<r>caf$/,
  );
});

test("an XML prolog decides the encoding before UTF-8 is tried", async () => {
  // 0xC3 0xA9 is valid UTF-8 for "e-acute", so a UTF-8-first decode succeeded and
  // the prolog was never consulted. Per the spec those bytes are two windows-1252
  // characters, which is what an XML parser reads and what we must send.
  const bytes = new Uint8Array([
    ...new TextEncoder().encode('<?xml version="1.0" encoding="windows-1252"?><r>'),
    0xc3,
    0xa9,
    ...new TextEncoder().encode("</r>"),
  ]);
  const text = await readTextAttachment(new File([bytes], "strings.resx"));
  assert.match(text, /\u00c3\u00a9/, "decoded as the prolog says, not as UTF-8");
});

test("a BOM-marked file decodes as strictly as everything else", async () => {
  // An odd trailing byte used to be padded with a replacement character and
  // attached as though it had been read.
  const odd = new Uint8Array([0xff, 0xfe, 0x61, 0x00, 0x62]);
  await assert.rejects(
    readTextAttachment(new File([odd], "export.reg")),
    (error: Error) => error instanceof UndecodableTextError,
  );
  // A bounded preview still drops only the incomplete unit at the cut.
  assert.equal(decodeTextAttachmentBytes(odd, "export.reg", true), "a");
  // Well-formed UTF-16 is unaffected.
  const good = new Uint8Array([0xff, 0xfe, 0x61, 0x00, 0x62, 0x00]);
  assert.equal(await readTextAttachment(new File([good], "export.reg")), "ab");
});

test("a body of boundary-shaped lines does not stall the composer", () => {
  // Restarting the header search from every candidate boundary was quadratic:
  // a megabyte of diff hunks took 14 seconds, inside add(), on the UI thread.
  const body = "--- a/file.txt\n".repeat(70_000);
  const bytes = new Uint8Array([
    ...new TextEncoder().encode(
      "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\n" + body,
    ),
    0xe9,
  ]);
  const started = performance.now();
  decodeTextAttachmentBytes(bytes, "thread.mbox");
  assert.ok(
    performance.now() - started < 2_000,
    "the scan is linear in the file, not quadratic in its boundary-shaped lines",
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

// Mirrors `bmff_box` and `three_gp_with_tracks` in native_intents.rs, so the
// browser classifier is tested against the same containers as the native one.
// The return types name their buffer: a bare Uint8Array is generic over
// ArrayBufferLike, which includes SharedArrayBuffer, and BlobPart takes neither
// that nor a view onto it. Every one of these ends up inside a File.
function bmffBox(kind: string, payload: Uint8Array): Uint8Array<ArrayBuffer> {
  const boxed = new Uint8Array(8 + payload.length);
  new DataView(boxed.buffer).setUint32(0, boxed.length);
  boxed.set(new TextEncoder().encode(kind), 4);
  boxed.set(payload, 8);
  return boxed;
}

function threeGpWithTracks(handlers: readonly string[]): Uint8Array<ArrayBuffer> {
  const traks: Uint8Array<ArrayBuffer>[] = [];
  for (const handler of handlers) {
    const hdlrPayload = new Uint8Array(12);
    hdlrPayload.set(new TextEncoder().encode(handler), 8);
    traks.push(bmffBox("trak", bmffBox("mdia", bmffBox("hdlr", hdlrPayload))));
  }
  const moovPayload = new Uint8Array(
    traks.reduce((total, trak) => total + trak.length, 0),
  );
  let offset = 0;
  for (const trak of traks) {
    moovPayload.set(trak, offset);
    offset += trak.length;
  }
  return bmffBox("moov", moovPayload);
}

test("3GP tracks decide audio or video, as they do natively", () => {
  assert.equal(isAudioOnly3gpBytes(threeGpWithTracks(["soun"])), true);
  assert.equal(isAudioOnly3gpBytes(threeGpWithTracks(["vide"])), false);
  assert.equal(isAudioOnly3gpBytes(threeGpWithTracks(["soun", "vide"])), false);
  // Not a container at all, and a truncated one.
  assert.equal(isAudioOnly3gpBytes(new Uint8Array([1, 2, 3])), false);
  assert.equal(
    isAudioOnly3gpBytes(threeGpWithTracks(["soun"]).subarray(0, 20)),
    false,
  );
});

test("an audio-only 3GP recording is classified as audio, not video", async () => {
  // The browser answers "" or video/3gpp for both kinds, and the video accept
  // list claims .3gp, so without this the recording reached the video adapter.
  const recording = new File([threeGpWithTracks(["soun"])], "voice.3gp", {
    type: "",
  });
  const classified = await classifiedAttachmentFile(recording);
  assert.equal(classified.type, "audio/3gpp");
  assert.equal(classified.name, "voice.3gp");
  assert.equal(isAudioAttachmentFile(classified), true);
});

test("a real 3GP clip stays video", async () => {
  const clip = new File([threeGpWithTracks(["soun", "vide"])], "clip.3gp", {
    type: "video/3gpp",
  });
  assert.equal(await classifiedAttachmentFile(clip), clip);
});

test("classification leaves every other container untouched", async () => {
  for (const [name, type] of [
    ["clip.mp4", "video/mp4"],
    ["take.mkv", ""],
    ["voice.m4a", ""],
    ["notes.txt", "text/plain"],
  ] as const) {
    const file = new File([new Uint8Array([0, 1, 2, 3])], name, { type });
    assert.equal(await classifiedAttachmentFile(file), file, name);
  }
});

test("the restamped recording routes to the audio adapter", async () => {
  // The composite's own matcher, not a copy of it: audio is registered before
  // video, so a file the audio accept list claims never reaches the video one.
  const { fileMatchesAccept } = (await import(
    new URL(
      "../node_modules/@assistant-ui/core/dist/adapters/attachment.js",
      import.meta.url,
    ).href
  )) as { fileMatchesAccept: (file: File, accept: string) => boolean };
  const recording = new File([threeGpWithTracks(["soun"])], "voice.3gp", {
    type: "",
  });
  assert.equal(fileMatchesAccept(recording, AUDIO_ATTACHMENT_ACCEPT), false);
  assert.equal(fileMatchesAccept(recording, VIDEO_ACCEPT), true);

  const classified = await classifiedAttachmentFile(recording);
  assert.equal(fileMatchesAccept(classified, AUDIO_ATTACHMENT_ACCEPT), true);
});

test("a file dialog offers 3GP recordings, and routing still does not", () => {
  // A dialog decides what is selectable, so it has to name .3gp: a platform
  // that maps it to video/3gpp, or to nothing, greys the recording out. Routing
  // must not, or the audio adapter would claim every 3GP video ahead of the
  // video one, which is matched after it.
  assert.ok(AUDIO_PICKER_ACCEPT.split(",").includes(".3gp"));
  assert.equal(AUDIO_ATTACHMENT_ACCEPT.split(",").includes(".3gp"), false);
  assert.ok(AUDIO_PICKER_ACCEPT.startsWith(AUDIO_ATTACHMENT_ACCEPT));

  const adapter = readFileSync(
    new URL(
      "../src/features/chat/audio-attachment-adapter.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(adapter, /accept = AUDIO_ATTACHMENT_ACCEPT;/);
  assert.equal(/AUDIO_PICKER_ACCEPT/.test(adapter), false);
});

test("a 3GP clip picked through the audio dialog is still refused as video", async () => {
  const clip = new File([threeGpWithTracks(["soun", "vide"])], "clip.3gp", {
    type: "video/3gpp",
  });
  const [classified] = await classifiedAttachmentFiles([clip]);
  assert.equal(isAudioAttachmentFile(classified!), false);
  assert.equal(isVideoFile(classified!), true);
});

test("the composer classifies before an adapter is picked", () => {
  // A composite matches on name and MIME synchronously, so the restamping has
  // to happen in the wrapper above it rather than inside an adapter.
  const src = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    src,
    /class PreStreamAwareAttachmentAdapter[\s\S]*?if \(!needsAttachmentTrackInspection\(state\.file\)\)[\s\S]*?await classifiedAttachmentFile\(state\.file\)/,
  );
  const composer = readFileSync(
    new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
    "utf8",
  );
  assert.match(composer, /await classifiedAttachmentFiles\(input\)/);
});

test("a From line separates messages in an archive, not in one message", async () => {
  // An mbox escapes a body line that starts with "From "; a standalone .eml has
  // no separator at all, so an ordinary sentence opening that way is body text.
  const enc = new TextEncoder();
  const bytes = new Uint8Array([
    ...enc.encode("Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf"),
    0xe9,
    ...enc.encode(
      "\r\nFrom here on it quotes a header:\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n",
    ),
  ]);
  assert.match(
    await readTextAttachment(new File([bytes], "message.eml")),
    /Café/,
  );
  await assert.rejects(
    readTextAttachment(new File([bytes], "thread.mbox")),
    (error: Error) => {
      assert.match(error.message, /ISO-8859-1, windows-1251/);
      return true;
    },
  );
});

function threeGpFile(
  name: string,
  handlers: readonly string[],
  mdatBytes: number,
): File {
  const ftyp = bmffBox("ftyp", new TextEncoder().encode("3gp4isom"));
  const moov = threeGpWithTracks(handlers);
  const mdat = bmffBox("mdat", new Uint8Array(mdatBytes));
  return new File([ftyp, moov, mdat], name, { type: "" });
}

/** Records every range read from a file, and forbids reading it whole. */
function watchReads(file: File): Array<[number, number]> {
  const reads: Array<[number, number]> = [];
  const slice = file.slice.bind(file);
  Object.defineProperty(file, "slice", {
    value: (start: number, end: number) => {
      reads.push([start, end]);
      return slice(start, end);
    },
  });
  Object.defineProperty(file, "arrayBuffer", {
    value: () => {
      throw new Error("the whole container must not be read");
    },
  });
  return reads;
}

test("only the track table is read, not the samples beside it", async () => {
  // moov holds the handlers and mdat the audio, so reading the file to reach a
  // handler retains the whole clip: 64 MB for one, and that per file in a drop.
  const file = threeGpFile("voice.3gp", ["soun"], 2 * 1024 * 1024);
  const reads = watchReads(file);

  const classified = await classifiedAttachmentFile(file);
  assert.equal(classified.type, "audio/3gpp");
  const bytesRead = reads.reduce((total, [start, end]) => total + end - start, 0);
  assert.ok(
    bytesRead < 4096,
    `read ${bytesRead} bytes of a ${file.size}-byte container`,
  );
});

test("a dropped batch is inspected one container at a time", async () => {
  // Promise.all put every container's payload in memory at once, so ten clips
  // at the 64 MB limit came to 640 MB before anything could be rejected.
  const files = ["a.3gp", "b.3gp", "c.3gp"].map((name) =>
    threeGpFile(name, ["soun"], 64 * 1024),
  );
  const order: string[] = [];
  for (const file of files) {
    const slice = file.slice.bind(file);
    Object.defineProperty(file, "slice", {
      value: (start: number, end: number) => {
        order.push(file.name);
        return slice(start, end);
      },
    });
  }

  const classified = await classifiedAttachmentFiles(files);
  assert.deepEqual(
    classified.map((file) => file.type),
    ["audio/3gpp", "audio/3gpp", "audio/3gpp"],
  );
  // Grouped, not interleaved: every read of a file precedes the next one's.
  assert.deepEqual([...new Set(order)], ["a.3gp", "b.3gp", "c.3gp"]);
  assert.deepEqual(
    order,
    [...order].sort((left, right) => left.localeCompare(right)),
  );
});

test("a vCard charset decides before UTF-8, like the other declarations", async () => {
  // C3 A9 is valid UTF-8 for "é" and two windows-1252 characters. The card says
  // which it holds, so decoding as UTF-8 delivered characters it never named.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xc3,
    0xa9,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  assert.match(
    await readTextAttachment(new File([vcf], "contact.vcf")),
    /CafÃ©/,
  );
  // A card that names UTF-8 still reads as UTF-8, so nothing regresses there.
  const utf8 = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:3.0\r\nFN;CHARSET=UTF-8:Caf"),
    0xc3,
    0xa9,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  assert.match(await readTextAttachment(new File([utf8], "modern.vcf")), /Café/);
  // A card with no charset parameter at all is untouched by this.
  const plain = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:4.0\r\nFN:Caf"),
    0xc3,
    0xa9,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  assert.match(await readTextAttachment(new File([plain], "v4.vcf")), /Café/);
});

test("a mail Content-Type stays a fallback, unlike the vCard parameter", async () => {
  // Clients mislabel 8-bit mail constantly, so valid UTF-8 outranks the header.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode("Content-Type: text/plain; charset=windows-1252\r\n\r\nCaf"),
    0xc3,
    0xa9,
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Café/);
});

test("a boundary does not carry into the next message of an archive", async () => {
  // Each message declares its own. A later body line repeating an earlier one
  // reopened the headers there, and the next header-shaped line counted.
  const enc = new TextEncoder();
  const mbox = new Uint8Array([
    ...enc.encode(
      "From sender@example.com Thu Aug 27 00:00:00 2026\r\n" +
        "Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode(
      "\r\nFrom other@example.com Thu Aug 27 01:00:00 2026\r\n" +
        "Subject: a plain message\r\n\r\n" +
        "The separator line below is body text here:\r\n--b\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n",
    ),
  ]);
  assert.match(await readTextAttachment(new File([mbox], "thread.mbox")), /Café/);
});

test("a gettext header past the prefix is still read", async () => {
  // A catalog with more than 64 KiB of translator comments above its header was
  // refused for carrying exactly the bytes that header describes.
  const enc = new TextEncoder();
  const comments = "# a translator comment line\n".repeat(4_000);
  assert.ok(comments.length > 64 * 1024);
  const po = new Uint8Array([
    ...enc.encode(
      `${comments}msgid ""\nmsgstr ""\n` +
        '"Content-Type: text/plain; charset=windows-1251\\n"\n\n' +
        'msgid "hi"\nmsgstr "',
    ),
    0xcf,
    0xf0,
    ...enc.encode('"\n'),
  ]);
  assert.match(await readTextAttachment(new File([po], "ru.po")), /Пр/);
});

test("the reference picker takes the formats its drop path does", async () => {
  const { REFERENCE_PICKER_ACCEPT, referenceFileRejection } = await import(
    "../src/features/video/reference-budget.ts"
  );
  // A browser answers "" for these, so `audio/*` alone hid and then refused them.
  for (const name of ["voice.amr", "clip.wma", "note.caf", "take.aiff"]) {
    assert.equal(
      referenceFileRejection("audio", { type: "", size: 10, name }),
      null,
      name,
    );
    assert.ok(REFERENCE_PICKER_ACCEPT.audio.includes(name.slice(name.indexOf("."))));
  }
  assert.equal(
    referenceFileRejection("video", { type: "", size: 10, name: "take.mkv" }),
    null,
  );
  // The kind check still holds, by name as well as by type.
  assert.equal(
    referenceFileRejection("audio", { type: "", size: 10, name: "clip.mkv" }),
    "Please choose an audio file",
  );
  assert.equal(
    referenceFileRejection("video", { type: "image/png", size: 10 }),
    "Please choose a video file",
  );
  // Every extension the native drop accepts is offered by the dialog too.
  for (const ext of CHAT_AUDIO_DROP_ACCEPT.split(",")) {
    assert.ok(REFERENCE_PICKER_ACCEPT.audio.split(",").includes(ext), ext);
  }
  for (const ext of CHAT_VIDEO_DROP_ACCEPT.split(",")) {
    assert.ok(REFERENCE_PICKER_ACCEPT.video.split(",").includes(ext), ext);
  }

  const picker = readFileSync(
    new URL("../src/features/video/reference-picker.tsx", import.meta.url),
    "utf8",
  );
  assert.match(picker, /accept=\{REFERENCE_PICKER_ACCEPT\[kind\]\}/);
});

test("one vCard declaration speaks for its property, not for the file", async () => {
  // A 2.1 card beside a 3.0 one, which declares nothing because it cannot. The
  // whole-file reading turned the 3.0 record's UTF-8 into windows-1252 mojibake.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nEND:VCARD\r\nBEGIN:VCARD\r\nVERSION:3.0\r\nFN:Stra"),
    0xc3,
    0x9f,
    ...enc.encode("e\r\nEND:VCARD\r\n"),
  ]);
  const text = await readTextAttachment(new File([vcf], "export.vcf"));
  assert.match(text, /FN;CHARSET=windows-1252:Café/);
  assert.match(text, /FN:Straße/);
});

test("a card whose value breaks its declaration still reads as one unit", async () => {
  // Falling back has to keep working: the per-property read is all-or-nothing.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nNOTE:plain "),
    0xe9,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  // NOTE declares nothing and is not UTF-8, so the per-property read gives up
  // and the single declaration is applied to the file as it was before.
  assert.match(
    await readTextAttachment(new File([vcf], "contact.vcf")),
    /NOTE:plain é/,
  );
});

test("a folded boundary parameter is still a boundary", async () => {
  // "multipart/mixed;\r\n boundary=..." is an ordinary wrap. Missing it meant
  // no part header was ever scanned, so the charset they declare went too.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode(
      'Content-Type: multipart/mixed;\r\n boundary="part"\r\n\r\n--part\r\n' +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode("\r\n--part--\r\n"),
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Café/);
});

test("a nested boundary stops being one after its multipart closes", async () => {
  // The inner delimiter is finished at "--inner--", so a sibling part repeating
  // that line is body text and the header-shaped line below it declares nothing.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode(
      "Content-Type: multipart/mixed; boundary=outer\r\n\r\n--outer\r\n" +
        "Content-Type: multipart/alternative; boundary=inner\r\n\r\n--inner\r\n" +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
    ...enc.encode(
      "\r\n--inner--\r\n--outer\r\n" +
        "Content-Type: text/plain\r\n\r\n" +
        "The sibling quotes the inner delimiter:\r\n--inner\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n--outer--\r\n",
    ),
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Café/);
});

test("two spellings of one encoding are one declaration", async () => {
  // windows-1252, CP1252 and latin1 are three names for the same decoder, so an
  // archive using them in different messages is not a multi-charset one.
  const enc = new TextEncoder();
  for (const [first, second] of [
    ["windows-1252", "CP1252"],
    ["ISO-8859-1", "latin1"],
  ] as const) {
    const mbox = new Uint8Array([
      ...enc.encode(
        "From a@example.com Mon Jan  1 00:00:00 2024\r\n" +
          `Content-Type: text/plain; charset=${first}\r\n\r\nCaf`,
      ),
      0xe9,
      ...enc.encode(
        "\r\n\r\nFrom b@example.com Mon Jan  1 00:00:00 2024\r\n" +
          `Content-Type: text/plain; charset=${second}\r\n\r\nna`,
      ),
      0xef,
      ...enc.encode("ve\r\n"),
    ]);
    const text = await readTextAttachment(new File([mbox], "inbox.mbox"));
    assert.match(text, /Café/, `${first} + ${second}`);
    assert.match(text, /naïve/, `${first} + ${second}`);
  }
  // Genuinely different encodings are still two.
  const mixed = new Uint8Array([
    ...enc.encode("Content-Type: text/plain; charset=windows-1252\r\n\r\nCaf"),
    0xe9,
    ...enc.encode(
      "\r\n\r\nFrom b@example.com Mon Jan  1 00:00:00 2024\r\n" +
        "Content-Type: text/plain; charset=windows-1251\r\n\r\n",
    ),
    0xcf,
    0xf0,
  ]);
  await assert.rejects(
    readTextAttachment(new File([mixed], "inbox.mbox")),
    (error: Error) => {
      assert.match(error.message, /windows-1252, windows-1251/);
      return true;
    },
  );
});

test("a wide XML declaration still names its encoding", async () => {
  // The grammar puts no bound on the whitespace between the parts, so a fixed
  // prefix could cut `encoding` off and refuse a file that states it plainly.
  const enc = new TextEncoder();
  const padding = " ".repeat(400);
  const xml = new Uint8Array([
    ...enc.encode(`<?xml version="1.0"${padding}encoding="windows-1252"?><t>Caf`),
    0xe9,
    ...enc.encode("</t>"),
  ]);
  assert.match(await readTextAttachment(new File([xml], "ui.resx")), /Café/);
  // A document that does not open with a declaration is untouched by the scan.
  const plain = new Uint8Array([
    ...enc.encode("<t>Caf"),
    0xc3,
    0xa9,
    ...enc.encode("</t>"),
  ]);
  assert.match(await readTextAttachment(new File([plain], "ui.resx")), /Café/);
});

test("the audio reference picker reads a 3GP recording's tracks", async () => {
  const { REFERENCE_PICKER_ACCEPT, referenceFileRejection } = await import(
    "../src/features/video/reference-budget.ts"
  );
  // Offered by the dialog, then settled from the container once it is in hand.
  assert.ok(REFERENCE_PICKER_ACCEPT.audio.split(",").includes(".3gp"));

  const recording = new File([threeGpWithTracks(["soun"])], "voice.3gp", {
    type: "",
  });
  const classifiedRecording = await classifiedAttachmentFile(recording);
  assert.equal(referenceFileRejection("audio", classifiedRecording), null);
  // And a real clip picked there is refused, rather than staged as audio.
  const clip = new File([threeGpWithTracks(["soun", "vide"])], "clip.3gp", {
    type: "",
  });
  assert.equal(
    referenceFileRejection("audio", await classifiedAttachmentFile(clip)),
    "Please choose an audio file",
  );
  // The video picker does not take the recording either, extension or not.
  assert.equal(isVideoFile(classifiedRecording), false);
  assert.equal(
    referenceFileRejection("video", classifiedRecording),
    "Please choose a video file",
  );

  const picker = readFileSync(
    new URL("../src/features/video/reference-picker.tsx", import.meta.url),
    "utf8",
  );
  assert.match(picker, /await classifiedAttachmentFile\(picked\)/);
});

test("only a real charset parameter is a charset declaration", async () => {
  // "name=" can carry a filename that reads like one. Taking the first match
  // anywhere in the header stopped at the filename and never saw the parameter.
  const enc = new TextEncoder();
  const named = new Uint8Array([
    ...enc.encode(
      'Content-Type: text/plain; name="charset=windows-1251.txt";' +
        " charset=windows-1252\r\n\r\nCaf",
    ),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([named], "message.eml")),
    /Café/,
  );
  // A filename naming a supported label would otherwise pick the wrong decoder
  // silently, which is the worse half of the same bug.
  const plausible = new Uint8Array([
    ...enc.encode(
      'Content-Type: text/plain; name="charset=windows-1251";' +
        " charset=windows-1252\r\n\r\nCaf",
    ),
    0xe9,
  ]);
  assert.match(
    await readTextAttachment(new File([plausible], "message.eml")),
    /Café/,
  );
  // An ordinary header is unaffected, quoted or not.
  for (const header of [
    "Content-Type: text/plain; charset=windows-1252",
    'Content-Type: text/plain; charset="windows-1252"',
    "Content-Type: text/plain;charset=windows-1252",
    "Content-Type: text/plain; CHARSET = windows-1252",
  ]) {
    const eml = new Uint8Array([
      ...enc.encode(`${header}\r\n\r\nCaf`),
      0xe9,
    ]);
    assert.match(
      await readTextAttachment(new File([eml], "message.eml")),
      /Café/,
      header,
    );
  }
});

test("a truncated vCard preview reads per property too", async () => {
  // The preview pane decodes a prefix. Applying the sole declaration to all of
  // it showed CafÃ© for text the sent attachment renders as Café.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nEND:VCARD\r\nBEGIN:VCARD\r\nVERSION:3.0\r\nFN:Stra"),
    0xc3,
    0x9f,
    ...enc.encode("e\r\nEND:VCARD\r\n"),
  ]);
  const preview = decodeTextAttachmentBytes(vcf, "export.vcf", true);
  assert.match(preview, /FN;CHARSET=windows-1252:Café/);
  assert.match(preview, /FN:Straße/);
  assert.equal(preview, await readTextAttachment(new File([vcf], "export.vcf")));

  // A prefix cut through a character drops it rather than failing the read.
  const cut = vcf.subarray(0, vcf.length - 12);
  const cutPreview = decodeTextAttachmentBytes(cut, "export.vcf", true);
  assert.match(cutPreview, /FN;CHARSET=windows-1252:Café/);
});

test("a 3GP is inspected however large the surface taking it allows", async () => {
  // The ceiling here was the composer's video cap, and the video reference
  // surface accepts a larger file, so a recording in between was staged as a
  // video reference on its extension alone.
  const { MAX_REFERENCE_BYTES: limits } = await import(
    "../src/features/video/reference-budget.ts"
  );
  const recording = threeGpWithTracks(["soun"]);
  const oversize = new File([recording], "voice.3gp", { type: "" });
  Object.defineProperty(oversize, "size", {
    value: Math.max(limits.video, limits.audio, MAX_TEXT_ATTACHMENT_BYTES) + 1,
  });
  const classified = await classifiedAttachmentFile(oversize);
  assert.equal(classified.type, "audio/3gpp");
  assert.equal(isVideoFile(classified), false);

  // Nothing in the predicate turns on size any more, so no surface's limit can
  // drift past it again.
  const source = readFileSync(
    new URL("../src/lib/video-utils.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /export function needsAttachmentTrackInspection[^)]*\)[^{]*\{\s*return \/\\\.3gp\$\/i\.test\(file\.name\);\s*\}/,
  );
});

test("a boundary counts only on the header that owns the parts", async () => {
  // A quoted filename on any header could register a delimiter, and a body line
  // repeating it then reopened the headers.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode(
      "Content-Type: text/plain; charset=ISO-8859-1\r\n" +
        'Content-Disposition: attachment; filename="report; boundary=fake"\r\n\r\nCaf',
    ),
    0xe9,
    ...enc.encode(
      "\r\n--fake\r\nContent-Type: text/plain; charset=windows-1251\r\n",
    ),
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Café/);

  // A real multipart boundary still registers, quoted or bare.
  for (const header of [
    'Content-Type: multipart/mixed; boundary="part"',
    "Content-Type: multipart/alternative; boundary=part",
  ]) {
    const multipart = new Uint8Array([
      ...enc.encode(
        `${header}\r\n\r\n--part\r\nContent-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf`,
      ),
      0xe9,
    ]);
    assert.match(
      await readTextAttachment(new File([multipart], "message.eml")),
      /Café/,
      header,
    );
  }
});

test("a preview finds a declaration that sits past its own slice", async () => {
  // The preview decodes a bounded prefix. Looking for the declaration inside
  // that prefix reported an error for a file the attachment itself decodes.
  const enc = new TextEncoder();
  // A translator comment in the file's own encoding, so the prefix is not UTF-8
  // and the strict decode has to fall back to the declaration to read it.
  const comment = new Uint8Array([
    ...enc.encode("# "),
    0xcf,
    0xf0,
    ...enc.encode(" translated by\n"),
  ]);
  const whole = new Uint8Array([
    ...Array.from({ length: 200 }).flatMap(() => Array.from(comment)),
    ...enc.encode(
      `msgid ""\nmsgstr ""\n` +
        '"Content-Type: text/plain; charset=windows-1251\\n"\n\n' +
        'msgid "hi"\nmsgstr "',
    ),
    0xcf,
    0xf0,
    ...enc.encode('"\n'),
  ]);
  // The slice stops above the header entry, exactly as the preview cap would.
  const slice = whole.subarray(0, comment.length * 200 - 40);
  assert.throws(() => decodeTextAttachmentBytes(slice, "ru.po", true));
  assert.equal(
    typeof decodeTextAttachmentBytes(slice, "ru.po", true, whole),
    "string",
  );

  const source = readFileSync(
    new URL("../src/features/chat/attachment-content.ts", import.meta.url),
    "utf8",
  );
  assert.match(source, /decodeTextAttachmentBytes\(bytes, file\.name, truncated, whole\)/);
  assert.match(source, /DECLARES_ITS_CHARSET_RE/);
});

test("a 3GP typed as audio by the platform is still inspected", async () => {
  // A platform that maps the shared extension to audio/3gpp says so for a clip
  // too, and the audio adapter is matched before the video one.
  const clip = new File([threeGpWithTracks(["soun", "vide"])], "clip.3gp", {
    type: "audio/3gpp",
  });
  const classified = await classifiedAttachmentFile(clip);
  assert.equal(classified.type, "video/3gpp");
  assert.equal(isVideoFile(classified), true);
  assert.equal(isAudioAttachmentFile(classified), false);

  // A recording already typed correctly is returned untouched, not rewrapped.
  const recording = new File([threeGpWithTracks(["soun"])], "voice.3gp", {
    type: "audio/3gpp",
  });
  assert.equal(await classifiedAttachmentFile(recording), recording);

  // Tracks that cannot be read decide nothing.
  const unreadable = new File([new Uint8Array([1, 2, 3, 4])], "odd.3gp", {
    type: "audio/3gpp",
  });
  assert.equal(await classifiedAttachmentFile(unreadable), unreadable);
});

test("the reference drop zone takes what its dialog offers", async () => {
  const { REFERENCE_DROP_ACCEPT, REFERENCE_PICKER_ACCEPT: picker } = await import(
    "../src/features/video/reference-budget.ts"
  );
  // The zone filters on the name before the classifier can look at the file, so
  // a list narrower than the dialog's refused what the button accepts.
  assert.ok(REFERENCE_DROP_ACCEPT.audio.split(",").includes(".3gp"));
  for (const [kind, offered] of Object.entries(picker)) {
    const dropped = REFERENCE_DROP_ACCEPT[kind as "audio" | "video"].split(",");
    for (const entry of offered.split(",")) {
      if (!entry.startsWith(".")) continue;
      assert.ok(dropped.includes(entry), `${kind} ${entry}`);
    }
    // Extensions only: the zone shows this list verbatim when it refuses a file.
    assert.equal(dropped.some((entry) => entry.includes("/")), false, kind);
  }
  // Nothing the chat drop lists may be missing from it either.
  for (const ext of CHAT_AUDIO_DROP_ACCEPT.split(",")) {
    assert.ok(REFERENCE_DROP_ACCEPT.audio.split(",").includes(ext), ext);
  }
  for (const ext of CHAT_VIDEO_DROP_ACCEPT.split(",")) {
    assert.ok(REFERENCE_DROP_ACCEPT.video.split(",").includes(ext), ext);
  }

  const picked = readFileSync(
    new URL("../src/features/video/reference-picker.tsx", import.meta.url),
    "utf8",
  );
  assert.match(picked, /accept: REFERENCE_DROP_ACCEPT\[kind\]/);
});

test("a quoted parameter value resolves its escapes", async () => {
  // The splitter honours "\\X" already; the unquoting did not, so a value
  // carrying a quote stopped at the escaped one and kept the backslashes.
  const enc = new TextEncoder();
  const eml = new Uint8Array([
    ...enc.encode(
      'Content-Type: multipart/mixed; boundary="part\\"one"\r\n\r\n--part"one\r\n' +
        "Content-Type: text/plain; charset=ISO-8859-1\r\n\r\nCaf",
    ),
    0xe9,
  ]);
  assert.match(await readTextAttachment(new File([eml], "message.eml")), /Café/);

  // And a parameter after the escaped quote is still found, rather than being
  // swallowed by a value that never terminated.
  const named = new Uint8Array([
    ...enc.encode(
      'Content-Type: text/plain; name="say \\"hi\\""; charset=windows-1252\r\n\r\nCaf',
    ),
    0xe9,
  ]);
  assert.match(await readTextAttachment(new File([named], "message.eml")), /Café/);
});

test("the clipboard reader takes what the native side hands it", () => {
  // Rust reads a pasted clip to MAX_CLIPBOARD_VIDEO_BYTES; refusing it here
  // threw away a file already read and encoded, and dropped the paste with it.
  const source = readFileSync(
    new URL("../src/features/chat/utils/clipboard-files.ts", import.meta.url),
    "utf8",
  );
  const rust = readFileSync(
    new URL("../../src-tauri/src/native_clipboard.rs", import.meta.url),
    "utf8",
  );
  const rustLimit = (name: string): number => {
    const raw = rust.match(
      new RegExp(`const ${name}: u64 = ([0-9_]+(?:\\\\s*\\\\*\\\\s*[0-9_]+)*)`),
    )?.[1];
    assert.ok(raw, name);
    return raw
      .split("*")
      .reduce((total, part) => total * Number(part.replace(/[\s_]/g, "")), 1);
  };
  const frontLimit = (name: string): number => {
    const raw = source.match(
      new RegExp(`const ${name} = ([0-9_]+(?:\\\\s*\\\\*\\\\s*[0-9_]+)*)`),
    )?.[1];
    assert.ok(raw, name);
    return raw
      .split("*")
      .reduce((total, part) => total * Number(part.replace(/[\s_]/g, "")), 1);
  };

  assert.equal(
    frontLimit("MAX_CLIPBOARD_VIDEO_BYTES"),
    rustLimit("MAX_CLIPBOARD_VIDEO_BYTES"),
  );
  assert.equal(
    frontLimit("MAX_CLIPBOARD_NON_AUDIO_BYTES"),
    rustLimit("MAX_CLIPBOARD_SOURCE_BYTES"),
  );
  // The total may not refuse a single file the per-file limits allow.
  assert.match(source, /const MAX_CLIPBOARD_BYTES = MAX_CLIPBOARD_VIDEO_BYTES;/);
  assert.match(rust, /const MAX_CLIPBOARD_TOTAL_BYTES: u64 = MAX_CLIPBOARD_VIDEO_BYTES;/);
  // And video is classified before the catch-all, as it is natively.
  assert.match(
    source,
    /file\.mimeType\.startsWith\("video\/"\)\s*\?\s*MAX_CLIPBOARD_VIDEO_BYTES/,
  );
});

test("an unsupported charset is refused however many others sit beside it", async () => {
  // The per-property read cannot honour UTF-7, and the fallback then read the
  // card as UTF-8 and accepted it with the escapes literal, because every other
  // byte in it happened to be ASCII. The same declaration alone was refused, so
  // one card's fate turned on how many of its neighbours declared anything.
  const card = (extra: string) =>
    new File(
      [
        `BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=UTF-7:Bj+APg-rk\r\n${extra}END:VCARD\r\n`,
      ],
      "contact.vcf",
    );
  const message = /declares charset "UTF-7", which this browser has no decoder for/;
  await assert.rejects(readTextAttachment(card("")), (error: Error) => {
    assert.match(error.message, message);
    return true;
  });
  await assert.rejects(
    readTextAttachment(card("NOTE;CHARSET=windows-1252:plain ascii\r\n")),
    (error: Error) => {
      assert.match(error.message, message);
      return true;
    },
  );
});

test("a property declaring two charsets is refused rather than read as UTF-8", async () => {
  const vcf = new File(
    [
      "BEGIN:VCARD\r\nVERSION:2.1\r\n" +
        "FN;CHARSET=windows-1252;CHARSET=windows-1251:ascii\r\n" +
        "NOTE;CHARSET=utf-8:ascii\r\nEND:VCARD\r\n",
    ],
    "contact.vcf",
  );
  await assert.rejects(readTextAttachment(vcf), (error: Error) => {
    assert.match(
      error.message,
      /One of its properties declares two charsets \(windows-1252, windows-1251\)/,
    );
    return true;
  });
});

test("a card whose undeclared property is legacy still reports its charsets", async () => {
  // Nothing about the declarations blocks this reading: it is an undeclared
  // property holding bytes that are not UTF-8. That case keeps the answer it
  // had, so the change only reaches files a declaration itself defeats.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode(
      "BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:ok\r\nORG;CHARSET=windows-1251:ok\r\nNOTE:Caf",
    ),
    0xe9,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  await assert.rejects(
    readTextAttachment(new File([vcf], "contact.vcf")),
    (error: Error) => {
      assert.match(error.message, /windows-1252, windows-1251/);
      return true;
    },
  );
});

test("a card every property of which reads still decodes per property", async () => {
  // The guard above must not refuse cards the per-property read handles.
  const enc = new TextEncoder();
  const vcf = new Uint8Array([
    ...enc.encode("BEGIN:VCARD\r\nVERSION:2.1\r\nFN;CHARSET=windows-1252:Caf"),
    0xe9,
    ...enc.encode("\r\nORG;CHARSET=windows-1251:"),
    0xcf,
    0xf0,
    ...enc.encode("\r\nEND:VCARD\r\n"),
  ]);
  const text = await readTextAttachment(new File([vcf], "contact.vcf"));
  assert.match(text, /FN;CHARSET=windows-1252:Café/);
  assert.match(text, /ORG;CHARSET=windows-1251:Пр/);
});

test("a processing instruction is not read as the document's declaration", async () => {
  // `xml-stylesheet` and `xml-model` open with the same five bytes as a
  // declaration. Taking one for the prolog decoded a UTF-8 document as a code
  // page and returned mojibake, which no error accompanies.
  const doc =
    '<?xml-stylesheet type="text/xsl" href="s.xsl" encoding="windows-1252"?>\n' +
    "<note>café naïve</note>\n";
  const text = await readTextAttachment(new File([doc], "doc.xml"));
  assert.match(text, /café naïve/);
});

test("a declaration still needs only the whitespace its grammar requires", async () => {
  // The check is a sixth byte of XML's own S, so tab and newline pass with space.
  const enc = new TextEncoder();
  for (const space of [" ", "\t", "\n", "\r"]) {
    const xml = new Uint8Array([
      ...enc.encode(`<?xml${space}version="1.0" encoding="windows-1252"?><t>Caf`),
      0xe9,
      ...enc.encode("</t>"),
    ]);
    assert.match(
      await readTextAttachment(new File([xml], "ui.resx")),
      /Café/,
      `whitespace ${JSON.stringify(space)}`,
    );
  }
});

test("a gettext charset split across two literals reads as one label", async () => {
  // PO concatenates adjacent quoted pieces, so the header is not the source
  // text. Matching the raw entry read `windows-` and refused the catalog for a
  // charset it does not declare.
  const enc = new TextEncoder();
  const po = new Uint8Array([
    ...enc.encode(
      'msgid ""\nmsgstr ""\n' +
        '"Project-Id-Version: demo\\n"\n' +
        '"Content-Type: text/plain; charset=windows-"\n' +
        '"1252\\n"\n\n' +
        'msgid "cafe"\nmsgstr "caf',
    ),
    0xe9,
    ...enc.encode('"\n'),
  ]);
  assert.match(await readTextAttachment(new File([po], "messages.po")), /café/);
});

test("a header entry the prefix cuts is answered from the whole file", async () => {
  // The same truncation by a different route: the 64 KiB prefix ends inside the
  // charset, and the last complete piece stops at `windows-`. An entry that runs
  // to the cut has to be re-read rather than answered from half a value.
  const enc = new TextEncoder();
  const head = 'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=windows-';
  const comment = "# " + "c".repeat(64 * 1024 - head.length - 4) + "\n";
  const source = comment + head;
  // The cut lands inside the value, with the rest of it in the next piece.
  assert.ok(source.length > 64 * 1024 - 8 && source.length <= 64 * 1024);
  const po = new Uint8Array([
    ...enc.encode(`${source}"\n"1252\\n"\n\nmsgid "c"\nmsgstr "caf`),
    0xe9,
    ...enc.encode('"\n'),
  ]);
  assert.match(await readTextAttachment(new File([po], "big.po")), /café/);
});

test("escapes in a header entry resolve before the charset is read", async () => {
  // `\n` is a newline, not the letter: emitting it literally glued the charset
  // to the header that follows and made `utf-8` read as `utf-8nContent`.
  const po =
    'msgid ""\nmsgstr ""\n' +
    '"Content-Type: text/plain; charset=utf-8\\nContent-Transfer-Encoding: 8bit\\n"\n\n' +
    'msgid "c"\nmsgstr "café"\n';
  assert.match(await readTextAttachment(new File([po], "m.po")), /café/);
});
