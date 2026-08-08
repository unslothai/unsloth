// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  dequeueNativeAttachments,
  enqueueNativeAttachments,
} from "../src/features/native-intents/attachment-queue.ts";
import { classifyDropPaths, CHAT_IMAGE_DROP_ACCEPT, SUPPORTED_DROP_HINT } from "../src/features/native-intents/drop-paths.ts";
import type { NativeIntent } from "../src/features/native-intents/types.ts";
import { RAG_UPLOAD_ACCEPT } from "../src/features/rag/types/rag.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useNativeIntentStore } = await import(
  "../src/features/native-intents/store.ts"
);

const BACKEND_UPLOAD_EXTS_RE = /UPLOAD_EXTS\s*=\s*\{([^}]+)\}/s;
const RUST_ATTACHMENT_EXTS_RE = /ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_IMAGE_ATTACHMENT_EXTS_RE = /IMAGE_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const DOTTED_EXTENSION_RE = /"(\.[^"]+)"/g;
const RUST_EXTENSION_RE = /"([^"]+)"/g;
const RUST_MIME_ARM_RE = /Some\("(image\/[^"]+)"\)/g;
const MIME_MATCH_BODY_RE = /fn attachment_mime_type[\s\S]*?match ext\.as_str\(\) \{([\s\S]*?)\n {4}\}/;
const MIME_ARM_EXTENSION_RE = /^\s*((?:"[^"]+"\s*\|?\s*)+)=>\s*Some\("image\//gm;
const COMPOSER_IMAGE_ACCEPT_RE = /const IMAGE_ACCEPT\s*=\s*"([^"]+)"/;
const VISION_ADAPTER_ACCEPT_RE =
  /class VisionImageAdapter[^{]*\{\s*accept\s*=\s*"([^"]+)"/s;

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

  assert.ok(stamped.length > 0, "no image MIME arms found in native_intents.rs");
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
    .flatMap((match) => [...match[1].matchAll(/"([^"]+)"/g)].map((ext) => ext[1]))
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

// A fresh chat persisting remounts the composer, so the instance that queued
// the batch cannot hand it over itself.
test("a remounted composer claims the batch its predecessor left behind", () => {
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

  store.addImageAttachments("single:new", [intent]);
  store.noteImageDropOwner("single:new", "composer-1");

  // A different composer must not take it.
  store.claimImageAttachments("composer-2", "single:other");
  assert.equal(
    useNativeIntentStore.getState().pendingImageAttachments["single:new"]?.length,
    1,
  );

  store.claimImageAttachments("composer-1", "single:thread-7");
  const after = useNativeIntentStore.getState();
  assert.equal(after.pendingImageAttachments["single:new"], undefined);
  assert.deepEqual(after.pendingImageAttachments["single:thread-7"], [intent]);
  assert.deepEqual(after.imageDropOwners, {});
});

// The ordering the first claim missed: the predecessor's requeue can land after
// the replacement composer has already claimed once.
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
  assert.equal(owners["single:new"], "composer-3", "the note survives for a later claim");

  store.claimImageAttachments("composer-3", "single:thread-9");
  const after = useNativeIntentStore.getState();
  assert.equal(after.pendingImageAttachments["single:new"], undefined);
  assert.deepEqual(after.pendingImageAttachments["single:thread-9"], [intent]);
});

test("document and image drop extensions stay disjoint", () => {
  // classifyDropPaths sums the two filters, so an overlap silently turns a
  // perfectly good drop into "unsupported".
  const docs = RAG_UPLOAD_ACCEPT.split(",").map((ext) => ext.trim().toLowerCase());
  const images = CHAT_IMAGE_DROP_ACCEPT.split(",").map((ext) =>
    ext.trim().toLowerCase(),
  );
  assert.deepEqual(
    images.filter((ext) => docs.includes(ext)),
    [],
  );
});
