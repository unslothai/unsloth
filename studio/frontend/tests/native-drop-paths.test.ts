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

const BACKEND_UPLOAD_EXTS_RE = /UPLOAD_EXTS\s*=\s*\{([^}]+)\}/s;
const RUST_ATTACHMENT_EXTS_RE = /ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const RUST_IMAGE_ATTACHMENT_EXTS_RE = /IMAGE_ATTACHMENT_EXTS[^=]*=\s*&\[([^\]]+)\]/s;
const DOTTED_EXTENSION_RE = /"(\.[^"]+)"/g;
const RUST_EXTENSION_RE = /"([^"]+)"/g;
const RUST_MIME_ARM_RE = /Some\("(image\/[^"]+)"\)/g;
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

// #7963 was a list reused across two features drifting apart. The drop path has
// one more seam like it: Rust stamps the File's MIME type, and the composer
// routes to an adapter by MIME, so a type Rust emits that VisionImageAdapter
// doesn't claim would land the image on the wrong adapter or nowhere.
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
