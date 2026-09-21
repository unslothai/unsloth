// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

// Only llama-server knows whether a GGUF takes video, so the flag travels from
// /props to the model row and the adapter reads it off that row. Every hop that
// carries the audio flag has to carry this one too: a hop that drops it leaves
// the adapter reading false and refusing video on a model that supports it.
// Two separate hops (syncModelCapabilities, then the direct status adoption)
// were each missing it, hence a rule rather than two spot checks.

const SRC = new URL("../src/", import.meta.url);

async function sourceFiles(dir: URL): Promise<URL[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const found: URL[] = [];
  for (const entry of entries) {
    if (entry.name === "node_modules") continue;
    if (entry.isDirectory()) {
      found.push(...(await sourceFiles(new URL(`${entry.name}/`, dir))));
    } else if (/\.tsx?$/.test(entry.name)) {
      found.push(new URL(entry.name, dir));
    }
  }
  return found;
}

const rel = (file: URL) =>
  path.relative(new URL(".", SRC).pathname, file.pathname);

test("every mapper that writes hasAudioInput writes hasVideoInput too", async () => {
  const files = await sourceFiles(SRC);
  const dropped: string[] = [];
  for (const file of files) {
    const source = await readFile(file, "utf8");
    // The write, not the read: `hasAudioInput:` assigns, `.hasAudioInput` reads.
    if (!/\bhasAudioInput\s*:/.test(source)) continue;
    // The runtime type declares both as optional fields, not a mapping.
    if (rel(file) === "features/chat/types/runtime.ts") continue;
    if (!/\bhasVideoInput\s*:/.test(source)) dropped.push(rel(file));
  }
  assert.deepEqual(dropped, []);
});

test("the direct status adoption carries the video capability", async () => {
  const source = await readFile(
    new URL("features/chat/lib/apply-inference-status-to-store.ts", SRC),
    "utf8",
  );
  // This path never calls syncModelCapabilities, so whatever it omits here is
  // simply absent from the row a server-adopted GGUF gets.
  const caps = source.slice(
    source.indexOf("function ensureActiveModelInStoreList"),
    source.indexOf("const existing = store.models.find"),
  );
  assert.match(caps, /hasAudioInput:\s*status\.has_audio_input/);
  assert.match(caps, /hasVideoInput:\s*status\.has_video_input/);
});

test("the video drain names video when a clip cannot be read", async () => {
  const source = await readFile(
    new URL("components/assistant-ui/thread.tsx", SRC),
    "utf8",
  );
  // Cloned from the audio drain, so the toast title came along with it. This is
  // the one path whose job is to explain why a dropped video did not attach.
  const drain = source.slice(source.indexOf("claimVideoAttachments"));
  const title = drain.match(/toast\.error\("Could not attach dropped (\w+)"/)?.[1];
  assert.equal(title, "video");
});
