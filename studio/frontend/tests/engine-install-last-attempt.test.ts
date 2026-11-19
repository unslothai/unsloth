// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { type ViteDevServer, createServer } from "vite";
import { readSrc } from "./helpers/kit.ts";

// #18492: a retry that fails with the same message has to read as a new attempt.
let vite: ViteDevServer;
let EngineInstall: (props: Record<string, unknown>) => unknown;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  ({ EngineInstall } = await vite.ssrLoadModule(
    "/src/features/model-picker/components/inference-engines.tsx",
  ));
});

after(async () => {
  await vite.close();
});

const UV_MISSING = "uv is unavailable. Repair the Unsloth installation and retry.";

const render = (job: Record<string, unknown>) =>
  renderToStaticMarkup(
    createElement(EngineInstall as never, {
      management: true,
      engine: {
        engine: "vllm",
        version: "0.30.0",
        installed_version: null,
        installed: false,
        in_use: false,
        current: false,
        can_rollback: false,
        unsupported_reason: null,
        job: { phase: null, ...job },
      },
    }),
  );

test("a failed install says when its last attempt ended", () => {
  const html = render({
    state: "error",
    message: UV_MISSING,
    finished_at: Date.UTC(2026, 10, 19, 16, 32, 5) / 1000,
  });
  assert.match(html, /role="alert"[^>]*>Engine installation failed/);
  assert.match(html, /Last attempt: [^<]*\d{1,2}:\d{2}:\d{2}/);
  assert.match(html, /uv is unavailable/);
});

test("two failures with the same message render different attempt times", () => {
  const first = render({ state: "error", message: UV_MISSING, finished_at: 1_800_000_000 });
  const second = render({ state: "error", message: UV_MISSING, finished_at: 1_800_000_007 });
  assert.notEqual(first, second);
});

test("an older backend without finished_at still shows the failure, without a time", () => {
  const html = render({ state: "error", message: UV_MISSING });
  assert.match(html, /Engine installation failed/);
  assert.doesNotMatch(html, /Last attempt/);
});

test("a running retry shows progress, not the previous failure", () => {
  const html = render({ state: "running", message: "Preparing installation", finished_at: null });
  assert.doesNotMatch(html, /Engine installation failed|Last attempt|uv is unavailable/);
});

test("opening the install prompt for a retry hides the previous failure", () => {
  const source = readSrc("features/model-picker/components/inference-engines.tsx");
  assert.match(source, /const failed = engine\.job\.state === "error" && !confirm && !busy;/);
  assert.match(source, /\{\(error \|\| failed\) && \(/);
  assert.match(source, /\{failed && \(\s*<details>/);
});
