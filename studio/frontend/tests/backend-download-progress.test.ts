// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as motion from "motion/react";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import { parseBackendDownloadProgress } from "../src/components/tauri/backend-download-progress.ts";
import type * as UpdateScreenModule from "../src/components/tauri/update-screen.tsx";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { UpdateScreen } = loadWithStubs<typeof UpdateScreenModule>(
  new URL("../src/components/tauri/update-screen.tsx", import.meta.url),
  {
    "react/jsx-runtime": jsxRuntime,
    "motion/react": motion,
    "@/components/tauri/backend-download-progress": {
      parseBackendDownloadProgress,
    },
    "@/components/tauri/diagnostics-copy-actions": {
      DiagnosticsCopyActions: () => null,
    },
    "@/components/tauri/log-details": { LogDetails: () => null },
    "@/components/ui/button": { Button: () => null },
    "@/components/ui/spinner": { Spinner: () => null },
  },
);

function render(
  status: React.ComponentProps<typeof UpdateScreen>["status"],
  logs: string[],
) {
  return renderToStaticMarkup(
    React.createElement(UpdateScreen, {
      status,
      logs,
      progress: 37,
      error: null,
      onRetry() {},
      onSkipRestart() {},
      onCopyDiagnostics: async () => ({
        ok: true,
        report: "",
        source: "tauri" as const,
      }),
    }),
  );
}

test("reads the downloader's percentage, bytes and speed", () => {
  assert.deepEqual(
    parseBackendDownloadProgress(
      "Downloading llama.cpp-source.tar.gz:  77.6% (28.0 MiB/36.1 MiB) at 118.4 KiB/s",
    ),
    {
      file: "llama.cpp-source.tar.gz",
      detail: "77.6% (28.0 MiB/36.1 MiB) at 118.4 KiB/s",
      percent: 77.6,
    },
  );
});

test("unknown-length downloads show bytes without inventing a percentage", () => {
  assert.deepEqual(
    parseBackendDownloadProgress(
      "Downloading whisper.tar.gz: 25.0 MiB downloaded at 2.0 MiB/s",
    ),
    {
      file: "whisper.tar.gz",
      detail: "25.0 MiB downloaded at 2.0 MiB/s",
      percent: null,
    },
  );
});

test("completion is per file and a later step clears download progress", () => {
  const lines = [
    "Downloading node.tar.gz: 100.0% (30.0 MiB/30.0 MiB) at 2.0 MiB/s",
    "   node           v24.18.0",
    "Downloading llama.tar.gz:  25.0% (10.0 MiB/40.0 MiB) at 2.0 MiB/s",
    "   llama.cpp      prebuilt installed and validated",
  ];
  assert.deepEqual(
    lines.map((line) => parseBackendDownloadProgress(line)?.percent ?? null),
    [100, null, 25, null],
  );
});

test("unrelated output and malformed percentages cannot become progress", () => {
  for (const line of [
    "",
    "existing install detected -- validating update",
    "download failed (1/3)",
    "Downloading llama.tar.gz: preparing",
    "Downloading llama.tar.gz: 101% (41 MiB/40 MiB) at 1 MiB/s",
    "Downloading llama.tar.gz: -1% (0 MiB/40 MiB) at 1 MiB/s",
  ]) {
    assert.equal(parseBackendDownloadProgress(line), null, line);
  }
});

test("the update screen shows live backend progress without opening details", () => {
  const html = render("updating-backend", [
    "Downloading llama.tar.gz: 50.0% (20.0 MiB/40.0 MiB) at 2.0 MiB/s",
  ]);
  assert.match(html, /Downloading llama.tar.gz/);
  assert.match(html, /20.0 MiB\/40.0 MiB/);
  assert.match(html, /<progress[^>]*value="50"/);
});

test("later steps clear the backend bar and shell downloads use their own percentage", () => {
  const logs = [
    "Downloading llama.tar.gz: 100.0% (40.0 MiB/40.0 MiB) at 2.0 MiB/s",
  ];
  assert.doesNotMatch(
    render("updating-backend", [...logs, "prebuilt installed"]),
    /<progress/,
  );
  assert.doesNotMatch(render("error", logs), /<progress/);
  assert.doesNotMatch(render("installing", logs), /<progress/);
  assert.match(render("downloading", logs), /<progress[^>]*value="37"/);
  assert.doesNotMatch(render("downloading", logs), /Downloading llama/);
});

test("unknown-size downloads show their bytes and speed without a bar", () => {
  const html = render("updating-backend", [
    "Downloading whisper.tar.gz: 25.0 MiB downloaded at 2.0 MiB/s",
  ]);
  assert.match(html, /25.0 MiB downloaded at 2.0 MiB\/s/);
  assert.doesNotMatch(html, /<progress/);
});
