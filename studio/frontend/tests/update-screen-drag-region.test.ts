// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as motion from "motion/react";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import type * as UpdateScreenModule from "../src/components/tauri/update-screen.tsx";
import { loadWithStubs } from "./helpers/module-stubs.ts";

let nativeMacTitlebar = true;

const { UpdateScreen } = loadWithStubs<typeof UpdateScreenModule>(
  new URL("../src/components/tauri/update-screen.tsx", import.meta.url),
  {
    "react/jsx-runtime": jsxRuntime,
    "motion/react": motion,
    "@/components/tauri/backend-download-progress": {
      parseBackendDownloadProgress: () => null,
    },
    "@/components/tauri/diagnostics-copy-actions": {
      DiagnosticsCopyActions: () => null,
    },
    "@/components/tauri/log-details": { LogDetails: () => null },
    "@/components/tauri/window-titlebar": {
      shouldUseNativeMacWindowTitlebar: () => nativeMacTitlebar,
    },
    "@/components/ui/button": { Button: () => null },
    "@/components/ui/spinner": { Spinner: () => null },
  },
);

function render(status: React.ComponentProps<typeof UpdateScreen>["status"]) {
  return renderToStaticMarkup(
    React.createElement(UpdateScreen, {
      status,
      logs: [],
      progress: 0,
      error: status === "error" ? "boom" : null,
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

test("the mac update screen keeps a window drag region", () => {
  nativeMacTitlebar = true;
  for (const status of [
    "updating-backend",
    "downloading",
    "installing",
    "error",
  ] as const) {
    assert.match(render(status), /data-tauri-drag-region/, status);
  }
});

// A Radix modal's body lock (pointer-events: none) is inherited and would block dragging.
test("the mac update screen drag region opts back into pointer events", () => {
  nativeMacTitlebar = true;
  assert.match(
    render("updating-backend"),
    /<div data-tauri-drag-region="true"[^>]*class="pointer-events-auto /,
  );
});

test("the update screen adds no drag region beside the custom titlebar", () => {
  nativeMacTitlebar = false;
  assert.doesNotMatch(render("updating-backend"), /data-tauri-drag-region/);
});
