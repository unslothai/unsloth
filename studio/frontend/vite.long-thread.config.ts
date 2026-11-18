// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Production build of the thread-weight fixture for tests/studio/playwright_long_thread_gate.py:
// the shipped CSS plugins and production React, since a dev build inflates the costs it gates.
import path from "node:path";
import { mergeConfig } from "vite";
import base from "./vite.config";

export default mergeConfig(base, {
  build: {
    emptyOutDir: true,
    outDir: path.resolve(__dirname, "../../.playwright-cli/long-thread-production"),
    rollupOptions: {
      // The app itself too: the gate's area matrix opens the real chat page from this build.
      input: {
        thread: path.resolve(__dirname, "smoke-thread-weight.html"),
        app: path.resolve(__dirname, "index.html"),
      },
    },
  },
});
