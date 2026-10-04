// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import { mergeConfig } from "vite";
import base from "./vite.config";

export default mergeConfig(base, {
  build: {
    emptyOutDir: true,
    outDir: path.resolve(
      __dirname,
      "../../.playwright-cli/reasoning-production",
    ),
    rollupOptions: {
      input: path.resolve(__dirname, "smoke-reasoning-transcript.html"),
    },
  },
});
