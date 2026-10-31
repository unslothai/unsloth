// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import manifest from "virtual:preview-runtime-manifest";
import { createRuntimeLoader } from "./runtime-loader";

// Plain fetch, not apiUrl(): the files ship with the app's own bundle, which the desktop app embeds.
export const runtimeLoader = createRuntimeLoader(manifest, async (url) => {
  const response = await fetch(new URL(`${import.meta.env.BASE_URL}${url}`, window.location.href));
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
});
