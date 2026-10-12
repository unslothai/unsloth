// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep in step with PreviewRuntimeManifest in vite-preview-runtime.ts.
declare module "virtual:preview-runtime-manifest" {
  export interface PreviewRuntimeManifest {
    files: Record<string, { url: string; bytes: number; version: string }>;
    modules: Record<string, string>;
    tailwind: string;
  }
  const manifest: PreviewRuntimeManifest;
  // biome-ignore lint/style/noDefaultExport: the plugin emits `export default <manifest>`.
  export default manifest;
}
