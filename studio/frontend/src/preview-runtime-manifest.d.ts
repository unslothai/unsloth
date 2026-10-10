// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Provided by the previewRuntime() plugin in vite-preview-runtime.ts; keep this in step with its
// PreviewRuntimeManifest type.
declare module "virtual:preview-runtime-manifest" {
  export interface PreviewRuntimeManifest {
    /** File name -> where it is served (relative to the app base, no leading slash), its exact UTF-8 size, and the package version it was built from. */
    files: Record<string, { url: string; bytes: number; version: string }>;
    /** Import specifier -> the file name that registers it. */
    modules: Record<string, string>;
    /** The file name of the Tailwind classic script. It is not importable. */
    tailwind: string;
  }
  const manifest: PreviewRuntimeManifest;
  // biome-ignore lint/style/noDefaultExport: the plugin emits `export default <manifest>`.
  export default manifest;
}
