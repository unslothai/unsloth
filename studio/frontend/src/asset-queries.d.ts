// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Vite `?inline` imports (data URIs); vite/client only types bare extensions.
declare module "*?inline" {
  const src: string;
  // biome-ignore lint/style/noDefaultExport: Vite asset modules export default.
  export default src;
}
