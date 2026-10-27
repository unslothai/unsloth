// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sandboxFilePath } from "./sandbox-files.ts";

export function pythonToolImagePath(
  sessionId: string,
  filename: string,
): string {
  // Encode per segment: an encoded "/" in the URL is refused by proxies.
  return sandboxFilePath(sessionId, filename);
}
