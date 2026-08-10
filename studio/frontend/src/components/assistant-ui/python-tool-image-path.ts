// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sandboxFilePath } from "./sandbox-files.ts";

export function pythonToolImagePath(
  sessionId: string,
  filename: string,
): string {
  // Segment by segment, like the file cards: a chart written to outputs/ keeps
  // a real "/" in the URL, and an encoded one is refused by proxies before the
  // route ever sees it.
  return sandboxFilePath(sessionId, filename);
}
