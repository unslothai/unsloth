// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sandboxRoutePrefix } from "./sandbox-files";

export function pythonToolImagePath(
  sessionId: string,
  filename: string,
): string {
  const { prefix, query } = sandboxRoutePrefix(sessionId);
  return `${prefix}/${encodeURIComponent(filename)}${query}`;
}
