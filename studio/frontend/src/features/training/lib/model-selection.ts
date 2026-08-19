// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "@/lib/local-path";

export function isLocalTrainingModelSelection({
  model,
  knownCached,
  localPath,
}: {
  model: string | null;
  knownCached: boolean;
  localPath: string | null;
}): boolean {
  return Boolean(
    model &&
      (looksLikeLocalPath(model) ||
        (!knownCached && localPath && localPath.trim())),
  );
}
