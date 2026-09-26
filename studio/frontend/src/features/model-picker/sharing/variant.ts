// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelPickTarget } from "../components/model-selector/types";

export function isRunConfigVariantUnresolved(target: ModelPickTarget): boolean {
  return (
    target.meta.source === "hub" &&
    target.isGguf &&
    !target.meta.isDownloaded &&
    (!target.ggufVariant || target.ggufVariant.toLowerCase().endsWith(".gguf"))
  );
}
