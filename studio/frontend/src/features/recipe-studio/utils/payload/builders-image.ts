// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ImageConfig } from "../../types";

// Prompt and preset are checked per node (getConfigErrors), so nothing is repeated here.
export function buildImageColumn(config: ImageConfig): Record<string, unknown> {
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "image",
    name: config.name,
    drop: config.drop ?? false,
    prompt: config.prompt,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: config.model_alias,
  };
}
