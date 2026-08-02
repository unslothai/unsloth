// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeProcessorConfig } from "../types";

export function buildDefaultSchemaTransform(): RecipeProcessorConfig {
  return {
    id: "schema-transform-1",
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "schema_transform",
    name: "schema_transform",
    template: '{\n  "text": "{{ column_name }}"\n}',
  };
}
