// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ImageConfig } from "../../../types";
import { readString } from "../helpers";

export function parseImage(
  column: Record<string, unknown>,
  name: string,
  id: string,
): ImageConfig {
  return {
    id,
    kind: "image",
    name,
    drop: column.drop === true,
    prompt: readString(column.prompt) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: readString(column.model_alias) ?? "",
  };
}
