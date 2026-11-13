// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ContainmentMode,
  defineContainmentFlag,
} from "./containment-flag.ts";

export type ReasoningRowContainmentMode = ContainmentMode;

export const REASONING_ROW_CONTAINMENT = defineContainmentFlag({
  attribute: "data-reasoning-row-containment",
  global: "__UNSLOTH_REASONING_ROW_CONTAINMENT__",
});

export const REASONING_ROW_CONTAINMENT_ATTRIBUTE =
  REASONING_ROW_CONTAINMENT.attribute;
export const REASONING_ROW_CONTAINMENT_ON = REASONING_ROW_CONTAINMENT.on;
export const SHIP_DEFAULT = REASONING_ROW_CONTAINMENT.shipDefault;
export const ROW_SETTLED_ATTRIBUTE = "data-settled";
export const CHUNK_ESTIMATE_PROPERTY = "--unsloth-chunk-estimate";
