// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { IconSvgElement } from "@hugeicons/react";
import type { ComponentType } from "react";
import { COLUMN_TYPE } from "../recipe/recipe-columns";
import { RECIPE_COLUMN_LOOK } from "../recipe/recipe-node-look";
import type { FlowNode, NodeSpec } from "../types";

/** Per-type content for the shared node card. Anything left out falls back to the
 *  spec: category icon and tone, typed ports, key fields and "Show inputs". */
export interface NodeLook {
  icon?: (node: FlowNode) => IconSvgElement;
  /** Icon tile classes, e.g. one of Recipe Studio's `RECIPE_STUDIO_NODE_TONES`. */
  tone?: (node: FlowNode) => string | undefined;
  /** Hex accent for the tile when there is no tone class. */
  accent?: (node: FlowNode) => string | undefined;
  subtitle?: (node: FlowNode, spec: NodeSpec) => string;
  HeaderExtra?: ComponentType<{ node: FlowNode }>;
  /** Replaces the typed port rows; `collapsed` means draw them on the header. */
  Ports?: ComponentType<{ node: FlowNode; collapsed: boolean }>;
  /** Replaces the key fields and "Show inputs". */
  Body?: ComponentType<{ node: FlowNode }>;
}

export const NODE_LOOKS: Record<string, NodeLook> = {
  [COLUMN_TYPE]: RECIPE_COLUMN_LOOK,
};
