// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeSpec } from "../types";

export const CARD_MIN_WIDTH = 220;

/** Smallest card height that still shows the header and every port row. */
export function cardMinHeight(spec: Pick<NodeSpec, "inputs" | "outputs" | "flow_out">): number {
  const rows = Math.max(
    spec.inputs.length,
    spec.outputs.length + spec.flow_out.length,
  );
  return 56 + rows * 24 + 16;
}
