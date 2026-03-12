// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Node } from "@xyflow/react";

function parseDim(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function readNodeWidth(node: Node): number | null {
  return (
    parseDim(node.width) ??
    parseDim(node.style?.width) ??
    parseDim(node.measured?.width) ??
    null
  );
}

export function readNodeHeight(node: Node): number | null {
  return (
    parseDim(node.height) ??
    parseDim(node.style?.height) ??
    parseDim(node.measured?.height) ??
    null
  );
}

