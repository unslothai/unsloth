// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Connection } from "@xyflow/react";
import type { LayoutDirection } from "../types";

export const HANDLE_IDS = {
  // data flow lanes
  // Inputs enter from the top or left, outputs leave from the right or bottom.
  dataIn: "data-in",
  dataInTop: "data-in-top",
  dataOut: "data-out",
  dataOutBottom: "data-out-bottom",
  // semantic dependency lanes
  semanticIn: "semantic-in",
  semanticInTop: "semantic-in-top",
  semanticOut: "semantic-out",
  semanticOutBottom: "semantic-out-bottom",
  // llm prompt/scorer lanes
  llmInputOutRight: "llm-input-out-right",
  llmInputOutBottom: "llm-input-out-bottom",
} as const;

export type RecipeHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];

// Handles older recipes may still reference. Each maps to the legal handle on the same axis,
// so an LR recipe stays horizontal and a TB one stays vertical; the edge itself is kept.
export const LEGACY_HANDLE_ALIAS_MAP: Record<string, string> = {
  "semantic-in-left": HANDLE_IDS.semanticIn,
  "semantic-out-right": HANDLE_IDS.semanticOut,
  "data-in-right": HANDLE_IDS.dataIn,
  "data-in-bottom": HANDLE_IDS.dataInTop,
  "data-out-left": HANDLE_IDS.dataOut,
  "data-out-top": HANDLE_IDS.dataOutBottom,
  "semantic-in-right": HANDLE_IDS.semanticIn,
  "semantic-in-bottom": HANDLE_IDS.semanticInTop,
  "semantic-out-left": HANDLE_IDS.semanticOut,
  "semantic-out-top": HANDLE_IDS.semanticOutBottom,
  "llm-input-out-left": HANDLE_IDS.llmInputOutRight,
  "llm-input-out-top": HANDLE_IDS.llmInputOutBottom,
};

const DATA_TARGET_HANDLES = new Set<string>([
  HANDLE_IDS.dataIn,
  HANDLE_IDS.dataInTop,
]);

const DATA_SOURCE_HANDLES = new Set<string>([
  HANDLE_IDS.dataOut,
  HANDLE_IDS.dataOutBottom,
]);

const SEMANTIC_TARGET_HANDLES = new Set<string>([
  HANDLE_IDS.semanticIn,
  HANDLE_IDS.semanticInTop,
]);

const SEMANTIC_SOURCE_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOut,
  HANDLE_IDS.semanticOutBottom,
]);

const DATA_TARGET_HORIZONTAL_HANDLES = new Set<string>([HANDLE_IDS.dataIn]);

const DATA_TARGET_VERTICAL_HANDLES = new Set<string>([HANDLE_IDS.dataInTop]);

const DATA_SOURCE_HORIZONTAL_HANDLES = new Set<string>([HANDLE_IDS.dataOut]);

const DATA_SOURCE_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.dataOutBottom,
]);

const SEMANTIC_TARGET_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticIn,
]);

const SEMANTIC_TARGET_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticInTop,
]);

const SEMANTIC_SOURCE_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOut,
]);

const SEMANTIC_SOURCE_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOutBottom,
]);

export function normalizeRecipeHandleId(
  handleId: string | null | undefined,
): string | null {
  if (!handleId) {
    return null;
  }
  return LEGACY_HANDLE_ALIAS_MAP[handleId] ?? handleId;
}

export type RecipeHandleSide = "left" | "top" | "right" | "bottom";

/** Side a handle sits on. Targets are only ever left/top, sources right/bottom. */
export function getRecipeHandleSide(
  handleId: string | null | undefined,
  kind: "source" | "target",
): RecipeHandleSide {
  const normalized = normalizeRecipeHandleId(handleId) ?? "";
  if (kind === "target") {
    return normalized.endsWith("-top") ? "top" : "left";
  }
  return normalized.endsWith("-bottom") ? "bottom" : "right";
}

/** Rewrites removed handle ids on a stored edge to their legal replacement. */
export function migrateRecipeEdgeHandles<
  T extends { sourceHandle?: string | null; targetHandle?: string | null },
>(edge: T): T {
  const sourceHandle = normalizeRecipeHandleId(edge.sourceHandle);
  const targetHandle = normalizeRecipeHandleId(edge.targetHandle);
  if (
    sourceHandle === (edge.sourceHandle ?? null) &&
    targetHandle === (edge.targetHandle ?? null)
  ) {
    return edge;
  }
  return { ...edge, sourceHandle, targetHandle };
}

export function normalizeRecipeConnectionHandles(
  connection: Connection,
): Connection {
  return {
    ...connection,
    sourceHandle: normalizeRecipeHandleId(connection.sourceHandle),
    targetHandle: normalizeRecipeHandleId(connection.targetHandle),
  };
}

function isKnownHandle(
  handleId: string | null | undefined,
  handles: Set<string>,
): boolean {
  if (!handleId) {
    return false;
  }
  return handles.has(normalizeRecipeHandleId(handleId) ?? "");
}

function remapHandleForDirection(
  handleId: string | null | undefined,
  direction: LayoutDirection,
  horizontalHandles: Set<string>,
  verticalHandles: Set<string>,
  defaultHandle: string,
): string {
  const normalizedHandleId = normalizeRecipeHandleId(handleId);
  if (!normalizedHandleId) {
    return defaultHandle;
  }
  if (direction === "LR") {
    if (verticalHandles.has(normalizedHandleId)) {
      return defaultHandle;
    }
    return normalizedHandleId;
  }
  if (horizontalHandles.has(normalizedHandleId)) {
    return defaultHandle;
  }
  return normalizedHandleId;
}

export function isDataTargetHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, DATA_TARGET_HANDLES);
}

export function isDataSourceHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, DATA_SOURCE_HANDLES);
}

export function isSemanticTargetHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, SEMANTIC_TARGET_HANDLES);
}

export function isSemanticSourceHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, SEMANTIC_SOURCE_HANDLES);
}

export function getDefaultDataTargetHandle(direction: LayoutDirection): string {
  return direction === "TB" ? HANDLE_IDS.dataInTop : HANDLE_IDS.dataIn;
}

export function getDefaultDataSourceHandle(direction: LayoutDirection): string {
  return direction === "TB" ? HANDLE_IDS.dataOutBottom : HANDLE_IDS.dataOut;
}

export function getDefaultSemanticTargetHandle(
  direction: LayoutDirection,
): string {
  return direction === "TB" ? HANDLE_IDS.semanticInTop : HANDLE_IDS.semanticIn;
}

export function getDefaultSemanticSourceHandle(
  direction: LayoutDirection,
): string {
  return direction === "TB" ? HANDLE_IDS.semanticOutBottom : HANDLE_IDS.semanticOut;
}

type RecipeEdgeHandles = {
  sourceHandle?: string | null;
  targetHandle?: string | null;
  type?: string | null;
};

export function remapRecipeEdgeHandlesForLayout(
  edge: RecipeEdgeHandles,
  direction: LayoutDirection,
): { sourceHandle: string; targetHandle: string } {
  const semantic =
    edge.type === "semantic" ||
    (isSemanticSourceHandle(edge.sourceHandle) &&
      isSemanticTargetHandle(edge.targetHandle));
  if (semantic) {
    const sourceIsData = isDataSourceHandle(edge.sourceHandle);
    const targetIsData = isDataTargetHandle(edge.targetHandle);
    return {
      sourceHandle: remapHandleForDirection(
        edge.sourceHandle,
        direction,
        sourceIsData
          ? DATA_SOURCE_HORIZONTAL_HANDLES
          : SEMANTIC_SOURCE_HORIZONTAL_HANDLES,
        sourceIsData
          ? DATA_SOURCE_VERTICAL_HANDLES
          : SEMANTIC_SOURCE_VERTICAL_HANDLES,
        sourceIsData
          ? getDefaultDataSourceHandle(direction)
          : getDefaultSemanticSourceHandle(direction),
      ),
      targetHandle: remapHandleForDirection(
        edge.targetHandle,
        direction,
        targetIsData
          ? DATA_TARGET_HORIZONTAL_HANDLES
          : SEMANTIC_TARGET_HORIZONTAL_HANDLES,
        targetIsData
          ? DATA_TARGET_VERTICAL_HANDLES
          : SEMANTIC_TARGET_VERTICAL_HANDLES,
        targetIsData
          ? getDefaultDataTargetHandle(direction)
          : getDefaultSemanticTargetHandle(direction),
      ),
    };
  }
  return {
    sourceHandle: remapHandleForDirection(
      edge.sourceHandle,
      direction,
      DATA_SOURCE_HORIZONTAL_HANDLES,
      DATA_SOURCE_VERTICAL_HANDLES,
      getDefaultDataSourceHandle(direction),
    ),
    targetHandle: remapHandleForDirection(
      edge.targetHandle,
      direction,
      DATA_TARGET_HORIZONTAL_HANDLES,
      DATA_TARGET_VERTICAL_HANDLES,
      getDefaultDataTargetHandle(direction),
    ),
  };
}
