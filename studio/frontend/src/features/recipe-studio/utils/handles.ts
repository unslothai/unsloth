// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Connection } from "@xyflow/react";
import type { LayoutDirection } from "../types";

export const HANDLE_IDS = {
  // data flow lanes
  dataIn: "data-in",
  dataInTop: "data-in-top",
  dataInRight: "data-in-right",
  dataInBottom: "data-in-bottom",
  dataOut: "data-out",
  dataOutLeft: "data-out-left",
  dataOutTop: "data-out-top",
  dataOutBottom: "data-out-bottom",
  // semantic dependency lanes
  semanticIn: "semantic-in",
  semanticInTop: "semantic-in-top",
  semanticInRight: "semantic-in-right",
  semanticInBottom: "semantic-in-bottom",
  semanticInLeft: "semantic-in-left",
  semanticOut: "semantic-out",
  semanticOutLeft: "semantic-out-left",
  semanticOutTop: "semantic-out-top",
  semanticOutBottom: "semantic-out-bottom",
  semanticOutRight: "semantic-out-right",
  // llm prompt/scorer lanes
  llmInputOutLeft: "llm-input-out-left",
  llmInputOutRight: "llm-input-out-right",
  llmInputOutTop: "llm-input-out-top",
  llmInputOutBottom: "llm-input-out-bottom",
} as const;

export type RecipeHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];

const LEGACY_HANDLE_ALIAS_MAP: Record<string, string> = {
  [HANDLE_IDS.semanticInLeft]: HANDLE_IDS.semanticIn,
  [HANDLE_IDS.semanticOutRight]: HANDLE_IDS.semanticOut,
};

const DATA_TARGET_HANDLES = new Set<string>([
  HANDLE_IDS.dataIn,
  HANDLE_IDS.dataInTop,
  HANDLE_IDS.dataInRight,
  HANDLE_IDS.dataInBottom,
]);

const DATA_SOURCE_HANDLES = new Set<string>([
  HANDLE_IDS.dataOut,
  HANDLE_IDS.dataOutLeft,
  HANDLE_IDS.dataOutTop,
  HANDLE_IDS.dataOutBottom,
]);

const SEMANTIC_TARGET_HANDLES = new Set<string>([
  HANDLE_IDS.semanticIn,
  HANDLE_IDS.semanticInTop,
  HANDLE_IDS.semanticInRight,
  HANDLE_IDS.semanticInBottom,
  HANDLE_IDS.semanticInLeft,
]);

const SEMANTIC_SOURCE_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOut,
  HANDLE_IDS.semanticOutLeft,
  HANDLE_IDS.semanticOutTop,
  HANDLE_IDS.semanticOutBottom,
  HANDLE_IDS.semanticOutRight,
]);

const DATA_TARGET_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.dataIn,
  HANDLE_IDS.dataInRight,
]);

const DATA_TARGET_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.dataInTop,
  HANDLE_IDS.dataInBottom,
]);

const DATA_SOURCE_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.dataOut,
  HANDLE_IDS.dataOutLeft,
]);

const DATA_SOURCE_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.dataOutTop,
  HANDLE_IDS.dataOutBottom,
]);

const SEMANTIC_TARGET_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticIn,
  HANDLE_IDS.semanticInRight,
  HANDLE_IDS.semanticInLeft,
]);

const SEMANTIC_TARGET_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticInTop,
  HANDLE_IDS.semanticInBottom,
]);

const SEMANTIC_SOURCE_HORIZONTAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOut,
  HANDLE_IDS.semanticOutLeft,
  HANDLE_IDS.semanticOutRight,
]);

const SEMANTIC_SOURCE_VERTICAL_HANDLES = new Set<string>([
  HANDLE_IDS.semanticOutTop,
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
