// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let operationId = 0;
let savePending = false;

export const MCP_IMAGE_SETTING_SAVED_EVENT = "mcp-image-setting-saved";

export function beginMcpImageSettingRefresh(): number | null {
  if (savePending) return null;
  operationId += 1;
  return operationId;
}

export function beginMcpImageSettingSave(): number {
  savePending = true;
  operationId += 1;
  return operationId;
}

export function canApplyMcpImageSettingRefresh(requestId: number): boolean {
  return !savePending && requestId === operationId;
}

export function canApplyMcpImageSettingSave(requestId: number): boolean {
  return requestId === operationId;
}

export function finishMcpImageSettingSave(requestId: number): boolean {
  if (requestId !== operationId) return false;
  savePending = false;
  operationId += 1;
  return true;
}
