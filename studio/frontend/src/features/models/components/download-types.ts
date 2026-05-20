// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface InventoryHint {
  kind: "model" | "gguf" | "dataset";
  repoId: string;
  bytes?: number;
}
