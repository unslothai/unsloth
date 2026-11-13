// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { NodeConfig } from "../types";
import {
  type ConnectionStatus,
  nodeConnectionStatus,
} from "../utils/node-connection-status";

export function useNodeConnectionStatus(
  nodeId: string,
  config: NodeConfig | undefined,
): ConnectionStatus {
  // Returns one of three shared objects, so a node re-renders only when its own status changes.
  return useRecipeStudioStore((state) =>
    nodeConnectionStatus(nodeId, config, state.edges, state.configs),
  );
}
