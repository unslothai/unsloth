// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { INFRA_NODE_KINDS, type NodeConfig } from "../types";

type ConnectionStatus = {
  /** True when the node has zero edges at all. */
  isDisconnected: boolean;
  /** True when an LLM node has no incoming data edge (only infra). */
  missingDataInput: boolean;
};

export function useNodeConnectionStatus(
  nodeId: string,
  config: NodeConfig | undefined,
): ConnectionStatus {
  const edges = useRecipeStudioStore((state) => state.edges);
  const configs = useRecipeStudioStore((state) => state.configs);

  return useMemo(() => {
    const empty: ConnectionStatus = {
      isDisconnected: false,
      missingDataInput: false,
    };

    if (!config || config.kind === "markdown_note") {
      return empty;
    }

    const nodeEdges = edges.filter(
      (e) => e.source === nodeId || e.target === nodeId,
    );
    const isDisconnected = nodeEdges.length === 0;

    let missingDataInput = false;
    if (config.kind === "llm" && !isDisconnected) {
      const hasDataEdge = nodeEdges.some((e) => {
        const otherId = e.source === nodeId ? e.target : e.source;
        const otherConfig = configs[otherId];
        return otherConfig && !INFRA_NODE_KINDS.has(otherConfig.kind);
      });
      missingDataInput = !hasDataEdge;
    }

    return {
      isDisconnected,
      missingDataInput,
    };
  }, [nodeId, config, edges, configs]);
}
