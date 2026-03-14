// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import type { NodeConfig } from "../types";

type ConnectionStatus = {
  /** True when the node has zero edges at all. */
  isDisconnected: boolean;
  /** True when an LLM node has no incoming data edge (only infra). */
  missingDataInput: boolean;
  /** True when an LLM node has no model config connection. */
  needsModelConfig: boolean;
  /** True when a model_config node has no provider connection. */
  needsProvider: boolean;
};

const INFRA_KINDS = new Set(["model_provider", "model_config", "tool_config"]);

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
      needsModelConfig: false,
      needsProvider: false,
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
        return otherConfig && !INFRA_KINDS.has(otherConfig.kind);
      });
      missingDataInput = !hasDataEdge;
    }

    const needsModelConfig =
      config.kind === "llm" && !config.model_alias?.trim();
    const needsProvider =
      config.kind === "model_config" && !config.provider?.trim();

    return {
      isDisconnected,
      missingDataInput,
      needsModelConfig,
      needsProvider,
    };
  }, [nodeId, config, edges, configs]);
}
