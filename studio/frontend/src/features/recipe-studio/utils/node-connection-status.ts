// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge } from "@xyflow/react";
import { INFRA_NODE_KINDS, type NodeConfig } from "../types";

export type ConnectionStatus = {
  isDisconnected: boolean;
  /** No incoming data edge (infra edges only). */
  missingDataInput: boolean;
};

const CONNECTED: ConnectionStatus = {
  isDisconnected: false,
  missingDataInput: false,
};
const DISCONNECTED: ConnectionStatus = {
  isDisconnected: true,
  missingDataInput: false,
};
const MISSING_DATA_INPUT: ConnectionStatus = {
  isDisconnected: false,
  missingDataInput: true,
};

type Adjacency = Map<string, string[]>;

// One index per edges array, shared by every node: the store replaces the array on each change.
const adjacencyByEdges = new WeakMap<readonly Edge[], Adjacency>();

export function edgeAdjacency(edges: readonly Edge[]): Adjacency {
  const cached = adjacencyByEdges.get(edges);
  if (cached) {
    return cached;
  }
  const index: Adjacency = new Map();
  const add = (nodeId: string, otherId: string) => {
    const others = index.get(nodeId);
    if (others) {
      others.push(otherId);
    } else {
      index.set(nodeId, [otherId]);
    }
  };
  for (const edge of edges) {
    add(edge.source, edge.target);
    add(edge.target, edge.source);
  }
  adjacencyByEdges.set(edges, index);
  return index;
}

export function nodeConnectionStatus(
  nodeId: string,
  config: NodeConfig | undefined,
  edges: readonly Edge[],
  configs: Record<string, NodeConfig>,
): ConnectionStatus {
  if (!config || config.kind === "markdown_note") {
    return CONNECTED;
  }
  const others = edgeAdjacency(edges).get(nodeId);
  if (!others) {
    return DISCONNECTED;
  }
  if (config.kind !== "llm") {
    return CONNECTED;
  }
  const hasDataEdge = others.some((otherId) => {
    const otherConfig = configs[otherId];
    return otherConfig && !INFRA_NODE_KINDS.has(otherConfig.kind);
  });
  return hasDataEdge ? CONNECTED : MISSING_DATA_INPUT;
}

// One key per configs object, shared by every node: the store replaces the object on each change.
const localProviderKeyByConfigs = new WeakMap<
  Record<string, NodeConfig>,
  string
>();

export function localProviderNamesKey(
  configs: Record<string, NodeConfig>,
): string {
  let key = localProviderKeyByConfigs.get(configs);
  if (key === undefined) {
    const names: string[] = [];
    for (const cfg of Object.values(configs)) {
      if (cfg.kind === "model_provider" && cfg.is_local === true) {
        names.push(cfg.name);
      }
    }
    key = JSON.stringify(names);
    localProviderKeyByConfigs.set(configs, key);
  }
  return key;
}
