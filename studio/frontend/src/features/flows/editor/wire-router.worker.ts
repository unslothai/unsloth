// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type WireEdgeInput,
  type WireRect,
  type WireRoute,
  type WireRouterCache,
  routeWires,
} from "./wire-router";

export interface RouteRequest {
  id: number;
  /** A new scope (another flow) drops the incremental cache. */
  scope: string;
  rects: WireRect[];
  zones: WireRect[];
  edges: WireEdgeInput[];
  hopRadius: number;
}

export interface RouteResponse {
  id: number;
  routes?: WireRoute[];
  ms?: number;
  error?: string;
}

const scope = self as unknown as {
  onmessage: ((event: MessageEvent<RouteRequest>) => void) | null;
  postMessage: (message: RouteResponse) => void;
};

let cache: { scope: string; value: WireRouterCache } | null = null;

scope.onmessage = (event) => {
  const request = event.data;
  try {
    const started = performance.now();
    const result = routeWires(
      request.rects,
      request.edges,
      { hopRadius: request.hopRadius },
      cache?.scope === request.scope ? cache.value : undefined,
      request.zones,
    );
    cache = { scope: request.scope, value: result.cache };
    scope.postMessage({
      id: request.id,
      routes: [...result.routes.values()],
      ms: performance.now() - started,
    });
  } catch (error) {
    scope.postMessage({
      id: request.id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
