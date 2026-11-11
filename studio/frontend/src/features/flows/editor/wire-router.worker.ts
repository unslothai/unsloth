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
  /** Only some wires were sent (a drag): the rest keep their routes. */
  partial?: boolean;
}

export interface RouteResponse {
  id: number;
  routes?: WireRoute[];
  ms?: number;
  partial?: boolean;
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
      // A full pass routes from scratch, so a document always gets the same routes; only a
      // drag's partial passes reuse the last result.
      request.partial && cache?.scope === request.scope ? cache.value : undefined,
      request.zones,
    );
    cache = { scope: request.scope, value: result.cache };
    scope.postMessage({
      id: request.id,
      routes: [...result.routes.values()],
      ms: performance.now() - started,
      partial: request.partial,
    });
  } catch (error) {
    scope.postMessage({
      id: request.id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
