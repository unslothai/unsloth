// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type LayoutRequest, computeLayout } from "./layout-core";

const scope = self as unknown as {
  onmessage:
    | ((event: MessageEvent<{ id: number; request: LayoutRequest }>) => void)
    | null;
  postMessage: (message: unknown) => void;
};

scope.onmessage = (event) => {
  const { id, request } = event.data;
  try {
    scope.postMessage({ id, result: computeLayout(request) });
  } catch (error) {
    scope.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
