// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Register AFTER settings-api-resolver so this sees the bare specifier first.
// See hub-inventory-stub.mjs.
const STUB = new URL("./hub-inventory-stub.mjs", import.meta.url).href;

export function resolve(specifier, context, next) {
  if (specifier === "@/features/hub" || /(^|\/)features\/hub$/.test(specifier)) {
    return next(STUB, context);
  }
  return next(specifier, context);
}
