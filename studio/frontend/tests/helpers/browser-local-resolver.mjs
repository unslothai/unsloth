// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolve as storeResolve } from "./browser-store-resolver.mjs";
export function resolve(specifier, context, next) {
  if (specifier === "@tauri-apps/api/core") {
    return {
      url: `data:text/javascript,${encodeURIComponent("export const invoke = (command, args) => globalThis.__localInvoke(command, args);")}`,
      shortCircuit: true,
    };
  }
  return storeResolve(specifier, context, next);
}
