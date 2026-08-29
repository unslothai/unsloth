// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store stubs, plus the one barrel the chat API adds on top of them.
// Register this to drive applyActiveModelStatusToStore against the real store.
import { resolve as resolveStoreSettings } from "./store-settings-resolver.mjs";

const STUB = new URL("./helpers/store-stubs/hf-auth.ts", import.meta.url).href;

export function resolve(specifier, context, next) {
  if (specifier === "@/features/hf-auth") {
    return next(STUB, context);
  }
  return resolveStoreSettings(specifier, context, next);
}
