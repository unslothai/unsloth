// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const STUBS = new Map([
  ["@/features/auth", "./helpers/store-stubs/settings-http.ts"],
  ["@/features/hf-auth", "./helpers/store-stubs/auth.ts"],
  ["@/features/hub", "./helpers/store-stubs/hub.ts"],
  ["@/features/model-picker", "./helpers/store-stubs/model-picker.ts"],
  ["@/config/env", "./helpers/store-stubs/env.ts"],
  ["@/lib/toast", "./helpers/store-stubs/toast.ts"],
]);

const PER_MODEL_CONFIG = "@/features/model-picker/model-config/per-model-config";
const RESIDENT_CONFIG_READER = "/src/features/chat/lib/apply-inference-status-to-store.ts";

export function resolve(specifier, context, next) {
  if (specifier === PER_MODEL_CONFIG && context.parentURL?.endsWith(RESIDENT_CONFIG_READER)) {
    return next(new URL("./helpers/store-stubs/per-model-config-resident.ts", import.meta.url).href, context);
  }
  const stub = STUBS.get(specifier);
  if (stub) {
    return next(new URL(stub, import.meta.url).href, context);
  }
  return resolveBundler(specifier, context, next);
}
