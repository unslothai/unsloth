// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The tracker's network, toast and hub imports do not load in bare node; one stub answers for all.
import { resolve as resolveBundler } from "./bundler-resolver.mjs";

const DEPS = "./helpers/recipe-tracker-deps.ts";

export function resolve(specifier, context, next) {
  if (
    specifier === "@/features/hub" ||
    specifier === "@/shared/toast" ||
    (specifier === "../api" &&
      context.parentURL?.endsWith("/recipe-studio/executions/tracker.ts"))
  ) {
    return next(new URL(DEPS, import.meta.url).href, context);
  }
  return resolveBundler(specifier, context, next);
}
