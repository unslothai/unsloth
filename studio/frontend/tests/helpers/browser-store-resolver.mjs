// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const CHAT_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const useChatArtifactsStore = { getState: () => ({ closeArtifactSurface() {} }) };");

export function resolve(specifier, context, next) {
  if (specifier === "@/features/chat") return { url: CHAT_STUB, shortCircuit: true };
  if (specifier.startsWith("@/lib/")) {
    return next(new URL(`../../src/lib/${specifier.slice(6)}.ts`, import.meta.url).href, context);
  }
  if (specifier.startsWith("./") && context.parentURL?.includes("/src/features/browser/") && !/\.\w+$/.test(specifier)) {
    return next(`${specifier}.ts`, context);
  }
  return next(specifier, context);
}
