// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// bundler-resolver's two rules, plus the barrels loaded-models-api.ts reaches that pull in
// TSX: chat, images, video and the embedder go to loaded-models-eject-stub.ts, auth to its store stub.
import { resolve as resolveBundler } from "../bundler-resolver.mjs";

const EJECT_STUB = new URL("./loaded-models-eject-stub.ts", import.meta.url)
  .href;
const STUBS = new Map([
  ["@/features/chat", EJECT_STUB],
  ["@/features/images/api", EJECT_STUB],
  ["@/features/video/api", EJECT_STUB],
  ["@/features/settings/api/embedding-model", EJECT_STUB],
  ["@/features/settings/stores/embedding-model-store", EJECT_STUB],
  ["@/features/auth", new URL("./store-stubs/auth.ts", import.meta.url).href],
]);

export function resolve(specifier, context, next) {
  const stub = STUBS.get(specifier);
  if (stub) return next(stub, context);
  return resolveBundler(specifier, context, next);
}
