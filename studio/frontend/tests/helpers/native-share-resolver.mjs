// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { resolve as nativeResolve } from './native-view-resolver.mjs';
import { resolve as executorResolve } from './browser-executor-resolver.mjs';
export function resolve(specifier, context, next) {
  if (specifier === '../native-view' && context.parentURL?.includes('/src/features/browser/agent/')) {
    return next(new URL('../../src/features/browser/native-view.ts', import.meta.url).href, context);
  }
  if (specifier === '../native-support' && context.parentURL?.includes('/src/features/browser/agent/')) {
    return nativeResolve('./native-support', context, next);
  }
  return nativeResolve(specifier, context, (s, c) => executorResolve(s, c, next));
}
