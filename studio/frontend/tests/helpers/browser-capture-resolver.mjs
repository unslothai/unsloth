// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const modules = {
  "@/lib/api-base": "export const isTauri = true; export const apiUrl = path => path;",
  "./native-support": "export const callNative = (...args) => globalThis.captureNativeCall(...args);",
  "./screenshot-support": "export const cropTargets = new WeakMap();",
  "./native-view": "export const hasNativeView = () => true; export const whenNativeViewShown = async () => true;",
  "./page-frame": "export const requestFrameSnapshot = async () => null;",
  "./store": "export const currentEntry = tab => tab.history[tab.index];",
};
export function resolve(specifier, context, next) {
  if (context.parentURL?.endsWith("/src/features/browser/capture.ts") && specifier in modules) {
    return { url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true };
  }
  return next(specifier, context);
}
