// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The executor without a DOM: the page answers through globalThis.__frameAgent, the panel is shown,
// and the native view, capture and history modules are inert.
import { resolve as browserStoreResolve } from "./browser-store-resolver.mjs";

const stub = (source) => ({ url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true });

const STUBS = {
  "../page-frame": "export const requestFrameAgent = (tabId, op, code) => globalThis.__frameAgent(tabId, op, code);",
  "../capture": "export const screenshotPage = async () => null;",
  // Off unless a test sets globalThis.__nativeView, which then answers the view's calls.
  "../native-view": "export const hasNativeView = () => Boolean(globalThis.__nativeView); export const nativeAction = () => {};",
  "../native-support":
    "export const useNativeBrowser = { getState: () => ({ enabled: Boolean(globalThis.__nativeView) }) }; export const callNative = async (cmd, args) => globalThis.__nativeView?.(cmd, args) ?? null;",
  "../panel-availability": "export const browserPanelAvailable = () => true;",
  "../history-store": "export const useBrowserHistoryStore = { getState: () => ({ recordVisit() {} }) };",
};

export function resolve(specifier, context, next) {
  if (context.parentURL?.includes("/src/features/browser/agent/executor.ts") && specifier in STUBS) {
    return stub(STUBS[specifier]);
  }
  if (specifier.startsWith("../") && context.parentURL?.includes("/src/features/browser/agent/") && !/\.\w+$/.test(specifier)) {
    return next(`${specifier}.ts`, context);
  }
  return browserStoreResolve(specifier, context, next);
}
