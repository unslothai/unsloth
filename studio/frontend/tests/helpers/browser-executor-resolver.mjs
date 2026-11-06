// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The executor without a DOM: the page answers through globalThis.__frameAgent, the panel is shown,
// and the native view, capture and history modules are inert.
import { resolve as browserStoreResolve } from "./browser-store-resolver.mjs";

const stub = (source) => ({ url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true });

const STUBS = {
  "../page-frame":
    "export const requestFrameAgent = (tabId, op, code, timeoutMs = 5000) => globalThis.__frameAgent(tabId, op, code, timeoutMs);" +
    " export const sendFrameCommand = (tabId, command) => globalThis.__frameCommand?.(tabId, command) ?? false;" +
    " export const whenWebPageReady = async (tabId, signal) => globalThis.__webReady ? globalThis.__webReady(tabId, signal) : true;",
  "../capture": "export const screenshotPage = async (...args) => globalThis.__screenshotPage?.(...args) ?? null; export const captureHiddenPage = async (...args) => globalThis.__captureHiddenPage?.(...args) ?? null;",
  // Native pages only when a test sets globalThis.__native; it answers their calls via __callNative.
  "../native-view":
    "export const hasNativeView = () => globalThis.__native === true; export const nativeAction = () => {};" +
    " export const keepInBackground = (tabId) => globalThis.__keep?.(tabId);" +
    " export const releaseBackground = (tabId) => void (globalThis.__released ??= []).push(tabId);" +
    " export const whenViewOpen = async () => true;",
  "../native-support":
    "export const useNativeBrowser = { getState: () => ({ enabled: globalThis.__native === true }) };" +
    " export const callNative = async (command, args) => (globalThis.__callNative ? globalThis.__callNative(command, args) : null);" +
    " export const localOriginGranted = (tabId, url) => globalThis.__granted?.get(tabId) === new URL(url).origin;",
  "../panel-availability": "export const browserPanelAvailable = () => globalThis.__panel !== false;",
  "../history-store": "export const useBrowserHistoryStore = { getState: () => ({ recordVisit() {} }) };",
  "./file-reader":
    "export const fileTabKind = ({ name }) => /\\.html?$/.test(name) ? 'preview' : /\\.pdf$/.test(name) ? 'document' : /\\.png$/.test(name) ? 'image' : 'text';" +
    " export const editable = (entry, kind) => kind === 'preview' || kind === 'text';" +
    " export const fileText = async (entry, blob) => ({ text: await blob.text(), cut: false });",
};

export function resolve(specifier, context, next) {
  if ((context.parentURL?.includes("/src/features/browser/agent/executor.ts") || context.parentURL?.includes("/src/features/browser/agent/tab-sharing.ts")) && specifier in STUBS) {
    return stub(STUBS[specifier]);
  }
  if (specifier.startsWith("../") && context.parentURL?.includes("/src/features/browser/agent/") && !/\.\w+$/.test(specifier)) {
    return next(`${specifier}.ts`, context);
  }
  return browserStoreResolve(specifier, context, next);
}
