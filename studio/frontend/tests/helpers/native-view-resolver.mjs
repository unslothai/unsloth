// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Registered after browser-store-resolver.mjs: native-view.ts with the Tauri bridge swapped for
// globalThis.nativeViewCall, and the modules it reaches for i18n, toasts, favicons and history stubbed.
const stub = (source) => `data:text/javascript,${encodeURIComponent(source)}`;

const STUBS = {
  "@tauri-apps/api/event": stub("export const listen = async () => () => {};"),
  "@/i18n": stub(
    "export const getLocale = () => 'en'; export const translate = (key) => key;",
  ),
  "@/lib/open-link": stub("export const openExternalLink = () => {};"),
  "@/lib/toast": stub(
    "export const toast = Object.assign(() => {}, { success() {}, error() {} });",
  ),
  "./native-support": stub(`
    export const useNativeBrowser = { getState: () => ({ enabled: true }) };
    export const callNative = (command, args) => globalThis.nativeViewCall(command, args);
    export const nativeClearing = () => false;
    export const onNativeViewsClosed = () => () => {};
    export const clearNativeBrowsingData = async () => {};
  `),
  "./favicon": stub("export const proxiedFavicon = async () => null;"),
  "./history-store": stub(
    "export const useBrowserHistoryStore = { getState: () => ({ recordVisit() {}, recordDownload() {} }) };",
  ),
};

export function resolve(specifier, context, next) {
  if (STUBS[specifier]) return { url: STUBS[specifier], shortCircuit: true };
  return next(specifier, context);
}
