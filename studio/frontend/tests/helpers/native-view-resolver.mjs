// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// register after browser-store-resolver.mjs and route Tauri calls through globalThis.nativeViewCall;
// events reach globalThis.nativeViewListener, toasts and recorded downloads land in globalThis.nativeViewSeen, visits in globalThis.nativeViewVisits
const stub = (source) => `data:text/javascript,${encodeURIComponent(source)}`;

const STUBS = {
  "@tauri-apps/api/event": stub(
    "export const listen = async (_name, callback) => { globalThis.nativeViewListener = callback; return () => {}; };",
  ),
  "@/i18n": stub(
    "export const getLocale = () => 'en'; export const translate = (key) => key;",
  ),
  "@/lib/open-link": stub("export const openExternalLink = () => {};"),
  "@/lib/toast": stub(`
    const seen = (level) => (message) => void (globalThis.nativeViewSeen ??= []).push({ level, message });
    export const toast = Object.assign(seen("info"), { success: seen("success"), error: seen("error"), warning: seen("warning") });
  `),
  "./native-support": stub(`
    export const useNativeBrowser = { getState: () => ({ enabled: true }) };
    export const callNative = (command, args) => globalThis.nativeViewCall(command, args);
    export const nativeClearing = () => false;
    export const onNativeViewsClosed = (callback) => { globalThis.nativeViewsClosed = callback; return () => {}; };
    export const clearNativeBrowsingData = async () => {};
  `),
  "./favicon": stub("export const proxiedFavicon = async () => null;"),
  "./native-downloads": stub("export const decideNativeDownload = async () => {};"),
  "./download-approval-queue": stub(
    "export const approveDownload = async () => globalThis.nativeViewApprove ?? false; export const downloadSiteOf = () => '';",
  ),
  "./history-store": stub(
    "const seen = (entry, temporary) => void (globalThis.nativeViewSeen ??= []).push(temporary ? { ...entry, temporary } : entry);" +
      " export const useBrowserHistoryStore = { getState: () => ({" +
      " recordVisit: (url, _title, temporary) => void (globalThis.nativeViewVisits ??= []).push({ url, temporary: Boolean(temporary) })," +
      " recordDownload: (item, temporary) => seen({ level: 'history', message: item.nativeId }, temporary) }) };",
  ),
};

export function resolve(specifier, context, next) {
  if (STUBS[specifier]) return { url: STUBS[specifier], shortCircuit: true };
  return next(specifier, context);
}
