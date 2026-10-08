// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// register after browser-store-resolver.mjs and route Tauri calls through globalThis.nativeViewCall;
// events reach globalThis.nativeViewListener, toasts and recorded downloads land in globalThis.nativeViewSeen
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
    export const onNativeViewsClosed = () => () => {};
    export const clearNativeBrowsingData = async () => {};
  `),
  "./favicon": stub("export const proxiedFavicon = async () => null;"),
  // globalThis.nativeViewDownloadsButton puts a Downloads button on screen; calls land in nativeViewActivity.
  "./download-activity": stub(`
    const buttons = () => (globalThis.nativeViewDownloadsButton ? 1 : 0);
    const log = (call, key) => void (globalThis.nativeViewActivity ??= []).push(call + " " + key);
    export const useDownloadActivity = { getState: () => ({ buttons: buttons() }) };
    export const beginDownload = (key) => log("begin", key);
    export const abandonDownload = (key) => log("abandon", key);
    export const finishDownload = (key) => (log("finish", key), buttons() > 0);
  `),
  "./native-downloads": stub("export const decideNativeDownload = async () => {};"),
  "./download-approval-queue": stub(
    "export const approveDownload = async () => Boolean(globalThis.nativeViewApprove); export const downloadSiteOf = () => '';",
  ),
  "./history-store": stub(
    "export const useBrowserHistoryStore = { getState: () => ({ recordVisit() {}," +
      " recordDownload: (item) => void (globalThis.nativeViewSeen ??= []).push({ level: 'history', message: item.nativeId }) }) };",
  ),
};

export function resolve(specifier, context, next) {
  if (STUBS[specifier]) return { url: STUBS[specifier], shortCircuit: true };
  return next(specifier, context);
}
