// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Downloads from the desktop app's native views. A file that runs code arrives under a neutral name
// (src-tauri/src/browser_webview.rs) and only takes its own once the reader keeps it.

import type { TranslationKey } from "@/i18n";
import type { InterpolationValues } from "@/i18n";

export type NativeDownloadEvent = {
  kind: "download";
  tabId: string;
  url: string;
  name: string;
  path: string | null;
  size: number | null;
  done: boolean;
  success: boolean;
  /** Set for a download staged until the reader keeps it. */
  id?: string | null;
  needsApproval?: boolean;
  /** Whether the file was marked as from the internet; null where nothing marks it. */
  marked?: boolean | null;
};

type ToastAction = { label: string; onClick: () => void };
type ToastOptions = {
  id?: string;
  duration?: number;
  description?: string;
  action?: ToastAction;
  cancel?: ToastAction;
  onDismiss?: () => void;
};
type Notify = {
  (message: string, options?: ToastOptions): unknown;
  success: (message: string, options?: ToastOptions) => unknown;
  error: (message: string, options?: ToastOptions) => unknown;
  warning: (message: string, options?: ToastOptions) => unknown;
};

export type NativeDownloadDeps = {
  call: <T>(command: string, args: Record<string, unknown>) => Promise<T>;
  toast: Notify;
  record: (item: { name: string; url: string | null; size: number; contentType: string }) => void;
  t: (key: TranslationKey, values?: InterpolationValues) => string;
};

// Ids already kept or discarded: sonner keeps a replaced toast's onDismiss, so a late close can't
// discard a kept file, and a second click can't act twice.
const settled = new Set<string>();

export function handleNativeDownload(event: NativeDownloadEvent, deps: NativeDownloadDeps): void {
  const { toast, t } = deps;
  const id = event.id ?? null;
  if (!event.done) {
    toast(t(id ? "browser.downloadSafety.stagedDownloading" : "browser.native.downloading", { name: event.name }));
    return;
  }
  if (!event.success) {
    toast.error(t("browser.native.downloadFailed", { name: event.name }));
    return;
  }
  if (id && event.needsApproval) {
    askToKeep(event, id, deps);
    return;
  }
  deps.record({ name: event.name, url: event.url, size: event.size ?? 0, contentType: "" });
  if (event.marked === false) toast.warning(t("browser.downloadSafety.notMarked", { name: event.name }));
  else toast.success(t("browser.native.downloaded", { name: event.name }));
}

function askToKeep(event: NativeDownloadEvent, id: string, deps: NativeDownloadDeps, description?: string): void {
  const { toast, t } = deps;
  const discard = () => {
    if (settled.has(id)) return;
    settled.add(id);
    void deps.call("browser_download_discard", { id }).catch(() => undefined);
  };
  const keep = () => {
    if (settled.has(id)) return;
    settled.add(id);
    deps.call<string>("browser_download_keep", { id }).then(
      (name) => {
        deps.record({ name, url: event.url, size: event.size ?? 0, contentType: "" });
        toast.success(t("browser.native.downloaded", { name }));
      },
      (error: unknown) => {
        // Still staged: offer Discard, with why it couldn't be kept.
        settled.delete(id);
        askToKeep(event, id, deps, error instanceof Error ? error.message : String(error));
      },
    );
  };
  toast(t(description ? "browser.downloadSafety.keepFailed" : "browser.downloadSafety.keepPrompt", { name: event.name }), {
    id: `browser-download-${id}`,
    duration: Number.POSITIVE_INFINITY,
    description,
    action: description ? undefined : { label: t("browser.downloadSafety.keep"), onClick: keep },
    cancel: { label: t("browser.downloadSafety.discard"), onClick: discard },
    onDismiss: discard,
  });
}
