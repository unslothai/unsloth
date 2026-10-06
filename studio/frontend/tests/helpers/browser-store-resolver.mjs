// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The browser store only needs the chat feature to close its artifact surface; stub that.
const CHAT_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const useChatArtifactsStore = { getState: () => ({ closeArtifactSurface() {} }) };");

// Messages come back as their keys; the browser stores only need translate() to return a string.
const I18N_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const getLocale = () => \"en\"; export const translate = (key) => key;");

// Toasts render through the UI kit; a no-op stands in.
const TOAST_STUB =
  "data:text/javascript," +
  encodeURIComponent("const toast = () => {}; toast.error = toast.success = () => {}; export { toast };");

export function resolve(specifier, context, next) {
  if (specifier === "@/lib/toast") return { url: TOAST_STUB, shortCircuit: true };
  if (specifier === "@/features/chat") return { url: CHAT_STUB, shortCircuit: true };
  if (specifier === "@/i18n") return { url: I18N_STUB, shortCircuit: true };
  if (specifier.startsWith("@/lib/")) {
    return next(new URL(`../../src/lib/${specifier.slice(6)}.ts`, import.meta.url).href, context);
  }
  if (specifier.startsWith("./") && context.parentURL?.includes("/src/features/browser/") && !/\.\w+$/.test(specifier)) {
    return next(`${specifier}.ts`, context);
  }
  return next(specifier, context);
}
