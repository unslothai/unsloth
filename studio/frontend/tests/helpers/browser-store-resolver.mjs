// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The browser store only needs the chat feature to close its artifact surface; stub that.
const CHAT_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const useChatArtifactsStore = { getState: () => ({ closeArtifactSurface() {} }) };");

const I18N_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const getLocale = () => \"en\"; export const translate = (key) => key;");

const TOAST_STUB =
  "data:text/javascript," +
  encodeURIComponent(
    "const toast = (message, options) => void (globalThis.__toasts ??= []).push({ message, options });" +
      " toast.error = toast.success = toast.warning = toast; export { toast };",
  );

const AUTH_STUB =
  "data:text/javascript," +
  encodeURIComponent("export const authFetch = () => { throw new Error(\"no network in tests\"); };");

export function resolve(specifier, context, next) {
  if (specifier === "@/features/auth") return { url: AUTH_STUB, shortCircuit: true };
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
