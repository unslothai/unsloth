// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { App } from "./app/app";
import {
  applyCodeBlockContainment,
  watchCodeBlockContainmentOverride,
} from "./components/assistant-ui/code-block-containment";
import {
  applyMathBlockContainment,
  watchMathBlockContainmentOverride,
} from "./components/assistant-ui/math-block-containment";
import {
  applyReasoningRowContainment,
  watchReasoningRowContainmentOverride,
} from "./components/assistant-ui/reasoning-row-containment";
import { fetchDeviceType } from "./config/env";
import { refreshSession } from "./features/auth/api";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasSettledAuthSession,
} from "./features/auth/session";
import { startStartupCredentialBootstrap } from "./features/credentials/bootstrap";
import { loadChatPage, prefetchChatPage } from "./features/chat/load-chat-page";
import {
  applyInterfaceScaleBeforeFirstPaint,
  useInterfaceScaleStore,
} from "./features/settings/stores/interface-scale-store";
import { initializeLocale } from "./i18n";
import { isTauri } from "./lib/api-base";
import { markAppSessionStart } from "./lib/app-session-start";
import { watchClosingDialogFocus } from "./lib/closing-dialog-focus";
import { setHubSessionRefresh } from "./lib/hf-endpoint";
import { watchInputModality } from "./lib/input-modality";
import { watchMacStrayKeyText } from "./lib/mac-stray-key-text";
import { watchOverlayScrollbarGutter } from "./lib/overlay-scrollbar";
import { pinPointerListenerRegions } from "./lib/pointer-listener-pin";
import { watchSpellCheck } from "./lib/spellcheck";
import { watchTimeFormat } from "./lib/time-format";

markAppSessionStart();
setHubSessionRefresh(refreshSession);

const startPath = window.location.pathname;
const chatPageAtStart =
  startPath === "/chat" ||
  (startPath === "/" && hasAuthToken() && getPostAuthRoute() === "/chat")
    ? loadChatPage().then(
        () => undefined,
        () => undefined,
      )
    : null;

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
const root = createRoot(rootElement);

if (isTauri) {
  document.documentElement.classList.add("tauri");
}

const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

if (uaLower.includes("windows")) {
  document.documentElement.classList.add("client-windows");
}

// Must run before the first render: arming the containment rule late relayouts the first thread.
applyMathBlockContainment();
watchMathBlockContainmentOverride();
applyCodeBlockContainment();
watchCodeBlockContainmentOverride();
applyReasoningRowContainment();
watchReasoningRowContainmentOverride();

watchOverlayScrollbarGutter(window);
watchInputModality(window);
// Before any panel group mounts: their document pointer listeners then never change WebKit's page-wide set.
pinPointerListenerRegions(document);
watchClosingDialogFocus(document);

watchSpellCheck(window);
watchTimeFormat(window);

// The macOS app types a control character for a key the page leaves unhandled (#14764).
if (isTauri && /Mac/.test(navigator.platform)) {
  watchMacStrayKeyText(window);
}

const APP_SHELL_READY_EVENT = "unsloth:app-shell-ready";
const CHAT_PREFETCH_FALLBACK_MS = 3000;

function renderApp(): void {
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  if (!chatPageAtStart) {
    const prefetchOnce = () => {
      window.removeEventListener(APP_SHELL_READY_EVENT, afterReady);
      window.clearTimeout(fallback);
      prefetchChatPage();
    };
    const afterReady = () => {
      window.setTimeout(prefetchOnce, 0);
    };
    window.addEventListener(APP_SHELL_READY_EVENT, afterReady, { once: true });
    const fallback = window.setTimeout(prefetchOnce, CHAT_PREFETCH_FALLBACK_MS);
  }
}

const localeInitialization = initializeLocale();
const interfaceScaleInitialization = applyInterfaceScaleBeforeFirstPaint(
  useInterfaceScaleStore.getState().scale,
);
if (typeof localeInitialization !== "string" || isTauri || chatPageAtStart) {
  Promise.all([
    localeInitialization,
    interfaceScaleInitialization,
    chatPageAtStart,
  ]).then(renderApp);
} else {
  renderApp();
}

fetchDeviceType().catch(() => undefined);

// Desktop signs in after its backend starts, so its gate owns reconciliation.
if (
  !isTauri &&
  hasSettledAuthSession() &&
  startPath !== "/login" &&
  startPath !== "/change-password"
) {
  startStartupCredentialBootstrap();
}
