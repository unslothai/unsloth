// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { App } from "./app/app";
import {
  applyMathBlockContainment,
  watchMathBlockContainmentOverride,
} from "./components/assistant-ui/math-block-containment";
import { fetchDeviceType } from "./config/env";
import { refreshSession } from "./features/auth/api";
import {
  applyInterfaceScaleBeforeFirstPaint,
  useInterfaceScaleStore,
} from "./features/settings/stores/interface-scale-store";
import { initializeLocale } from "./i18n";
import { isTauri } from "./lib/api-base";
import { setHubSessionRefresh } from "./lib/hf-endpoint";
import { watchInputModality } from "./lib/input-modality";
import { watchOverlayScrollbarGutter } from "./lib/overlay-scrollbar";
import { watchSpellCheck } from "./lib/spellcheck";

setHubSessionRefresh(refreshSession);

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
const root = createRoot(rootElement);

if (isTauri) {
  document.documentElement.classList.add("tauri");
}

// Rasterization follows the browser OS, not the possibly remote server.
// Calibrated for desktop Linux, so Android is excluded.
const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

// index.css keys off this to restore ::-webkit-scrollbar styling on Windows.
if (uaLower.includes("windows")) {
  document.documentElement.classList.add("client-windows");
}

// Must run before the first render: arming the containment rule late relayouts the first thread.
applyMathBlockContainment();
// Reapply when the devtools override `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` flips.
watchMathBlockContainmentOverride();

watchOverlayScrollbarGutter(window);
watchInputModality(window);

// Before the first render, so a disabled spell check never flashes underlines.
watchSpellCheck(window);

function renderApp(): void {
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

const localeInitialization = initializeLocale();
const interfaceScaleInitialization = applyInterfaceScaleBeforeFirstPaint(
  useInterfaceScaleStore.getState().scale,
);
if (typeof localeInitialization !== "string" || isTauri) {
  Promise.all([localeInitialization, interfaceScaleInitialization]).then(
    renderApp,
  );
} else {
  renderApp();
}

fetchDeviceType().catch(() => undefined);
