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
import {
  applyInterfaceScaleBeforeFirstPaint,
  useInterfaceScaleStore,
} from "./features/settings/stores/interface-scale-store";
import { initializeLocale } from "./i18n";
import { isTauri } from "./lib/api-base";
import { watchOverlayScrollbarGutter } from "./lib/overlay-scrollbar";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
const root = createRoot(rootElement);

if (isTauri) {
  document.documentElement.classList.add("tauri");
}

// Rasterization follows the browser OS, not the potentially remote server.
// This adjustment is calibrated for desktop Linux, so exclude Android.
const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

// Whether off-screen maths takes containment. ON by default, subject to a feature detect for the
// engine's find-in-page, so on a recent engine this normally SETS the attribute and arms the rule;
// on an older one it removes an attribute that was never there. Before the first render, because
// the rule it arms is a rendering rule and arming it late would relayout the first thread that
// mounts.
applyMathBlockContainment();
// And keep watching, so a devtools flip of `__UNSLOTH_MATH_BLOCK_CONTAINMENT__` reapplies instead of
// leaving the session measuring the arm it was already in.
watchMathBlockContainmentOverride();

// Keep right-edge controls clear of overlay scrollbars.
watchOverlayScrollbarGutter(window);

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
