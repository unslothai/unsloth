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
import { ORB_WIDGET_QUERY } from "./features/chat/voice/orb-widget";
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

// Rasterization follows the browser OS, not the potentially remote server.
// This adjustment is calibrated for desktop Linux, so exclude Android.
const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

// index.css keys off this to restore ::-webkit-scrollbar styling on Windows.
if (uaLower.includes("windows")) {
  document.documentElement.classList.add("client-windows");
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
applyCodeBlockContainment();
watchCodeBlockContainmentOverride();
applyReasoningRowContainment();
watchReasoningRowContainmentOverride();

// Keep right-edge controls clear of overlay scrollbars.
watchOverlayScrollbarGutter(window);
watchInputModality(window);

// Before the first render, so a disabled spell check never flashes underlines.
watchSpellCheck(window);

// The floating voice orb's own window (orb-widget-host.ts): the ball and nothing else.
const isOrbWidget =
  isTauri && new URLSearchParams(window.location.search).has(ORB_WIDGET_QUERY);
if (isOrbWidget) {
  const transparent = document.createElement("style");
  transparent.textContent = "html, body, #root { background: transparent !important; overflow: hidden; }";
  document.head.appendChild(transparent);
}

function renderApp(): void {
  if (isOrbWidget) {
    void import("./components/assistant-ui/orb-widget-view").then(({ OrbWidgetView }) =>
      root.render(
        <StrictMode>
          <OrbWidgetView />
        </StrictMode>,
      ),
    );
    return;
  }
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
