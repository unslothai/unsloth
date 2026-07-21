// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { App } from "./app/app";
import { fetchDeviceType } from "./config/env";
import { initializeLocale } from "./i18n";

const globalCrypto = globalThis.crypto as Crypto | undefined;

if (globalCrypto && typeof globalCrypto.randomUUID !== "function") {
  // Some envs ship `crypto` without `randomUUID()`. Provide a best-effort v4
  // UUID using `getRandomValues` when available.
  const cryptoRef = globalCrypto;

  function getRandomByte(): number {
    if (typeof cryptoRef.getRandomValues === "function") {
      return cryptoRef.getRandomValues(new Uint8Array(1))[0];
    }
    return Math.floor(Math.random() * 256);
  }

  cryptoRef.randomUUID = (() =>
    "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
      (+c ^ (getRandomByte() & (15 >> (+c / 4)))).toString(16),
    )) as Crypto["randomUUID"];
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

initializeLocale();

// Font rasterization happens in the rendering client, so key off navigator
// rather than the server-reported platform (the backend may be a remote host
// on a different OS). Linux FreeType draws Inter visibly heavier than macOS
// CoreText renders it under the app's antialiased smoothing; index.css
// lightens chat text under this class to compensate. Android is excluded:
// it also uses FreeType, but its display densities hide the difference and
// the compensation is untested there.
const uaLower = navigator.userAgent.toLowerCase();
if (uaLower.includes("linux") && !uaLower.includes("android")) {
  document.documentElement.classList.add("render-linux");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

fetchDeviceType().catch(() => undefined);
