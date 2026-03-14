// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { fetchDeviceType } from "./config/env";
import { App } from "./app/app";

const globalCrypto = globalThis.crypto as Crypto | undefined;

if (globalCrypto && typeof globalCrypto.randomUUID !== "function") {
  // Some envs ship `crypto` but no `randomUUID()` (or a non-function stub).
  // Provide a best-effort v4 UUID using `getRandomValues` when available.
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

fetchDeviceType().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
