import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

const globalCrypto = globalThis.crypto as Crypto | undefined;
const hasUuid =
  globalCrypto && typeof (globalCrypto as Crypto).randomUUID === "function";

if (globalCrypto && !hasUuid) {
  // Some envs ship `crypto` but no `randomUUID()` (or a non-function stub).
  // Provide a best-effort v4 UUID using `getRandomValues` when available.
  const getRandomByte = () => {
    if (typeof globalCrypto.getRandomValues === "function") {
      return globalCrypto.getRandomValues(new Uint8Array(1))[0];
    }
    return Math.floor(Math.random() * 256);
  };

  (globalCrypto as Crypto).randomUUID = (() =>
    "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
      (+c ^ (getRandomByte() & (15 >> (+c / 4)))).toString(16),
    )) as Crypto["randomUUID"];
}

import "./index.css";
import { App } from "./app/app";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
