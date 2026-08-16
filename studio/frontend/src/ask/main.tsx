// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoot } from "react-dom/client";

import "@/index.css";
import { initializeLocale } from "@/i18n";
import { AskApp } from "./ask-app";

// Follows the main window's theme: storage events fire across same-origin
// windows.
function applyTheme(): void {
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem("theme");
  } catch {
    // default to system below
  }
  const dark =
    stored === "dark" ||
    ((stored === null || stored === "system") &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.classList.toggle("light", !dark);
}

applyTheme();
window.addEventListener("storage", applyTheme);
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", applyTheme);

initializeLocale();

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

createRoot(rootElement).render(<AskApp />);
