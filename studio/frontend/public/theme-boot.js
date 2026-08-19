// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Apply the stored theme and palette before the bundle loads so the first
// paint is never the wrong mode. Loaded as an external classic script (it
// blocks parsing, like an inline script) because the backend CSP only
// allows script-src 'self'.
try {
  // Storage reads get their own guards so a blocked localStorage (private
  // browsing) still resolves a mode from the OS preference.
  var theme = "system";
  var palette = null;
  try {
    theme = localStorage.getItem("theme") || "system";
    palette = localStorage.getItem("palette");
  } catch (e) {}
  var dark =
    theme === "dark" ||
    (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  var root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", !dark);
  root.style.colorScheme = dark ? "dark" : "light";
  if (palette === "classic" || palette === "minimal") {
    root.setAttribute("data-palette", palette);
  }
} catch (e) {}
