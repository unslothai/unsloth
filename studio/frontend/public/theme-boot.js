// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Apply the stored theme before first paint. External classic script since CSP is script-src 'self'.
try {
  // Guarded reads so blocked localStorage still falls back to the OS preference.
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
  // Keep in sync with COLOR_THEME_IDS ("standard" sets no attribute).
  var palettes = [
    "classic", "minimal", "blueberry", "butterfly-pea", "cherry",
    "cinnamon", "cotton-candy", "dragon-fruit", "earl-grey", "espresso",
    "honey", "licorice", "macaron", "matcha", "mint", "neon-cyberpunk",
    "oat-milk", "peach", "pina-paraiso", "plum", "tangerine", "taro",
    "wasabi", "yuzu",
  ];
  if (palettes.indexOf(palette) !== -1) {
    root.setAttribute("data-palette", palette);
  }
} catch (e) {}
