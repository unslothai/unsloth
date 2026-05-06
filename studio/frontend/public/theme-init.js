// Apply theme class before the bundle loads to avoid a flash of the wrong
// theme. Mirrors the resolution logic in features/settings/stores/theme-store.ts.
// Lives outside index.html as a same-origin file so Tauri's default-src 'self'
// CSP allows it (inline scripts are blocked).
(() => {
  let stored = null;
  try {
    stored = localStorage.getItem("theme");
  } catch {}
  try {
    const isDark =
      stored === "dark" ||
      (stored !== "light" &&
        globalThis.matchMedia("(prefers-color-scheme: dark)").matches);
    if (isDark) document.documentElement.classList.add("dark");
  } catch {}
})();
