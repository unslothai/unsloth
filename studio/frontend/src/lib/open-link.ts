import { isTauri } from "@/lib/api-base";

/**
 * Open a URL in the system browser (Tauri) or new tab (web).
 * Handles anchor links and mailto: natively without the opener plugin.
 * Returns true when the caller should preventDefault; false when the
 * browser's native navigation should proceed (relative URLs, empty, etc.).
 */
export function openLink(url: string): boolean {
  if (!url) return false;

  // Anchor links — scroll within the page, don't open externally
  if (url.startsWith("#")) {
    window.location.hash = url;
    return true;
  }

  // Relative URLs — let the browser / router handle them natively
  if (!url.includes("://") && !url.startsWith("mailto:")) {
    return false;
  }

  if (isTauri) {
    import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
      openUrl(url).catch(console.error);
    });
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
  return true;
}
