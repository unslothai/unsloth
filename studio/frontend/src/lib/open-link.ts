import { isTauri } from "@/lib/api-base";

/**
 * Open a URL in the system browser (Tauri) or new tab (web).
 * Handles anchor links and mailto: natively without the opener plugin.
 */
export function openLink(url: string) {
  if (!url) return;

  // Anchor links — scroll within the page, don't open externally
  if (url.startsWith("#")) {
    window.location.hash = url;
    return;
  }

  // Relative URLs — ignore in Tauri (no meaningful navigation target)
  if (!url.includes("://") && !url.startsWith("mailto:")) {
    return;
  }

  if (isTauri) {
    import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
      openUrl(url).catch(console.error);
    });
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}
