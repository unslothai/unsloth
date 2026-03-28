import { isTauri } from "@/lib/api-base";

/**
 * Open a URL in the system browser (Tauri) or new tab (web).
 * Use this instead of target="_blank" so links work in the desktop app.
 */
export function openLink(url: string) {
  if (!url) return;
  if (isTauri) {
    import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
      openUrl(url);
    });
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}
