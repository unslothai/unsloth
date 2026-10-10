import { isTauri } from "@/lib/api-base";

type InAppLinkHandler = (url: string) => boolean;

let inAppLinkHandler: InAppLinkHandler | null = null;

/** Routes web links to the chat's browser panel. Returning false falls back to the system browser. */
export function setInAppLinkHandler(handler: InAppLinkHandler | null): void {
  inAppLinkHandler = handler;
}

// Other schemes (javascript:, data:, file:) are unsafe to open.
const EXTERNAL_SCHEMES = /^(https?|mailto):/i;

/** Open a URL in the system browser (Tauri) or a new tab (web). */
export function openExternalLink(url: string): void {
  if (!EXTERNAL_SCHEMES.test(url.trim())) return;
  if (isTauri) {
    import("@tauri-apps/plugin-opener").then(({ openUrl }) => {
      openUrl(url).catch(console.error);
    });
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}

/** Open in the browser panel if available, else system browser / new tab. True = caller should
 *  preventDefault; false lets native navigation proceed (relative, empty). */
export function openLink(url: string): boolean {
  if (!url) return false;
  // The URL parser drops tabs and newlines, so `https:\t//host` navigates like `https://host`.
  url = url.replace(/[\t\n\r]/g, "");

  // `https:host` (no slashes) is absolute too: open it as a web link instead of letting native navigation replace
  // Studio's page.
  if (/^https?:/i.test(url) && !url.includes("://")) {
    try {
      url = new URL(url).href;
    } catch {
      return true;
    }
  }

  // Anchor links scroll within the page, don't open externally
  if (url.startsWith("#")) {
    window.location.hash = url;
    return true;
  }

  // Relative URLs: let the browser / router handle them natively
  if (!url.includes("://") && !url.startsWith("mailto:")) {
    return false;
  }

  if (/^https?:\/\//i.test(url) && inAppLinkHandler?.(url)) {
    return true;
  }

  openExternalLink(url);
  return true;
}
