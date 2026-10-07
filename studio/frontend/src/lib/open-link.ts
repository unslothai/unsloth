import { isTauri } from "@/lib/api-base";

type InAppLinkHandler = (url: string) => boolean;

let inAppLinkHandler: InAppLinkHandler | null = null;

/** Routes web links to the chat's browser panel. Returning false falls back to the system browser. */
export function setInAppLinkHandler(handler: InAppLinkHandler | null): void {
  inAppLinkHandler = handler;
}

// Other schemes (javascript:, data:, file:) are unsafe to open.
const EXTERNAL_SCHEMES = /^(https?|mailto):/i;
const PROTOCOL_RELATIVE = /^[/\\]{2}/;
const LEADING_SLASHES = /^[/\\]+/;

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

  // Anchor links scroll within the page, don't open externally
  if (url.startsWith("#")) {
    window.location.hash = url;
    return true;
  }

  // Protocol-relative ("//host", "/\host") names another site, not an app route: native
  // navigation would replace the Desktop main window, so open it like any https link.
  if (PROTOCOL_RELATIVE.test(url)) {
    return openLink(`https://${url.replace(LEADING_SLASHES, "")}`);
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
