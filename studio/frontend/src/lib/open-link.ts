import { isTauri } from "@/lib/api-base";

type InAppLinkHandler = (url: string) => boolean;

let inAppLinkHandler: InAppLinkHandler | null = null;

export function setInAppLinkHandler(handler: InAppLinkHandler | null): void {
  inAppLinkHandler = handler;
}

// Other schemes (javascript:, data:, file:) are unsafe to open.
const EXTERNAL_SCHEMES = /^(https?|mailto):/i;

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

/** True means the caller should preventDefault. */
export function openLink(url: string): boolean {
  if (!url) return false;
  // The URL parser drops tabs and newlines, so `https:\t//host` navigates like `https://host`.
  url = url.replace(/[\t\n\r]/g, "");
  // It also strips leading and trailing C0 controls and spaces, so ` //host` is another site too.
  let start = 0;
  let end = url.length;
  while (start < end && url.charCodeAt(start) <= 0x20) start++;
  while (end > start && url.charCodeAt(end - 1) <= 0x20) end--;
  url = url.slice(start, end);

  // `https:host` (no slashes) is absolute too: open it as a web link instead of letting native navigation replace
  // Studio's page.
  if (/^https?:/i.test(url) && !url.includes("://")) {
    try {
      url = new URL(url).href;
    } catch {
      return true;
    }
  }

  // `//host/path` (or with backslashes, which browsers read the same way) is another site, not a path in Studio:
  // native navigation would replace Studio's page (the Desktop main window) with it.
  if (/^[\\/]{2}/.test(url)) {
    try {
      url = new URL(`https:${url}`).href;
    } catch {
      return true;
    }
  }

  // Anchor links scroll within the page, don't open externally
  if (url.startsWith("#")) {
    window.location.hash = url;
    return true;
  }

  if (!url.includes("://") && !url.startsWith("mailto:")) {
    return false;
  }

  if (/^https?:\/\//i.test(url) && inAppLinkHandler?.(url)) {
    return true;
  }

  openExternalLink(url);
  return true;
}
