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
