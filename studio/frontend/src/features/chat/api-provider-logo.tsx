// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { DashboardSquare01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { isCustomProviderType } from "./external-providers";
import {
  isDarkMonochromeIcon,
  resolveApiProviderLogoSrc,
} from "./provider-logo-source";

interface ApiProviderLogoProps {
  providerType: string | undefined | null;
  className?: string;
  title?: string;
  /** Opt-in custom-provider site icon; must be a backend-validated image data URL. */
  faviconDataUrl?: string | null | undefined;
}

const DARK_INVERT_LOGOS = new Set(["openai", "openai_codex", "ollama", "liquid", "typesafe"]);
// Pale brand marks get a hairline edge on light surfaces instead of inverting.
const LIGHT_OUTLINE_LOGOS = new Set(["openrouter"]);

// Each stored site icon is analysed once per session.
const darkMonochromeCache = new Map<string, boolean>();

function useDarkMonochromeIcon(dataUrl: string | undefined): boolean {
  const [, rerender] = useState(0);
  useEffect(() => {
    if (!dataUrl || darkMonochromeCache.has(dataUrl)) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      let dark = false;
      try {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 32;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(img, 0, 0, 32, 32);
          dark = isDarkMonochromeIcon(ctx.getImageData(0, 0, 32, 32).data);
        }
      } catch {
        // No canvas: leave the icon as is.
      }
      darkMonochromeCache.set(dataUrl, dark);
      if (!cancelled) rerender((n) => n + 1);
    };
    img.onerror = () => darkMonochromeCache.set(dataUrl, false);
    img.src = dataUrl;
    return () => {
      cancelled = true;
    };
  }, [dataUrl]);
  return dataUrl ? darkMonochromeCache.get(dataUrl) === true : false;
}

export function ApiProviderLogo({
  providerType,
  className,
  title,
  faviconDataUrl,
}: ApiProviderLogoProps) {
  const src = resolveApiProviderLogoSrc(
    providerType,
    faviconDataUrl,
    import.meta.env.BASE_URL,
  );
  // Fetched site icons only get dark-mode treatment when analysed as a dark monochrome glyph.
  const isSiteIcon = src !== undefined && src === faviconDataUrl;
  // A black site glyph vanishes on dark surfaces; invert it like the preset marks.
  const invertSiteIcon = useDarkMonochromeIcon(isSiteIcon ? src : undefined);
  if (!src && isCustomProviderType(providerType)) {
    return (
      <span title={title} aria-hidden className="inline-flex shrink-0">
        <HugeiconsIcon icon={DashboardSquare01Icon} className={cn("shrink-0", className)} />
      </span>
    );
  }

  if (!src) return null;
  return (
    <img
      src={src}
      alt=""
      title={title}
      aria-hidden
      className={cn(
        "shrink-0 object-contain",
        invertSiteIcon && "dark:invert",
        !isSiteIcon && providerType && DARK_INVERT_LOGOS.has(providerType) && "dark:invert",
        !isSiteIcon &&
          providerType &&
          LIGHT_OUTLINE_LOGOS.has(providerType) &&
          "drop-shadow-[0_0_0.5px_rgb(0_0_0/0.7)] dark:drop-shadow-none",
        className,
      )}
    />
  );
}
