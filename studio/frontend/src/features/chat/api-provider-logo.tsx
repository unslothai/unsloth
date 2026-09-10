// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { DashboardSquare01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { isCustomProviderType } from "./external-providers";
import { providerLogoPath } from "./provider-logo-path";

export function apiProviderLogoSrc(
  providerType: string | undefined | null,
): string | undefined {
  const path = providerLogoPath(providerType);
  return path ? `${import.meta.env.BASE_URL}${path.slice(1)}` : undefined;
}

interface ApiProviderLogoProps {
  providerType: string | undefined | null;
  className?: string;
  title?: string;
}

const DARK_INVERT_LOGOS = new Set(["openai", "openai_codex", "ollama", "openrouter"]);

/** Shared hub or connection logo; monochrome ones invert in dark mode. */
export function ApiProviderLogo({ providerType, className, title }: ApiProviderLogoProps) {
  const src = apiProviderLogoSrc(providerType);
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
        providerType && DARK_INVERT_LOGOS.has(providerType) && "dark:invert",
        className,
      )}
    />
  );
}
