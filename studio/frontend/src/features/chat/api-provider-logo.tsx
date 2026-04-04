// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";

/**
 * Registry logos live at `public/provider-logos/{provider_type}.{ext}` where `provider_type`
 * matches `PROVIDER_REGISTRY` keys exactly (lowercase). Extension varies by asset (svg preferred).
 */
const PROVIDER_LOGO_EXT: Record<string, "svg" | "png" | "jpg"> = {
  openai: "svg",
  mistral: "svg",
  gemini: "svg",
  anthropic: "svg",
  deepseek: "svg",
  huggingface: "svg",
  kimi: "jpg",
  qwen: "png",
  openrouter: "svg",
};

export function apiProviderLogoSrc(
  providerType: string | undefined | null,
): string | undefined {
  if (!providerType) return undefined;
  const ext = PROVIDER_LOGO_EXT[providerType];
  if (!ext) return undefined;
  return `${import.meta.env.BASE_URL}provider-logos/${providerType}.${ext}`;
}

interface ApiProviderLogoProps {
  providerType: string | undefined | null;
  className?: string;
  title?: string;
}

/**
 * Renders the logo for a registry provider type when `provider_type.{ext}` exists under
 * `public/provider-logos/`.
 * OpenAI's asset is black-on-transparent; it is inverted in dark mode for contrast.
 */
export function ApiProviderLogo({ providerType, className, title }: ApiProviderLogoProps) {
  const src = apiProviderLogoSrc(providerType);
  if (!src) return null;
  return (
    <img
      src={src}
      alt=""
      title={title}
      aria-hidden
      className={cn(
        "shrink-0 object-contain",
        providerType === "openai" && "dark:invert",
        className,
      )}
    />
  );
}
