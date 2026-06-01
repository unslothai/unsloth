// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { useHfOwnerAvatar } from "../lib/hf-owner-avatar";
import {
  type ProviderLogo,
  resolveOwnerProviderLogo,
} from "../lib/provider-logos";

const PALETTE = [
  "hsl(214 42% 42%)",
  "hsl(172 32% 36%)",
  "hsl(30 42% 42%)",
  "hsl(348 36% 44%)",
  "hsl(198 38% 40%)",
  "hsl(140 26% 38%)",
  "hsl(222 24% 46%)",
  "hsl(16 34% 44%)",
  "hsl(260 22% 48%)",
  "hsl(44 44% 40%)",
];

function hashOwner(owner: string): number {
  let h = 0;
  for (let i = 0; i < owner.length; i++) {
    h = (h * 31 + owner.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

type AvatarSize = "xs" | "sm" | "md" | "lg";

const SIZES: Record<AvatarSize, string> = {
  xs: "size-5 rounded-[6px] text-[9px]",
  sm: "size-7 rounded-[8px] text-[11px]",
  md: "size-9 rounded-[10px] text-[13px]",
  lg: "size-12 rounded-[13px] text-[16px]",
};

const AVATAR_IMAGE_RETRY_BASE_MS = 60_000;
const AVATAR_IMAGE_RETRY_MAX_MS = 30 * 60_000;

interface FailedAvatarImage {
  url: string;
  failures: number;
  retryReady: boolean;
}

function avatarImageRetryDelay(failures: number): number {
  return Math.min(
    AVATAR_IMAGE_RETRY_BASE_MS * 2 ** Math.max(failures - 1, 0),
    AVATAR_IMAGE_RETRY_MAX_MS,
  );
}

export function OwnerAvatar({
  owner,
  repoName,
  size = "md",
  className,
}: {
  owner: string;
  /**
   * Repo name (part after `owner/`). When provided alongside an eligible
   * owner (currently "unsloth"), the avatar tries to match a known upstream
   * provider from the registry and renders that provider's logo instead of
   * the owner's HF profile picture. Used so an Unsloth re-upload of e.g.
   * Qwen2.5-7B shows the Qwen logo while still labeling the owner "unsloth".
   */
  repoName?: string;
  size?: AvatarSize;
  className?: string;
}) {
  const providerLogo = resolveOwnerProviderLogo(owner, repoName);
  if (providerLogo) {
    return (
      <ProviderLogoTile
        provider={providerLogo}
        size={size}
        className={className}
      />
    );
  }
  return <DefaultAvatar owner={owner} size={size} className={className} />;
}

function ProviderLogoTile({
  provider,
  size,
  className,
}: {
  provider: ProviderLogo;
  size: AvatarSize;
  className?: string;
}) {
  const altLabel = `${provider.name} logo`;
  const bgClass =
    provider.background === "white" ? "bg-white" : "bg-transparent";

  if (provider.treatment === "original") {
    const fit = provider.fit ?? "contain";
    return (
      <span
        role="img"
        aria-label={altLabel}
        className={cn(
          "hub-avatar-tile inline-flex shrink-0 items-center justify-center overflow-hidden",
          bgClass,
          SIZES[size],
          className,
        )}
      >
        <img
          src={provider.logoPath}
          alt=""
          loading="lazy"
          className={cn(
            fit === "cover" ? "size-full object-cover" : "size-3/4 object-contain",
          )}
        />
      </span>
    );
  }

  // mono treatments paint a silhouette via CSS mask; the host element's
  // text color drives the silhouette color (theme-aware vs. fixed black).
  const colorClass =
    provider.treatment === "mono-black" ? "text-black" : "text-foreground";
  return (
    <span
      role="img"
      aria-label={altLabel}
      className={cn(
        "hub-avatar-tile inline-flex shrink-0 items-center justify-center overflow-hidden",
        bgClass,
        colorClass,
        SIZES[size],
        className,
      )}
    >
      <span
        aria-hidden="true"
        className="size-3/4 bg-current"
        style={{
          mask: `url(${provider.logoPath}) center / contain no-repeat`,
          WebkitMask: `url(${provider.logoPath}) center / contain no-repeat`,
        }}
      />
    </span>
  );
}

function DefaultAvatar({
  owner,
  size,
  className,
}: {
  owner: string;
  size: AvatarSize;
  className?: string;
}) {
  const owned = owner.trim() || "?";
  const initial = owned[0]?.toUpperCase() ?? "?";
  const color = PALETTE[hashOwner(owned) % PALETTE.length];
  const remoteUrl = useHfOwnerAvatar(owned);
  const [failedImage, setFailedImage] = useState<FailedAvatarImage | null>(
    null,
  );
  const imageBlocked =
    Boolean(remoteUrl) &&
    failedImage?.url === remoteUrl &&
    !failedImage.retryReady;
  const showImage = Boolean(remoteUrl) && !imageBlocked;

  useEffect(() => {
    const activeFailure =
      remoteUrl && failedImage?.url === remoteUrl ? failedImage : null;
    if (!activeFailure || activeFailure.retryReady) {
      return;
    }

    const timer = globalThis.setTimeout(() => {
      setFailedImage((current) =>
        current?.url === remoteUrl && !current.retryReady
          ? { ...current, retryReady: true }
          : current,
      );
    }, avatarImageRetryDelay(activeFailure.failures));
    return () => globalThis.clearTimeout(timer);
  }, [remoteUrl, failedImage]);

  return (
    <span
      style={showImage ? undefined : { backgroundColor: color }}
      className={cn(
        "hub-avatar-tile inline-flex shrink-0 items-center justify-center overflow-hidden font-semibold text-white",
        SIZES[size],
        className,
      )}
      aria-hidden="true"
    >
      {showImage ? (
        <img
          src={remoteUrl ?? ""}
          alt=""
          loading="lazy"
          className="size-full object-cover"
          onLoad={() => setFailedImage(null)}
          onError={() => {
            if (!remoteUrl) {
              return;
            }
            setFailedImage((current) => ({
              url: remoteUrl,
              failures: current?.url === remoteUrl ? current.failures + 1 : 1,
              retryReady: false,
            }));
          }}
        />
      ) : (
        initial
      )}
    </span>
  );
}
