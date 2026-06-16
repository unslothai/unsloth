// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { ownerPaletteColor } from "../lib/avatar-theme";
import { useHfOwnerAvatar } from "../lib/hf-owner-avatar";
import {
  type ProviderLogo,
  resolveOwnerProviderLogo,
} from "../lib/provider-logos";

type AvatarSize = "xs" | "sm" | "md" | "lg";

const SIZES: Record<AvatarSize, string> = {
  xs: "size-5 rounded-[8px] text-[9px]",
  sm: "size-7 rounded-[10px] text-[11px]",
  md: "size-9 rounded-[12px] text-[13px]",
  lg: "size-12 rounded-[15px] text-[16px]",
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
  remote = true,
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
  /**
   * When false, the fallback avatar never fetches the owner's HF profile
   * picture — it shows a locally-resolved provider logo or the colored-initial
   * tile instantly. Virtualized list rows pass `false` so browsing the whole
   * Hub (every row a different owner) doesn't trigger a per-row request storm.
   * The single-model inspector keeps the default (`true`).
   */
  remote?: boolean;
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
  return (
    <DefaultAvatar
      owner={owner}
      size={size}
      className={className}
      remote={remote}
    />
  );
}

export function useAvatarImageUrl(
  owner: string,
  repoName?: string,
): string | null {
  const provider = resolveOwnerProviderLogo(owner, repoName);
  const remoteUrl = useHfOwnerAvatar(owner.trim() || "?");
  if (provider) {
    return provider.treatment === "original" ? provider.logoPath : null;
  }
  return remoteUrl;
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
          decoding="async"
          className={cn(
            fit === "cover"
              ? "size-full object-cover"
              : "size-3/4 object-contain",
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
  remote = true,
}: {
  owner: string;
  size: AvatarSize;
  className?: string;
  remote?: boolean;
}) {
  const owned = owner.trim() || "?";
  const initial = owned[0]?.toUpperCase() ?? "?";
  const color = ownerPaletteColor(owned);
  const remoteUrl = useHfOwnerAvatar(owned, remote);
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
          decoding="async"
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
