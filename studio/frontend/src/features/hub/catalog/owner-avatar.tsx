// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { ownerPaletteColor } from "../lib/avatar-theme";
import {
  expireHfOwnerAvatarUrl,
  useHfOwnerAvatar,
} from "../lib/hf-owner-avatar";
import {
  type ProviderLogo,
  resolveOwnerProviderLogo,
} from "../lib/provider-logos";

type AvatarSize = "xs" | "sm" | "md" | "lg";

const SIZES: Record<AvatarSize, string> = {
  xs: "size-5 rounded-sm text-ui-9",
  sm: "size-7 rounded-sm text-ui-11",
  md: "size-9 rounded-md text-ui-13",
  lg: "size-12 rounded-lg text-ui-16",
};

const UNSLOTH_OWNER_LOGO: ProviderLogo = {
  id: "unsloth",
  name: "Unsloth",
  logoPath: "/rounded.png",
  treatment: "original",
  background: "transparent",
  fit: "cover",
  prefixes: [],
};

function isUnslothOwner(owner: string): boolean {
  return owner.trim().toLowerCase() === "unsloth";
}

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

/** Hides a URL whose <img> failed, then retries it with backoff instead of for the mount's life. */
export function useAvatarImageRetry(url: string | null): {
  show: boolean;
  onLoad: () => void;
  onError: () => void;
} {
  const [failedImage, setFailedImage] = useState<FailedAvatarImage | null>(
    null,
  );
  const blocked =
    Boolean(url) && failedImage?.url === url && !failedImage.retryReady;

  useEffect(() => {
    const activeFailure = url && failedImage?.url === url ? failedImage : null;
    if (!activeFailure || activeFailure.retryReady) {
      return;
    }
    const timer = globalThis.setTimeout(() => {
      setFailedImage((current) =>
        current?.url === url && !current.retryReady
          ? { ...current, retryReady: true }
          : current,
      );
    }, avatarImageRetryDelay(activeFailure.failures));
    return () => globalThis.clearTimeout(timer);
  }, [url, failedImage]);

  return {
    show: Boolean(url) && !blocked,
    onLoad: () => setFailedImage(null),
    onError: () => {
      if (!url) return;
      expireHfOwnerAvatarUrl(url);
      setFailedImage((current) => ({
        url,
        failures: current?.url === url ? current.failures + 1 : 1,
        retryReady: false,
      }));
    },
  };
}

export function OwnerAvatar({
  owner,
  repoName,
  size = "md",
  className,
  remote = true,
}: {
  owner: string;
  /** Repo name after `owner/`; for eligible owners renders the upstream provider's logo. */
  repoName?: string;
  size?: AvatarSize;
  className?: string;
  /** False skips the HF profile fetch; virtualized rows pass false to avoid request storms. */
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

  if (isUnslothOwner(owner)) {
    return (
      <ProviderLogoTile
        provider={UNSLOTH_OWNER_LOGO}
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
  remote = true,
): string | null {
  const provider = resolveOwnerProviderLogo(owner, repoName);
  const isUnsloth = isUnslothOwner(owner);
  const remoteUrl = useHfOwnerAvatar(
    owner.trim() || "?",
    remote && provider == null && !isUnsloth,
  );
  if (provider) {
    return provider.treatment === "original" ? provider.logoPath : null;
  }
  return isUnsloth ? UNSLOTH_OWNER_LOGO.logoPath : remoteUrl;
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
  const hasTileBg = provider.background === "white";
  const bgClass = hasTileBg ? "bg-white" : "bg-transparent";

  if (provider.treatment === "original") {
    const fit = provider.fit ?? "contain";
    return (
      <span
        role="img"
        aria-label={altLabel}
        className={cn(
          hasTileBg && "hub-avatar-tile",
          "inline-flex shrink-0 items-center justify-center overflow-hidden",
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

  const colorClass =
    provider.treatment === "mono-black" ? "text-black" : "text-foreground";
  return (
    <span
      role="img"
      aria-label={altLabel}
      className={cn(
        hasTileBg && "hub-avatar-tile",
        "inline-flex shrink-0 items-center justify-center overflow-hidden",
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
  const image = useAvatarImageRetry(remoteUrl);
  const showImage = image.show;

  return (
    <span
      style={showImage ? undefined : { backgroundColor: color }}
      className={cn(
        !showImage && "hub-avatar-tile",
        "inline-flex shrink-0 items-center justify-center overflow-hidden font-semibold text-white",
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
          onLoad={image.onLoad}
          onError={image.onError}
        />
      ) : (
        initial
      )}
    </span>
  );
}
