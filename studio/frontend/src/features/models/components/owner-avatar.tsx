// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useState } from "react";
import { useHfOwnerAvatar } from "../lib/hf-owner-avatar";

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

const SIZES: Record<"xs" | "sm" | "md" | "lg", string> = {
  xs: "size-5 rounded-[6px] text-[9px]",
  sm: "size-7 rounded-[8px] text-[11px]",
  md: "size-9 rounded-[10px] text-[13px]",
  lg: "size-12 rounded-[13px] text-[16px]",
};

export function OwnerAvatar({
  owner,
  size = "md",
  className,
}: {
  owner: string;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
}) {
  const owned = owner.trim() || "?";
  const initial = owned[0]?.toUpperCase() ?? "?";
  const color = PALETTE[hashOwner(owned) % PALETTE.length];
  const remoteUrl = useHfOwnerAvatar(owned);
  const [errored, setErrored] = useState(false);

  const showImage = Boolean(remoteUrl) && !errored;

  return (
    <span
      style={showImage ? undefined : { backgroundColor: color }}
      className={cn(
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
          loading="lazy"
          className="size-full object-cover"
          onError={() => setErrored(true)}
        />
      ) : (
        initial
      )}
    </span>
  );
}
