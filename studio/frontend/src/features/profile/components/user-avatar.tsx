// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { avatarBgStyle, initialsFromName } from "../utils/avatar-initials";

type UserAvatarProps = {
  name: string;
  imageUrl: string | null;
  size: "sm" | "md" | "lg";
  className?: string;
};

const SIZE: Record<"sm" | "md" | "lg", string> = {
  sm: "size-9 text-xs",
  md: "size-11 text-sm",
  /** ~10% larger than `size-24` / `text-2xl` for the edit-profile dialog. */
  lg: "size-[106px] text-[1.65rem]",
};

export function UserAvatar({ name, imageUrl, size, className }: UserAvatarProps) {
  const label = initialsFromName(name);

  if (imageUrl) {
    return (
      <span className={cn("relative inline-flex shrink-0 overflow-hidden rounded-full", SIZE[size], className)}>
        <img src={imageUrl} alt="" className="size-full object-cover" />
      </span>
    );
  }

  return (
    <span
      style={avatarBgStyle()}
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-full font-semibold text-white",
        SIZE[size],
        className,
      )}
      aria-hidden
    >
      {label}
    </span>
  );
}
