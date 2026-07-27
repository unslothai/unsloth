// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { avatarBgStyle, initialsFromName } from "../utils/avatar-initials";
import {
  useUserProfileStore,
  type AvatarShape,
} from "../stores/user-profile-store";

type UserAvatarProps = {
  name: string;
  imageUrl: string | null;
  size: "sm" | "md" | "lg";
  className?: string;
  /** Override the stored shape preference (defaults to the user's setting). */
  shape?: AvatarShape;
};

const SIZE: Record<"sm" | "md" | "lg", string> = {
  sm: "size-9 text-xs",
  md: "size-11 text-sm",
  /** ~10% larger than `size-24` / `text-2xl` for the edit-profile dialog. */
  lg: "size-[106px] text-[calc(1.65rem*var(--ui-font-scale,1))]",
};

// Percentage radius keeps the rounded-rectangle proportional across sizes.
const SHAPE: Record<AvatarShape, string> = {
  circle: "rounded-full",
  rounded: "rounded-[22%]",
};

export function UserAvatar({ name, imageUrl, size, className, shape }: UserAvatarProps) {
  const label = initialsFromName(name);
  const storedShape = useUserProfileStore((s) => s.avatarShape);
  const shapeClass = SHAPE[shape ?? storedShape];

  if (imageUrl) {
    return (
      <span className={cn("relative inline-flex shrink-0 overflow-hidden bg-transparent", shapeClass, SIZE[size], className)}>
        <img src={imageUrl} alt="" className="size-full object-cover" />
      </span>
    );
  }

  return (
    <span
      style={avatarBgStyle()}
      className={cn(
        "inline-flex shrink-0 items-center justify-center font-semibold",
        shapeClass,
        SIZE[size],
        className,
      )}
      aria-hidden
    >
      {label}
    </span>
  );
}
