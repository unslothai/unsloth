// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { CSSProperties, MouseEvent } from "react";
import { initials, nameHue, useImageUrl, useIsCutout } from "../images";

type PortraitProps = {
  kind?: "character" | "persona";
  id: string | null | undefined;
  name: string;
  hasImage: boolean;
  rev: number;
  className?: string;
  /** "avatar" crops to the face area (top), "card" fills a portrait frame. */
  variant?: "avatar" | "card";
  onClick?: (event: MouseEvent<HTMLElement>) => void;
  label?: string;
  imageUrl?: string | null;
  dimmed?: boolean;
  /** A picture on a transparent background shows whole, standing on the frame's bottom, with no tile behind it. */
  cutout?: boolean;
};

/** A character's or persona's picture, or a tinted monogram when there is none. */
export function Portrait({
  kind = "character",
  id,
  name,
  hasImage,
  rev,
  className,
  variant = "avatar",
  onClick,
  label,
  imageUrl,
  dimmed,
  cutout: allowCutout = false,
}: PortraitProps) {
  const loaded = useImageUrl(kind, id, rev, hasImage && imageUrl === undefined);
  const url = imageUrl === undefined ? loaded : imageUrl;
  const cutout = useIsCutout(url, allowCutout);
  const hue = nameHue(name);
  const style: CSSProperties = {
    background: `linear-gradient(150deg, hsl(${hue} 42% 62%), hsl(${(hue + 28) % 360} 46% 40%))`,
  };
  const content = url ? (
    <img
      src={url}
      alt=""
      draggable={false}
      className={cn(
        "size-full",
        cutout ? "object-contain object-bottom" : "object-cover",
        !cutout && (variant === "avatar" ? "object-[50%_18%]" : "object-[50%_22%]"),
      )}
    />
  ) : (
    <span
      aria-hidden={true}
      className={cn(
        "flex size-full select-none items-center justify-center font-heading font-semibold leading-none text-white/95",
        // Sized by the frame (a container), so a monogram fills a 20px dot and a full card alike.
        variant === "card" ? "text-[30cqw]" : "text-[40cqw]",
      )}
    >
      {initials(name)}
    </span>
  );
  const classes = cn(
    "relative block shrink-0 overflow-hidden transition-[filter,opacity] [container-type:inline-size]",
    !cutout && "bg-muted",
    dimmed && "opacity-55 grayscale-[35%]",
    className,
  );
  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        aria-label={label ?? name}
        className={cn(
          classes,
          "cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
        )}
        style={url ? undefined : style}
        data-cutout={cutout || undefined}
      >
        {content}
      </button>
    );
  }
  return (
    <span
      className={classes}
      style={url ? undefined : style}
      aria-label={label}
      role={label ? "img" : undefined}
      data-cutout={cutout || undefined}
    >
      {content}
    </span>
  );
}
