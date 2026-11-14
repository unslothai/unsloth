// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { DashboardSquare01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useLayoutEffect, useRef, useState } from "react";
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
  /** The connection's name: a type with no logo shows its first letter on a colour from it. */
  name?: string;
}

// Mid-tone hues that hold white text in light and dark themes alike.
const LETTER_TILE_COLORS = [
  "#e5484d",
  "#f76b15",
  "#d6a800",
  "#30a46c",
  "#12a594",
  "#0090ff",
  "#6e56cf",
  "#d6409f",
];

function letterTileColor(name: string): string {
  let hash = 0;
  for (const ch of name) {
    hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  }
  return LETTER_TILE_COLORS[Math.abs(hash) % LETTER_TILE_COLORS.length];
}

const DARK_INVERT_LOGOS = new Set(["openai", "openai_codex", "ollama", "liquid", "typesafe"]);
// Pale brand marks get a hairline edge on light surfaces instead of inverting.
const LIGHT_OUTLINE_LOGOS = new Set(["openrouter"]);

/**
 * A letter in its circle, painted on a canvas and placed by its measured ink: HTML and SVG text
 * centre the glyph's advance and snap it to device pixels, which left letters a pixel off.
 */
function LetterTile({
  letter,
  name,
  title,
  className,
}: { letter: string; name: string; title?: string; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const color = letterTileColor(name);
  useLayoutEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!(canvas && ctx)) {
      return;
    }
    // Layout size; a CSS scale on an ancestor only raises the backing resolution.
    const size = canvas.offsetWidth;
    const dpr = (window.devicePixelRatio || 1) * (canvas.getBoundingClientRect().width / size || 1);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = `600 ${size * 0.54}px ${getComputedStyle(canvas).fontFamily}`;
    const m = ctx.measureText(letter);
    ctx.fillStyle = "#fff";
    ctx.fillText(
      letter,
      size / 2 - (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2,
      size / 2 + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2,
    );
  }, [letter, color]);
  return (
    <canvas
      ref={ref}
      title={title}
      aria-hidden
      className={cn("shrink-0 rounded-full", className)}
    />
  );
}

export function ApiProviderLogo({ providerType, className, title, name }: ApiProviderLogoProps) {
  const src = apiProviderLogoSrc(providerType);
  const letter = name?.trim().charAt(0).toUpperCase();
  if (!src && letter) {
    return (
      <LetterTile
        letter={letter}
        name={name ?? ""}
        title={title}
        className={className}
      />
    );
  }
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
        providerType &&
          LIGHT_OUTLINE_LOGOS.has(providerType) &&
          "drop-shadow-[0_0_0.5px_rgb(0_0_0/0.7)] dark:drop-shadow-none",
        className,
      )}
    />
  );
}
