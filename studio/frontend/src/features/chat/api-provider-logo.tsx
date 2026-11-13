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
 * A letter in its circle, centred on its ink: flex centres the glyph's advance and line box, which
 * leave a capital high and its side bearings off by a pixel in most fonts.
 */
function LetterTile({
  letter,
  name,
  title,
  className,
}: { letter: string; name: string; title?: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [nudge, setNudge] = useState<{ x: number; y: number } | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    const ctx = el && document.createElement("canvas").getContext("2d");
    if (!(el && ctx)) {
      return;
    }
    const style = getComputedStyle(el);
    ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    const m = ctx.measureText(letter);
    setNudge({
      x: (m.width - m.actualBoundingBoxRight + m.actualBoundingBoxLeft) / 2,
      y:
        (m.fontBoundingBoxAscent -
          m.fontBoundingBoxDescent -
          (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent)) /
        2,
    });
  }, [letter]);
  return (
    <span
      ref={ref}
      title={title}
      aria-hidden
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-full font-semibold leading-none text-white",
        className,
      )}
      style={{ backgroundColor: letterTileColor(name), fontSize: "0.62em" }}
    >
      <span
        style={
          nudge ? { transform: `translate(${nudge.x}px, ${nudge.y}px)` } : undefined
        }
      >
        {letter}
      </span>
    </span>
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
