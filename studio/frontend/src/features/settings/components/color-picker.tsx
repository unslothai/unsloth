// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ColorPickerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";

/* ------------------------- HSV ↔ hex conversions ------------------------- */

type Hsv = { h: number; s: number; v: number };

function hexToHsv(hex: string): Hsv {
  const r = Number.parseInt(hex.slice(1, 3), 16) / 255;
  const g = Number.parseInt(hex.slice(3, 5), 16) / 255;
  const b = Number.parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s: max === 0 ? 0 : d / max, v: max };
}

function hsvToHex({ h, s, v }: Hsv): string {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let rgb: [number, number, number];
  if (h < 60) rgb = [c, x, 0];
  else if (h < 120) rgb = [x, c, 0];
  else if (h < 180) rgb = [0, c, x];
  else if (h < 240) rgb = [0, x, c];
  else if (h < 300) rgb = [x, 0, c];
  else rgb = [c, 0, x];
  const channel = (value: number) =>
    Math.round((value + m) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${channel(rgb[0])}${channel(rgb[1])}${channel(rgb[2])}`;
}

const HEX_PATTERN = /^#?([0-9a-fA-F]{6})$/;

function isLightColor(hex: string): boolean {
  const r = Number.parseInt(hex.slice(1, 3), 16);
  const g = Number.parseInt(hex.slice(3, 5), 16);
  const b = Number.parseInt(hex.slice(5, 7), 16);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 > 0.62;
}

/* ------------------------------- Component ------------------------------- */

type EyeDropperResult = { sRGBHex: string };
type EyeDropperConstructor = new () => {
  open: () => Promise<EyeDropperResult>;
};

/**
 * In-app color picker in a Popover: saturation/value area, hue slider, hex
 * field, and (where supported) a screen eyedropper. Replaces the native
 * <input type="color"> so no OS-level color panel is left dangling when the
 * user clicks away; the popover dismisses like any other popup.
 */
export function ColorPickerSwatch({
  value,
  onChange,
  label,
}: {
  value: string;
  onChange: (hex: string) => void;
  label: string;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [hsv, setHsv] = useState<Hsv>(() => hexToHsv(value));
  const [hexDraft, setHexDraft] = useState(value);
  const areaRef = useRef<HTMLDivElement>(null);

  // Re-seed the picker from the outside value each time it opens (the value
  // may have changed via reset, palette switch, or remote sync).
  useEffect(() => {
    if (open) {
      setHsv(hexToHsv(value));
      setHexDraft(value);
    }
  }, [open, value]);

  const emit = (next: Hsv) => {
    setHsv(next);
    const hex = hsvToHex(next);
    setHexDraft(hex);
    onChange(hex);
  };

  const moveFromPointer = (e: React.PointerEvent) => {
    const area = areaRef.current;
    if (!area) return;
    const rect = area.getBoundingClientRect();
    const s = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const v =
      1 - Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
    emit({ ...hsv, s, v });
  };

  const commitHex = () => {
    const match = HEX_PATTERN.exec(hexDraft.trim());
    if (!match) {
      setHexDraft(hsvToHex(hsv));
      return;
    }
    const hex = `#${match[1].toLowerCase()}`;
    setHsv(hexToHsv(hex));
    setHexDraft(hex);
    onChange(hex);
  };

  const eyeDropperCtor = (
    globalThis as { EyeDropper?: EyeDropperConstructor }
  ).EyeDropper;

  const pickFromScreen = async () => {
    if (!eyeDropperCtor) return;
    try {
      const result = await new eyeDropperCtor().open();
      const hex = result.sRGBHex.toLowerCase();
      if (HEX_PATTERN.test(hex)) {
        setHsv(hexToHsv(hex));
        setHexDraft(hex);
        onChange(hex);
      }
    } catch {
      // user dismissed the eyedropper
    }
  };

  const hueColor = hsvToHex({ h: hsv.h, s: 1, v: 1 });
  const light = isLightColor(value);

  return (
    <Popover open={open} onOpenChange={setOpen} modal={true}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className={cn(
            "flex h-8 w-24 cursor-pointer items-center gap-1.5 rounded-full border px-2.5 font-mono text-xs uppercase transition-colors",
            light
              ? "border-black/10 text-black/80"
              : "border-white/15 text-white",
          )}
          style={{ backgroundColor: value }}
        >
          <span
            className={cn(
              "size-3.5 shrink-0 rounded-full border",
              light ? "border-black/20" : "border-white/40",
            )}
          />
          {value.toUpperCase()}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-60 rounded-lg p-3">
        <div className="flex flex-col gap-3">
          <div
            ref={areaRef}
            role="slider"
            aria-label={label}
            aria-valuetext={hexDraft}
            aria-valuenow={Math.round(hsv.v * 100)}
            tabIndex={0}
            className="relative h-36 w-full cursor-crosshair touch-none rounded-lg"
            style={{
              background: `linear-gradient(to top, #000, transparent), linear-gradient(to right, #fff, ${hueColor})`,
            }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              moveFromPointer(e);
            }}
            onPointerMove={(e) => {
              if (e.buttons === 1) moveFromPointer(e);
            }}
            onKeyDown={(e) => {
              // Arrow keys move the saturation (x) / value (y) selection so the
              // area is operable without a pointer; Shift takes coarser steps.
              const step = e.shiftKey ? 0.1 : 0.01;
              let next: Hsv | null = null;
              if (e.key === "ArrowLeft")
                next = { ...hsv, s: Math.max(0, hsv.s - step) };
              else if (e.key === "ArrowRight")
                next = { ...hsv, s: Math.min(1, hsv.s + step) };
              else if (e.key === "ArrowDown")
                next = { ...hsv, v: Math.max(0, hsv.v - step) };
              else if (e.key === "ArrowUp")
                next = { ...hsv, v: Math.min(1, hsv.v + step) };
              if (next) {
                e.preventDefault();
                emit(next);
              }
            }}
          >
            <span
              className="pointer-events-none absolute size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
              style={{
                left: `${hsv.s * 100}%`,
                top: `${(1 - hsv.v) * 100}%`,
                backgroundColor: hsvToHex(hsv),
              }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={360}
            step={1}
            value={Math.round(hsv.h)}
            aria-label={t("settings.appearance.custom.colorPicker.hue")}
            onChange={(e) => emit({ ...hsv, h: Number(e.target.value) })}
            className="h-3 w-full cursor-pointer appearance-none rounded-full outline-none [&::-moz-range-thumb]:size-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-white [&::-moz-range-thumb]:bg-transparent [&::-webkit-slider-thumb]:size-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
            style={{
              background:
                "linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00)",
            }}
          />
          <div className="flex items-center gap-2">
            <input
              value={hexDraft.toUpperCase()}
              onChange={(e) => setHexDraft(e.target.value)}
              onBlur={commitHex}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitHex();
                  e.currentTarget.blur();
                }
              }}
              aria-label={t("settings.appearance.custom.colorPicker.hex")}
              spellCheck={false}
              className="h-8 w-full min-w-0 rounded-full border border-border bg-background px-3 font-mono text-xs text-foreground uppercase outline-none focus-visible:border-ring dark:focus-visible:border-transparent dark:focus-visible:bg-white/[0.12] dark:border-transparent dark:bg-white/[0.06]"
            />
            {eyeDropperCtor && (
              <button
                type="button"
                onClick={() => void pickFromScreen()}
                aria-label={t(
                  "settings.appearance.custom.colorPicker.eyedropper",
                )}
                title={t("settings.appearance.custom.colorPicker.eyedropper")}
                className="flex size-8 shrink-0 items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:text-foreground"
              >
                <HugeiconsIcon icon={ColorPickerIcon} className="size-4" />
              </button>
            )}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
