// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { motion, useReducedMotionConfig } from "motion/react";
import type { GpuRenderingSetting } from "../api/gpu-rendering";

const OPTIONS: { value: GpuRenderingSetting; labelKey: TranslationKey }[] = [
  { value: "auto", labelKey: "settings.appearance.gpuRendering.auto" },
  { value: "on", labelKey: "settings.appearance.gpuRendering.on" },
  { value: "off", labelKey: "settings.appearance.gpuRendering.off" },
];

export function GpuRenderingSegmented({
  value,
  onChange,
  disabled = false,
}: {
  value: GpuRenderingSetting | null;
  onChange: (value: GpuRenderingSetting) => void;
  disabled?: boolean;
}) {
  const t = useT();
  const reduced = useReducedMotionConfig();
  return (
    <div className="hub-tab-toggle inline-flex h-8 items-center rounded-full">
      {OPTIONS.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            aria-pressed={active}
            disabled={disabled}
            className={cn(
              "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors",
              disabled && "cursor-not-allowed",
              active
                ? "text-foreground"
                : disabled
                  ? "text-muted-foreground/45"
                  : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="gpu-rendering-pill"
                className="hub-tab-toggle-pill absolute inset-0 rounded-full"
                transition={
                  reduced
                    ? { duration: 0 }
                    : { type: "spring", stiffness: 500, damping: 35, mass: 0.5 }
                }
              />
            )}
            <span className="relative z-10">{t(opt.labelKey)}</span>
          </button>
        );
      })}
    </div>
  );
}
