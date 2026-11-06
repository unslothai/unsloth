// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  effectiveUpscaler,
  hiresGuidanceSupported,
  hiresMinDenoiseFor,
  hiresTargetSize,
  type HiresCapability,
  type HiresChoice,
} from "./lib/hires-choice";

function Label({ label, hint }: { label: string; hint: ReactNode }) {
  return (
    <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
      {label}
      <InfoHint>{hint}</InfoHint>
    </span>
  );
}

function NumberInput({
  ariaLabel,
  value,
  placeholder,
  min,
  max,
  step,
  integer,
  onChange,
}: {
  ariaLabel: string;
  value: number | null;
  placeholder?: string;
  min: number;
  max: number;
  step: number;
  integer?: boolean;
  onChange: (v: number | null) => void;
}) {
  return (
    <Input
      aria-label={ariaLabel}
      type="number"
      inputMode={integer ? "numeric" : "decimal"}
      className="h-8 w-24 text-xs"
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value.trim();
        if (raw === "") return onChange(null);
        const n = Number(raw);
        if (!Number.isFinite(n)) return;
        const bounded = Math.min(max, Math.max(min, integer ? Math.round(n) : n));
        onChange(bounded);
      }}
    />
  );
}

export function HiresControls({
  capability,
  choice,
  onChange,
  width,
  height,
}: {
  capability: HiresCapability;
  choice: HiresChoice;
  onChange: (next: HiresChoice) => void;
  width: number;
  height: number;
}) {
  const target = hiresTargetSize(width, height, choice.scale, capability.max_side);
  const upscaler = effectiveUpscaler(choice, capability);
  const minDenoise = hiresMinDenoiseFor(choice, capability);
  return (
    <div className="flex flex-col gap-2 pt-2">
      <div className="flex items-center justify-between">
        <Label
          label="Hires fix"
          hint="Render, enlarge, then redraw the enlarged image at a denoise, in one run (AUTOMATIC1111 Hires. fix, ComfyUI 2-pass). Adds detail a single pass at the large size would not."
        />
        <Switch
          checked={choice.enabled}
          onCheckedChange={(enabled) =>
            onChange(
              enabled && !choice.enabled
                ? {
                    ...choice,
                    enabled,
                    scale: choice.scale || capability.defaults.scale,
                    denoise: choice.denoise ?? capability.defaults.denoise,
                  }
                : { ...choice, enabled },
            )
          }
          aria-label="Hires fix"
        />
      </div>
      {choice.enabled && (
        <div className="flex flex-col gap-2 rounded-lg border border-border/60 p-2">
          <div className="flex items-center justify-between gap-2">
            <Label label="Upscale by" hint="Output size is the base size times this factor, rounded to a multiple of 16." />
            <NumberInput
              ariaLabel="Hires upscale by"
              value={choice.scale}
              min={1}
              max={4}
              step={0.05}
              onChange={(v) => onChange({ ...choice, scale: v ?? capability.defaults.scale })}
            />
          </div>
          <div className="flex items-center justify-between gap-2">
            <Label
              label="Denoise"
              hint={
                minDenoise > 0
                  ? "How much the second pass redraws. 0.5 is ComfyUI's two-pass default; the latent upscaler needs about 0.5 or more."
                  : "How much the second pass redraws. 0.5 is ComfyUI's two-pass default; the latent upscaler needs about 0.5 or more. 0 only upscales."
              }
            />
            <NumberInput
              ariaLabel="Hires denoise"
              value={choice.denoise}
              min={minDenoise}
              max={1}
              step={0.05}
              onChange={(v) => onChange({ ...choice, denoise: v ?? capability.defaults.denoise })}
            />
          </div>
          <div className="flex items-center justify-between gap-2">
            <Label label="Hires steps" hint="Exactly how many steps the second pass runs. Empty uses the base Steps." />
            <NumberInput
              ariaLabel="Hires steps"
              value={choice.steps}
              placeholder="Same"
              min={1}
              max={100}
              step={1}
              integer
              onChange={(steps) => onChange({ ...choice, steps })}
            />
          </div>
          <div className="flex items-center justify-between gap-2">
            <Label
              label="Upscaler"
              hint="Latent enlarges the latents (SDXL), Lanczos the pixels, R-ESRGAN runs an upscaling model first (downloaded on first use)."
            />
            <Select
              value={upscaler}
              onValueChange={(id) => {
                const next = { ...choice, upscaler: id };
                onChange({ ...next, denoise: Math.max(next.denoise, hiresMinDenoiseFor(next, capability)) });
              }}
            >
              <SelectTrigger className="h-8 w-44 text-xs" aria-label="Hires upscaler">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {capability.upscalers.map((u) => (
                  <SelectItem key={u.id} value={u.id}>
                    {u.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {hiresGuidanceSupported(capability) && (
            <div className="flex items-center justify-between gap-2">
              <Label label="Hires CFG" hint="Guidance for the second pass. Empty uses the base guidance." />
              <NumberInput
                ariaLabel="Hires CFG"
                value={choice.guidance}
                placeholder="Same"
                min={0}
                max={20}
                step={0.5}
                onChange={(guidance) => onChange({ ...choice, guidance })}
              />
            </div>
          )}
          <p className="text-xs text-muted-foreground" aria-label="Hires output size">
            {width}x{height} {"->"} {target.width}x{target.height}
          </p>
        </div>
      )}
    </div>
  );
}

export function UpscaleModelField({
  models,
  value,
  onChange,
}: {
  models: Array<{ id: string; label: string }> | undefined;
  value: string | null;
  onChange: (id: string | null) => void;
}) {
  if (!models || models.length === 0) return null;
  const current = value && models.some((m) => m.id === value) ? value : "lanczos";
  return (
    <div className="flex items-center justify-between gap-2 pt-2">
      <Label
        label="Upscaler"
        hint="Lanczos resizes the pixels. A model upscaler (R-ESRGAN) enlarges first and keeps edges sharper; it downloads on first use."
      />
      <Select value={current} onValueChange={(id) => onChange(id === "lanczos" ? null : id)}>
        <SelectTrigger className="h-8 w-44 text-xs" aria-label="Upscaler">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="lanczos">Lanczos</SelectItem>
          {models.map((m) => (
            <SelectItem key={m.id} value={m.id}>
              {m.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
