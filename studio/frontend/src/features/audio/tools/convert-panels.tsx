// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ParamSlider } from "@/features/chat";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import {
  CHATTERBOX_CONVERT_STEPS_RANGE,
  type ChatterboxConvertValue,
  type RvcValue,
  SEED_VC_ENGINES,
  SEED_VC_GUIDANCE_RANGE,
  SEED_VC_LENGTH_RANGE,
  SEED_VC_STEPS_RANGE,
  type SeedVcEngine,
  type SeedVcValue,
  type Vevo2StyleValue,
  chatterboxConvertLogic,
  rvcLogic,
  seedVcLogic,
  seedVcRoute,
  vevo2StyleLogic,
} from "./convert-panel-logic";
import type { AudioToolPanel } from "./types";

function PanelSection({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section className="grid gap-3" aria-label={title}>
      <div className="grid gap-0.5">
        <h3 className="text-ui-13 font-medium text-foreground">{title}</h3>
        {hint ? (
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {hint}
          </p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

function MoreSettings({
  open,
  onOpenChange,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => onOpenChange(!open)}
        className="flex w-fit items-center gap-1 rounded-full text-ui-11p5 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        More settings
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={cn(
            "size-3 transition-transform duration-150 motion-reduce:transition-none",
            open && "rotate-180",
          )}
        />
      </button>
      {open ? children : null}
    </div>
  );
}

export const seedVcPanel: AudioToolPanel<SeedVcValue> = {
  ...seedVcLogic,
  Component: ({ value, onChange, disabled, ctx }) => {
    const singing = ctx.convertMode === "singing";
    const v2 = seedVcRoute(value, ctx) === "v2_vc";
    return (
      <PanelSection
        title="Seed-VC"
        hint={singing ? "Singing uses the V1 singing engine." : undefined}
      >
        {singing ? null : (
          <div className="grid gap-1.5">
            <label
              htmlFor="convert-seed-vc-engine"
              className="text-ui-13 font-medium text-foreground"
            >
              Engine
            </label>
            <Select
              value={value.engine}
              onValueChange={(engine) =>
                onChange({ ...value, engine: engine as SeedVcEngine })
              }
              disabled={disabled}
            >
              <SelectTrigger
                id="convert-seed-vc-engine"
                size="sm"
                className="w-full"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SEED_VC_ENGINES.map((engine) => (
                  <SelectItem key={engine.value} value={engine.value}>
                    {engine.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        {v2 ? (
          <>
            <ParamSlider
              label="Sound like the target"
              value={value.similarity}
              min={SEED_VC_GUIDANCE_RANGE.min}
              max={SEED_VC_GUIDANCE_RANGE.max}
              step={0.05}
              disabled={disabled}
              info="Higher follows the target voice more closely."
              onChange={(similarity) => onChange({ ...value, similarity })}
            />
            <ParamSlider
              label="Keep words clear"
              value={value.intelligibility}
              min={SEED_VC_GUIDANCE_RANGE.min}
              max={SEED_VC_GUIDANCE_RANGE.max}
              step={0.05}
              disabled={disabled}
              info="Higher keeps the words easier to understand."
              onChange={(intelligibility) =>
                onChange({ ...value, intelligibility })
              }
            />
          </>
        ) : (
          <ParamSlider
            label="Guidance"
            value={value.guidance}
            min={SEED_VC_GUIDANCE_RANGE.min}
            max={SEED_VC_GUIDANCE_RANGE.max}
            step={0.05}
            disabled={disabled}
            info="How strongly the output is pushed toward the target voice."
            onChange={(guidance) => onChange({ ...value, guidance })}
          />
        )}
        <MoreSettings
          open={value.more === true}
          onOpenChange={(more) => onChange({ ...value, more })}
        >
          <ParamSlider
            label="Length"
            value={value.length}
            min={SEED_VC_LENGTH_RANGE.min}
            max={SEED_VC_LENGTH_RANGE.max}
            step={0.05}
            disabled={disabled}
            displayValue={`${value.length.toFixed(2)}×`}
            info="Below 1 speeds the result up, above 1 slows it down."
            onChange={(length) => onChange({ ...value, length })}
          />
          <ParamSlider
            label="Steps"
            value={value.steps}
            min={SEED_VC_STEPS_RANGE.min}
            max={SEED_VC_STEPS_RANGE.max}
            step={1}
            disabled={disabled}
            info="More steps can sound cleaner but take longer."
            onChange={(steps) => onChange({ ...value, steps })}
          />
          {v2 ? (
            <div className="grid gap-1.5">
              <label
                htmlFor="convert-seed-vc-anonymize"
                className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
              >
                Anonymize
                <Switch
                  id="convert-seed-vc-anonymize"
                  checked={value.anonymize}
                  disabled={disabled}
                  onCheckedChange={(anonymize) =>
                    onChange({ ...value, anonymize })
                  }
                />
              </label>
              <p className="text-ui-11p5 leading-snug text-muted-foreground">
                Makes the voice hard to recognise instead of copying the target.
              </p>
            </div>
          ) : null}
        </MoreSettings>
      </PanelSection>
    );
  },
};

export const rvcPanel: AudioToolPanel<RvcValue> = {
  ...rvcLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection title="RVC">
      <ParamSlider
        label="Index blend"
        value={value.blend}
        min={0}
        max={1}
        step={0.05}
        disabled={disabled}
        info="Higher brings in more of the voice's accent and texture; too high can blur words."
        onChange={(blend) => onChange({ ...value, blend })}
      />
      <ParamSlider
        label="Protect consonants"
        value={value.protect}
        min={0}
        max={0.5}
        step={0.01}
        disabled={disabled}
        info="Keeps breaths and soft consonants from the recording. 0.5 turns it off."
        onChange={(protect) => onChange({ ...value, protect })}
      />
      <ParamSlider
        label="Volume envelope mix"
        value={value.rms}
        min={0}
        max={1}
        step={0.05}
        disabled={disabled}
        info="0 keeps the recording's loudness changes; 1 uses the voice's own."
        onChange={(rms) => onChange({ ...value, rms })}
      />
    </PanelSection>
  ),
};

export const chatterboxConvertPanel: AudioToolPanel<ChatterboxConvertValue> = {
  ...chatterboxConvertLogic,
  Component: ({ value, onChange, disabled }) => (
    <PanelSection title="Chatterbox">
      <ParamSlider
        label="Guidance"
        value={value.guidance}
        min={0}
        max={2}
        step={0.05}
        disabled={disabled}
        info="How strongly the output is pushed toward the target voice."
        onChange={(guidance) => onChange({ ...value, guidance })}
      />
      <MoreSettings
        open={value.more === true}
        onOpenChange={(more) => onChange({ ...value, more })}
      >
        <ParamSlider
          label="Steps"
          value={value.steps}
          min={CHATTERBOX_CONVERT_STEPS_RANGE.min}
          max={CHATTERBOX_CONVERT_STEPS_RANGE.max}
          step={1}
          disabled={disabled}
          info="More steps can sound cleaner but take longer."
          onChange={(steps) => onChange({ ...value, steps })}
        />
      </MoreSettings>
    </PanelSection>
  ),
};

// ---- Vevo2 -----------------------------------------------------------------------------------

const VEVO2_HINTS = {
  singing: "Singing keeps the recording's own style.",
  target:
    "Takes the target's accent and delivery too. Needs what's said in the recording.",
  source: "Changes only the voice; the recording's delivery stays.",
} as const;

export const vevo2StylePanel: AudioToolPanel<Vevo2StyleValue> = {
  ...vevo2StyleLogic,
  Component: ({ value, onChange, disabled, ctx }) => (
    <PanelSection
      title="Vevo2"
      hint={
        VEVO2_HINTS[ctx.convertMode === "singing" ? "singing" : value.style]
      }
    >
      {ctx.convertMode === "singing" ? null : (
        <PillTabs
          ariaLabel="Style"
          value={value.style}
          onValueChange={(style) =>
            onChange({ style: style === "target" ? "target" : "source" })
          }
          disabled={disabled}
          fit={true}
          compact={true}
          className="[&>button]:px-3"
          tabs={[
            { value: "source", label: "Keep source style" },
            { value: "target", label: "Take target style" },
          ]}
        />
      )}
    </PanelSection>
  ),
};

export const CONVERT_TOOL_PANELS = [
  seedVcPanel,
  rvcPanel,
  chatterboxConvertPanel,
  vevo2StylePanel,
] as const;
