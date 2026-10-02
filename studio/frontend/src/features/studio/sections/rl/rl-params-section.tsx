// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { GRPO_DEFAULT_SYSTEM_PROMPT } from "@/config/training";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import {
  GRPO_VARIANTS,
  type GrpoVariant,
  type TrainingObjective,
} from "@/types/training";
import type { ReactElement } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "../params-section-controls";

// Matches DEFAULT_BETA in studio/backend/core/training/rl.py.
const DEFAULT_BETA: Record<Exclude<TrainingObjective, "sft">, number> = {
  dpo: 0.1,
  orpo: 0.1,
  grpo: 0,
};

function parseOptional(value: string, integer: boolean): number | null {
  if (value.trim() === "") {
    return null;
  }
  const n = integer ? Number.parseInt(value, 10) : Number(value);
  return Number.isFinite(n) ? n : null;
}

function NumberField({
  value,
  onChange,
  min,
  max,
  step,
  integer = false,
  placeholder,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  placeholder?: string;
}): ReactElement {
  return (
    <Input
      type="number"
      inputMode={integer ? "numeric" : "decimal"}
      className="h-8 w-[calc(110px*var(--ui-space-scale,1))] text-right text-xs"
      value={value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => onChange(parseOptional(e.target.value, integer))}
    />
  );
}

export function RlParamsSection({
  objective,
}: {
  objective: Exclude<TrainingObjective, "sft">;
}): ReactElement {
  const t = useT();
  const s = useTrainingConfigStore(
    useShallow((state) => ({
      rlBeta: state.rlBeta,
      rlMaxPromptLength: state.rlMaxPromptLength,
      grpoNumGenerations: state.grpoNumGenerations,
      grpoMaxCompletionLength: state.grpoMaxCompletionLength,
      grpoTemperature: state.grpoTemperature,
      grpoSystemPrompt: state.grpoSystemPrompt,
      grpoVariant: state.grpoVariant,
      grpoMaskTruncatedCompletions: state.grpoMaskTruncatedCompletions,
      grpoEpsilonHigh: state.grpoEpsilonHigh,
      setRlBeta: state.setRlBeta,
      setRlMaxPromptLength: state.setRlMaxPromptLength,
      setGrpoNumGenerations: state.setGrpoNumGenerations,
      setGrpoMaxCompletionLength: state.setGrpoMaxCompletionLength,
      setGrpoTemperature: state.setGrpoTemperature,
      setGrpoSystemPrompt: state.setGrpoSystemPrompt,
      setGrpoVariant: state.setGrpoVariant,
      setGrpoMaskTruncatedCompletions: state.setGrpoMaskTruncatedCompletions,
      setGrpoEpsilonHigh: state.setGrpoEpsilonHigh,
    })),
  );
  const auto = t("rl.params.auto");

  return (
    <div className="flex flex-col gap-1 border-t border-border/70 pt-4">
      <p className="mb-1 flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {t("rl.params.title", { objective: objective.toUpperCase() })}
        <NewBadge />
      </p>
      {objective === "grpo" && (
        <>
          <ParamsRow
            label={t("rl.params.variant")}
            tooltip={t("rl.params.variantHint")}
          >
            <Select
              value={s.grpoVariant}
              onValueChange={(v) => s.setGrpoVariant(v as GrpoVariant)}
            >
              <SelectTrigger
                size="sm"
                className="w-[calc(170px*var(--ui-space-scale,1))]"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {GRPO_VARIANTS.map((v) => (
                  <SelectItem key={v} value={v}>
                    {t(`rl.params.variants.${v}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ParamsRow>
          <ParamsRow
            label={t("rl.params.generations")}
            tooltip={t("rl.params.generationsHint")}
          >
            <NumberField
              integer={true}
              min={2}
              max={16}
              value={s.grpoNumGenerations}
              onChange={(v) =>
                s.setGrpoNumGenerations(Math.min(16, Math.max(2, v ?? 4)))
              }
            />
          </ParamsRow>
          <ParamsRow
            label={t("rl.params.maxCompletion")}
            tooltip={t("rl.params.maxCompletionHint")}
          >
            <NumberField
              integer={true}
              min={16}
              value={s.grpoMaxCompletionLength}
              placeholder={auto}
              onChange={s.setGrpoMaxCompletionLength}
            />
          </ParamsRow>
          <ParamsRow
            label={t("rl.params.temperature")}
            tooltip={t("rl.params.temperatureHint")}
          >
            <NumberField
              min={0.1}
              max={2}
              step={0.1}
              value={s.grpoTemperature}
              onChange={(v) =>
                s.setGrpoTemperature(Math.min(2, Math.max(0.1, v ?? 1)))
              }
            />
          </ParamsRow>
          <ParamsRow
            label={t("rl.params.epsilonHigh")}
            tooltip={t("rl.params.epsilonHighHint")}
          >
            <NumberField
              min={0.01}
              max={1}
              step={0.01}
              value={s.grpoEpsilonHigh}
              placeholder={auto}
              onChange={(v) =>
                s.setGrpoEpsilonHigh(
                  v === null ? null : Math.min(1, Math.max(0.01, v)),
                )
              }
            />
          </ParamsRow>
          <ParamsRow
            label={t("rl.params.maskTruncated")}
            tooltip={t("rl.params.maskTruncatedHint")}
          >
            <Switch
              checked={s.grpoMaskTruncatedCompletions}
              onCheckedChange={s.setGrpoMaskTruncatedCompletions}
            />
          </ParamsRow>
        </>
      )}
      {objective === "grpo" && (
        <div className="flex flex-col gap-1.5 pt-2">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-medium text-foreground">
              {t("rl.params.systemPrompt")}
            </p>
            {s.grpoSystemPrompt !== GRPO_DEFAULT_SYSTEM_PROMPT && (
              <button
                type="button"
                className="text-ui-11p5 text-primary underline-offset-2 hover:underline"
                onClick={() =>
                  s.setGrpoSystemPrompt(GRPO_DEFAULT_SYSTEM_PROMPT)
                }
              >
                {t("rl.params.systemPromptReset")}
              </button>
            )}
          </div>
          <Textarea
            rows={4}
            aria-label={t("rl.params.systemPrompt")}
            className="font-mono text-xs"
            value={s.grpoSystemPrompt}
            onChange={(e) => s.setGrpoSystemPrompt(e.target.value)}
          />
          <p className="text-ui-11p5 text-muted-foreground/85">
            {t("rl.params.systemPromptHint")}
          </p>
        </div>
      )}
      <ParamsRow
        label={t("rl.params.maxPrompt")}
        tooltip={t("rl.params.maxPromptHint")}
      >
        <NumberField
          integer={true}
          min={16}
          value={s.rlMaxPromptLength}
          placeholder={auto}
          onChange={s.setRlMaxPromptLength}
        />
      </ParamsRow>
      <ParamsRow
        label={t("rl.params.beta")}
        tooltip={t("rl.params.betaHint", { value: DEFAULT_BETA[objective] })}
      >
        <NumberField
          min={0}
          max={10}
          step={0.01}
          value={s.rlBeta}
          placeholder={String(DEFAULT_BETA[objective])}
          onChange={s.setRlBeta}
        />
      </ParamsRow>
    </div>
  );
}
