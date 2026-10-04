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
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import {
  GRPO_VARIANTS,
  type GrpoVariant,
  type TrainingObjective,
} from "@/types/training";
import { type ReactElement, useState } from "react";
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
  // Commit on blur: clamping per keystroke turned "0.28" into "0.0128".
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) {
      return;
    }
    const n = parseOptional(draft, integer);
    onChange(
      n === null
        ? null
        : Math.min(
            max ?? Number.POSITIVE_INFINITY,
            Math.max(min ?? Number.NEGATIVE_INFINITY, n),
          ),
    );
    setDraft(null);
  };
  return (
    <Input
      type="number"
      inputMode={integer ? "numeric" : "decimal"}
      className="h-8 w-[calc(110px*var(--ui-space-scale,1))] text-right text-xs"
      value={draft ?? value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
        }
      }}
    />
  );
}

export function RlParamsSection({
  objective,
  inTab = false,
}: {
  objective: Exclude<TrainingObjective, "sft">;
  inTab?: boolean;
}): ReactElement {
  const t = useT();
  const s = useTrainingConfigStore(
    useShallow((state) => ({
      rlBeta: state.rlBeta,
      rlMaxPromptLength: state.rlMaxPromptLength,
      grpoNumGenerations: state.grpoNumGenerations,
      grpoMaxCompletionLength: state.grpoMaxCompletionLength,
      grpoTemperature: state.grpoTemperature,
      grpoVariant: state.grpoVariant,
      grpoMaskTruncatedCompletions: state.grpoMaskTruncatedCompletions,
      grpoEpsilonHigh: state.grpoEpsilonHigh,
      setRlBeta: state.setRlBeta,
      setRlMaxPromptLength: state.setRlMaxPromptLength,
      setGrpoNumGenerations: state.setGrpoNumGenerations,
      setGrpoMaxCompletionLength: state.setGrpoMaxCompletionLength,
      setGrpoTemperature: state.setGrpoTemperature,
      setGrpoVariant: state.setGrpoVariant,
      setGrpoMaskTruncatedCompletions: state.setGrpoMaskTruncatedCompletions,
      setGrpoEpsilonHigh: state.setGrpoEpsilonHigh,
    })),
  );
  const auto = t("rl.params.auto");

  return (
    <div
      className={
        inTab
          ? "flex flex-col gap-1"
          : "flex flex-col gap-1 border-t border-border/70 pt-4"
      }
    >
      {!inTab && (
        <p className="mb-1 flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
          {t("rl.params.title", { objective: objective.toUpperCase() })}
          <NewBadge />
        </p>
      )}
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
              onChange={(v) => s.setGrpoNumGenerations(v ?? 4)}
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
              onChange={(v) => s.setGrpoTemperature(v ?? 1)}
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
              onChange={s.setGrpoEpsilonHigh}
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
