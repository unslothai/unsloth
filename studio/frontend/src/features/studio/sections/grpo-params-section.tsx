// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { getRewardFunctionPresets } from "@/features/training";
import type { RewardFunctionPreset } from "@/features/training";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow, ParamsSliderRow } from "./params-section-controls";
import { selectableOptionStateClassName } from "./params-section-styles";

function parseWeight(input: string): number | null {
  const value = Number(input);
  return input !== "" && Number.isFinite(value) && value > 0 ? value : null;
}

export function GrpoParamsSection(): ReactElement | null {
  const t = useT();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      trainingMethod: state.trainingMethod,
      rewardFunctions: state.rewardFunctions,
      numGenerations: state.numGenerations,
      maxPromptLength: state.maxPromptLength,
      maxCompletionLength: state.maxCompletionLength,
      grpoTemperature: state.grpoTemperature,
      grpoTopP: state.grpoTopP,
      grpoBeta: state.grpoBeta,
      setRewardFunctions: state.setRewardFunctions,
      setNumGenerations: state.setNumGenerations,
      setMaxPromptLength: state.setMaxPromptLength,
      setMaxCompletionLength: state.setMaxCompletionLength,
      setGrpoTemperature: state.setGrpoTemperature,
      setGrpoTopP: state.setGrpoTopP,
      setGrpoBeta: state.setGrpoBeta,
    })),
  );
  const [open, setOpen] = useState(true);
  const [presets, setPresets] = useState<RewardFunctionPreset[]>([]);
  const isGrpo = store.trainingMethod === "grpo";

  useEffect(() => {
    if (!isGrpo || presets.length > 0) {
      return;
    }
    const controller = new AbortController();
    getRewardFunctionPresets(controller.signal)
      .then(setPresets)
      .catch(() => {
        // The curated list is static server-side; a failed fetch leaves the
        // saved selection in place rather than clearing the user's rewards.
      });
    return () => controller.abort();
  }, [isGrpo, presets.length]);

  if (!isGrpo) {
    return null;
  }

  const selectedById = new Map(
    store.rewardFunctions.map((selection) => [selection.id, selection]),
  );

  const toggle = (preset: RewardFunctionPreset, selected: boolean) => {
    store.setRewardFunctions(
      selected
        ? [
            ...store.rewardFunctions,
            { id: preset.id, weight: preset.default_weight },
          ]
        : store.rewardFunctions.filter(
            (selection) => selection.id !== preset.id,
          ),
    );
  };

  const setWeight = (id: string, weight: number) => {
    store.setRewardFunctions(
      store.rewardFunctions.map((selection) =>
        selection.id === id ? { ...selection, weight } : selection,
      ),
    );
  };

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.grpo.settings")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <div className="flex flex-col gap-4 pt-1.5">
          <div className="flex flex-col gap-2">
            <span className="text-xs font-medium text-muted-foreground">
              {t("studio.grpo.rewardFunctions")}
            </span>
            {presets.length === 0 ? (
              <span className="text-xs text-muted-foreground/70">
                {t("studio.grpo.rewardFunctionsLoading")}
              </span>
            ) : null}
            {presets.map((preset) => {
              const selection = selectedById.get(preset.id);
              const selected = selection !== undefined;
              return (
                <div
                  key={preset.id}
                  className={`flex items-start gap-3 rounded-lg border p-3 ${selectableOptionStateClassName(selected)}`}
                >
                  <Checkbox
                    id={`reward-${preset.id}`}
                    className="mt-0.5"
                    checked={selected}
                    onCheckedChange={(next) => toggle(preset, !!next)}
                  />
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <label
                      htmlFor={`reward-${preset.id}`}
                      className="cursor-pointer text-xs font-medium text-foreground"
                    >
                      {preset.name}
                    </label>
                    <span className="text-xs leading-snug text-muted-foreground">
                      {preset.description}
                    </span>
                    {preset.expected_columns.length > 0 ? (
                      <span className="text-ui-10 text-muted-foreground/70">
                        {t("studio.grpo.expectedColumns", {
                          columns: preset.expected_columns.join(", "),
                        })}
                      </span>
                    ) : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <span className="text-ui-10 text-muted-foreground">
                      {t("studio.grpo.weight")}
                    </span>
                    <Input
                      className="h-7 w-16 text-xs"
                      inputMode="decimal"
                      disabled={!selected}
                      value={String(selection?.weight ?? preset.default_weight)}
                      onChange={(event) => {
                        const weight = parseWeight(event.target.value);
                        if (weight !== null) {
                          setWeight(preset.id, weight);
                        }
                      }}
                      aria-label={`${preset.name} ${t("studio.grpo.weight")}`}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <ParamsSliderRow
            label={t("studio.grpo.numGenerations")}
            tooltip={t("studio.grpo.numGenerationsTooltip")}
            value={store.numGenerations}
            onChange={store.setNumGenerations}
            min={2}
            max={16}
            step={1}
          />
          <ParamsSliderRow
            label={t("studio.grpo.maxPromptLength")}
            tooltip={t("studio.grpo.maxPromptLengthTooltip")}
            value={store.maxPromptLength}
            onChange={store.setMaxPromptLength}
            min={64}
            max={2048}
            step={64}
          />
          <ParamsSliderRow
            label={t("studio.grpo.maxCompletionLength")}
            tooltip={t("studio.grpo.maxCompletionLengthTooltip")}
            value={store.maxCompletionLength}
            onChange={store.setMaxCompletionLength}
            min={64}
            max={4096}
            step={64}
          />
          <ParamsSliderRow
            label={t("studio.grpo.temperature")}
            tooltip={t("studio.grpo.temperatureTooltip")}
            value={store.grpoTemperature}
            onChange={store.setGrpoTemperature}
            min={0.1}
            max={2}
            step={0.05}
            format={(value) => value.toFixed(2)}
          />
          <ParamsSliderRow
            label={t("studio.grpo.topP")}
            tooltip={t("studio.grpo.topPTooltip")}
            value={store.grpoTopP}
            onChange={store.setGrpoTopP}
            min={0.05}
            max={1}
            step={0.05}
            format={(value) => value.toFixed(2)}
          />
          <ParamsSliderRow
            label={t("studio.grpo.beta")}
            tooltip={t("studio.grpo.betaTooltip")}
            value={store.grpoBeta}
            onChange={store.setGrpoBeta}
            min={0}
            max={0.5}
            step={0.01}
            format={(value) => value.toFixed(2)}
          />
          <ParamsRow label={t("studio.grpo.generationBackend")}>
            <span className="text-xs text-muted-foreground">
              {t("studio.grpo.generationBackendValue")}
            </span>
          </ParamsRow>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
