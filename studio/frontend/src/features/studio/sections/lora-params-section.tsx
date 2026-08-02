// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { usePlatformStore } from "@/config/env";
import { CPT_TARGET_MODULES, TARGET_MODULES } from "@/config/training";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { isAdapterMethod } from "@/types/training";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsSliderRow } from "./params-section-controls";
import { selectableOptionStateClassName } from "./params-section-styles";

export function LoraParamsSection(): ReactElement | null {
  const t = useT();
  const isMac = usePlatformStore((state) => state.deviceType === "mac");
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      trainingMethod: state.trainingMethod,
      isVisionModel: state.isVisionModel,
      isDatasetImage: state.isDatasetImage,
      loraRank: state.loraRank,
      loraAlpha: state.loraAlpha,
      loraDropout: state.loraDropout,
      loraVariant: state.loraVariant,
      finetuneVisionLayers: state.finetuneVisionLayers,
      finetuneLanguageLayers: state.finetuneLanguageLayers,
      finetuneAttentionModules: state.finetuneAttentionModules,
      finetuneMLPModules: state.finetuneMLPModules,
      targetModules: state.targetModules,
      setLoraRank: state.setLoraRank,
      setLoraAlpha: state.setLoraAlpha,
      setLoraDropout: state.setLoraDropout,
      setLoraVariant: state.setLoraVariant,
      setFinetuneVisionLayers: state.setFinetuneVisionLayers,
      setFinetuneLanguageLayers: state.setFinetuneLanguageLayers,
      setFinetuneAttentionModules: state.setFinetuneAttentionModules,
      setFinetuneMLPModules: state.setFinetuneMLPModules,
      setTargetModules: state.setTargetModules,
    })),
  );
  // Only mounted in advanced mode, so start expanded when the user switches to it.
  const [open, setOpen] = useState(true);
  const isCpt = store.trainingMethod === "cpt";
  const showVisionLora = store.isVisionModel && store.isDatasetImage === true;

  if (!isAdapterMethod(store.trainingMethod)) {
    return null;
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.params.loraSettings")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <div className="pt-1.5 flex flex-col gap-4">
          <ParamsSliderRow
            label={t("studio.params.rank")}
            tooltip={
              <>
                {t("studio.params.rankTooltip")}{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  {t("studio.params.readMore")}
                </a>
              </>
            }
            value={store.loraRank}
            onChange={store.setLoraRank}
            min={4}
            max={128}
            step={4}
          />
          <ParamsSliderRow
            label={t("studio.params.alpha")}
            tooltip={
              <>
                {t("studio.params.alphaTooltip")}{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  {t("studio.params.readMore")}
                </a>
              </>
            }
            value={store.loraAlpha}
            onChange={store.setLoraAlpha}
            min={4}
            max={256}
            step={4}
          />
          <ParamsSliderRow
            label={t("studio.params.dropout")}
            tooltip={
              <>
                {t("studio.params.dropoutTooltip")}{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  {t("studio.params.readMore")}
                </a>
              </>
            }
            value={store.loraDropout}
            onChange={store.setLoraDropout}
            min={0}
            max={0.5}
            step={0.01}
            format={(value) => value.toFixed(2)}
          />

          {showVisionLora && (
            <div className="flex flex-col gap-2 pt-1">
              {(
                [
                  [
                    "finetuneVisionLayers",
                    t("studio.params.visionLayers"),
                    store.finetuneVisionLayers,
                    store.setFinetuneVisionLayers,
                  ],
                  [
                    "finetuneLanguageLayers",
                    t("studio.params.languageLayers"),
                    store.finetuneLanguageLayers,
                    store.setFinetuneLanguageLayers,
                  ],
                  [
                    "finetuneAttentionModules",
                    t("studio.params.attentionModules"),
                    store.finetuneAttentionModules,
                    store.setFinetuneAttentionModules,
                  ],
                  [
                    "finetuneMLPModules",
                    t("studio.params.mlpModules"),
                    store.finetuneMLPModules,
                    store.setFinetuneMLPModules,
                  ],
                ] as const
              ).map(([key, label, value, setter]) => (
                <div key={key} className="flex items-center gap-2">
                  <Checkbox
                    id={key}
                    checked={value}
                    onCheckedChange={(nextValue) => setter(!!nextValue)}
                  />
                  <label
                    htmlFor={key}
                    className="text-xs cursor-pointer text-muted-foreground"
                  >
                    {label}
                  </label>
                </div>
              ))}
            </div>
          )}

          {!showVisionLora && (
            <div className="flex flex-col gap-2 pt-1">
              <span className="text-xs font-medium text-muted-foreground">
                {t("studio.params.targetModules")}
              </span>
              <div className="flex flex-wrap gap-1.5">
                {(isCpt ? CPT_TARGET_MODULES : TARGET_MODULES).map((module) => {
                  const active = store.targetModules.includes(module);
                  return (
                    <button
                      key={module}
                      type="button"
                      aria-pressed={active}
                      onClick={() =>
                        store.setTargetModules(
                          active
                            ? store.targetModules.filter(
                                (candidate) => candidate !== module,
                              )
                            : [...store.targetModules, module],
                        )
                      }
                      className={`cursor-pointer rounded-full border px-2.5 py-0.5 text-ui-11 font-mono transition-colors ${selectableOptionStateClassName(active)} ${
                        active ? "text-foreground" : "text-muted-foreground"
                      }`}
                    >
                      {module}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2">
            {(
              [
                {
                  value: "lora",
                  label: t("studio.params.enableLora"),
                  desc: t("studio.params.trainWithLora"),
                },
                {
                  value: "rslora",
                  label: "RS-LoRA",
                  desc: t("studio.params.stableRank"),
                },
                {
                  value: "loftq",
                  label: "LoftQ",
                  desc: t("studio.params.memoryEfficient"),
                },
                {
                  value: "dora",
                  label: "DoRA",
                  desc: t("studio.params.weightDecomposed"),
                },
              ] as const
            ).map((option) => {
              const unsupportedOnMlx =
                isMac && (option.value === "loftq" || option.value === "dora");
              return (
                <button
                  key={option.value}
                  type="button"
                  disabled={unsupportedOnMlx}
                  aria-pressed={store.loraVariant === option.value}
                  onClick={() => store.setLoraVariant(option.value)}
                  className={`flex-1 corner-squircle rounded-[14px] border px-3 py-2 text-left transition-colors cursor-pointer disabled:cursor-not-allowed disabled:opacity-60 ${selectableOptionStateClassName(store.loraVariant === option.value)}`}
                >
                  <p className="text-xs font-medium">{option.label}</p>
                  <p className="text-ui-10 text-muted-foreground">
                    {unsupportedOnMlx
                      ? t("studio.params.notSupportedAppleSilicon")
                      : option.desc}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
