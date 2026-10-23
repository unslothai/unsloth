// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { usePlatformStore } from "@/config/env";
import {
  drafterEligible,
  useDrafterCapability,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ParamsRow } from "./params-section-controls";

export function DrafterSection(): ReactElement | null {
  const t = useT();
  const deviceType = usePlatformStore((state) => state.deviceType);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      eligible: drafterEligible(state),
      capability: state.drafterCapability,
      enabled: state.drafterEnabled,
      dataSource: state.drafterDataSource,
      maxExamples: state.drafterMaxExamples,
      epochs: state.drafterEpochs,
      learningRate: state.drafterLearningRate,
      draftSteps: state.drafterDraftSteps,
      setEnabled: state.setDrafterEnabled,
      setDataSource: state.setDrafterDataSource,
      setMaxExamples: state.setDrafterMaxExamples,
      setEpochs: state.setDrafterEpochs,
      setLearningRate: state.setDrafterLearningRate,
      setDraftSteps: state.setDrafterDraftSteps,
    })),
  );
  const [open, setOpen] = useState(false);
  useDrafterCapability(open || store.enabled);

  // MLX trains through its own worker, which has no drafter phase.
  if (!store.eligible || deviceType === "mac") {
    return null;
  }

  const capability =
    store.capability && store.capability.model_name === store.selectedModel
      ? store.capability
      : null;
  const supported = capability?.supported === true;
  const helper = !capability
    ? t("studio.params.drafterChecking")
    : !supported
      ? t("studio.params.drafterUnavailable", {
          reason: capability.reason ?? "",
        })
      : capability.has_native_mtp
        ? t("studio.params.drafterRetrainMtp")
        : t("studio.params.drafterEagle3");

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        {t("studio.params.drafterSettings")}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
        <div className="pt-1.5 flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <ParamsRow
              label={t("studio.params.drafterTrain")}
              tooltip={t("studio.params.drafterTrainTooltip")}
            >
              <Checkbox
                id="drafterEnabled"
                aria-label={t("studio.params.drafterTrain")}
                checked={store.enabled && supported}
                disabled={!supported}
                onCheckedChange={(value) => store.setEnabled(!!value)}
              />
            </ParamsRow>
            <span
              className="text-xs text-muted-foreground"
              data-testid="drafter-helper"
            >
              {helper}
            </span>
          </div>
          {store.enabled && supported && (
            <>
              <ParamsRow label={t("studio.params.drafterDataSource")}>
                <Select
                  value={store.dataSource}
                  onValueChange={(value) =>
                    store.setDataSource(
                      value === "regenerate" ? "regenerate" : "dataset",
                    )
                  }
                >
                  <SelectTrigger className="w-56">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="dataset">
                      {t("studio.params.drafterDataSourceDataset")}
                    </SelectItem>
                    <SelectItem value="regenerate">
                      {t("studio.params.drafterDataSourceRegenerate")}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </ParamsRow>
              <ParamsRow label={t("studio.params.drafterMaxExamples")}>
                <Input
                  type="number"
                  step="100"
                  min="16"
                  max="100000"
                  value={store.maxExamples}
                  onChange={(event) =>
                    store.setMaxExamples(Number(event.target.value))
                  }
                  className="w-28 font-mono"
                />
              </ParamsRow>
              <ParamsRow label={t("studio.params.drafterEpochs")}>
                <Input
                  type="number"
                  step="0.5"
                  min="0.1"
                  max="20"
                  value={store.epochs}
                  onChange={(event) =>
                    store.setEpochs(Number(event.target.value))
                  }
                  className="w-28 font-mono"
                />
              </ParamsRow>
              <ParamsRow label={t("studio.params.drafterLearningRate")}>
                <Input
                  type="number"
                  step="0.00001"
                  min="0"
                  placeholder={t("studio.params.drafterLearningRateDefault")}
                  value={store.learningRate ?? ""}
                  onChange={(event) => {
                    const raw = event.target.value.trim();
                    const value = Number(raw);
                    store.setLearningRate(
                      raw === "" || !Number.isFinite(value) || value <= 0
                        ? null
                        : value,
                    );
                  }}
                  className="w-28 font-mono"
                />
              </ParamsRow>
              <ParamsRow label={t("studio.params.drafterDraftSteps")}>
                <Input
                  type="number"
                  step="1"
                  min="1"
                  max="8"
                  value={store.draftSteps}
                  onChange={(event) =>
                    store.setDraftSteps(Number(event.target.value))
                  }
                  className="w-28 font-mono"
                />
              </ParamsRow>
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
