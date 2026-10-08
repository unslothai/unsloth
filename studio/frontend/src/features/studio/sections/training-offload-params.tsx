// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTrainingConfigStore } from "@/features/training";
import type { OffloadLayers, PrefetchDepth } from "@/features/training/types/config";
import { useT } from "@/i18n";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { offloadCountFromInput } from "./offload-panel-layout";
import { ParamsRow } from "./params-section-controls";

type OffloadMode = "off" | "auto" | "layers";

function modeOf(value: OffloadLayers): OffloadMode {
  if (value === "auto") return "auto";
  return value > 0 ? "layers" : "off";
}

const PREFETCH_CHOICES: PrefetchDepth[] = ["auto", 1, 2, 3, 4];

export function OffloadLayersParams({ budget = true }: { budget?: boolean }): ReactElement {
  const t = useT();
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      offloadLayers: state.offloadLayers,
      offloadVramGb: state.offloadVramGb,
      prefetchDepth: state.prefetchDepth,
      setOffloadLayers: state.setOffloadLayers,
      setOffloadVramGb: state.setOffloadVramGb,
      setPrefetchDepth: state.setPrefetchDepth,
    })),
  );
  // The count box's text while it is being edited: clearing it to retype must not read as Off.
  const [countDraft, setCountDraft] = useState<string | null>(null);
  const mode = countDraft !== null ? "layers" : modeOf(store.offloadLayers);

  return (
    <>
      <ParamsRow
        label={t("studio.params.offloadLayers")}
        tooltip={t("studio.params.offloadLayersTooltip")}
      >
        <div className="flex items-center gap-2">
          <Select
            value={mode}
            onValueChange={(value) => {
              const next = value as OffloadMode;
              setCountDraft(null);
              if (next === "off") store.setOffloadLayers(0);
              else if (next === "auto") store.setOffloadLayers("auto");
              else store.setOffloadLayers(typeof store.offloadLayers === "number" && store.offloadLayers > 0 ? store.offloadLayers : 8);
            }}
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="off">{t("studio.params.offloadOff")}</SelectItem>
              <SelectItem value="auto">{t("studio.params.offloadAuto")}</SelectItem>
              <SelectItem value="layers">{t("studio.params.offloadCount")}</SelectItem>
            </SelectContent>
          </Select>
          {mode === "layers" && (
            <Input
              type="number"
              inputMode="numeric"
              min={1}
              step={1}
              aria-label={t("studio.params.offloadCount")}
              value={countDraft ?? (store.offloadLayers === "auto" ? "" : store.offloadLayers || "")}
              onChange={(e) => {
                setCountDraft(e.target.value);
                const count = offloadCountFromInput(e.target.value);
                if (count !== null) store.setOffloadLayers(count);
              }}
              // An empty or zero box keeps the last count; Off is chosen from the select.
              onBlur={() => setCountDraft(null)}
              className="w-20 font-mono"
            />
          )}
        </div>
      </ParamsRow>
      {mode === "auto" && budget && (
        <ParamsRow
          label={t("studio.params.offloadVramBudget")}
          tooltip={t("studio.params.offloadVramBudgetTooltip")}
        >
          <Input
            type="number"
            inputMode="decimal"
            min={1}
            step={0.5}
            placeholder={t("studio.params.offloadWholeCard")}
            title={t("studio.params.offloadWholeCard")}
            value={store.offloadVramGb ?? ""}
            onChange={(e) => {
              const gb = Number(e.target.value);
              store.setOffloadVramGb(e.target.value === "" || !(gb > 0) ? null : gb);
            }}
            className="w-36 font-mono placeholder:font-sans"
          />
        </ParamsRow>
      )}
      {mode !== "off" && (
        <ParamsRow
          label={t("studio.params.prefetchDepth")}
          tooltip={t("studio.params.prefetchDepthTooltip")}
        >
          <Select
            value={String(store.prefetchDepth)}
            onValueChange={(value) =>
              store.setPrefetchDepth(value === "auto" ? "auto" : Number(value))
            }
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PREFETCH_CHOICES.map((choice) => (
                <SelectItem key={String(choice)} value={String(choice)}>
                  {choice === "auto" ? t("studio.params.offloadAuto") : choice}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ParamsRow>
      )}
    </>
  );
}
