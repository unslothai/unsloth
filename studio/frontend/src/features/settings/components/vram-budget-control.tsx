// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";
import { usePlatformStore } from "@/config/env";
// eslint-disable-next-line no-restricted-imports -- leaf import; the UI barrels would cycle.
import { AdvancedGpuSlider } from "@/features/model-picker/components/advanced-gpu-slider";
// eslint-disable-next-line no-restricted-imports -- Pure conversions; avoid the model-picker UI barrel.
import {
  VRAM_BUDGET_PERCENT_STEP,
  vramFractionToPercent,
  vramPercentToFraction,
} from "@/features/model-picker/model-config/per-model-config";
import {
  type VramBudgetSettings,
  flushVramBudgetSave,
  hasStagedVramBudgetSave,
  isVramBudgetLocked,
  loadVramBudgetSettings,
  stageVramBudgetSave,
  subscribeVramBudgetLock,
  subscribeVramBudgetSettings,
  updateVramBudgetSettings,
} from "../api/vram-budget";

export function VramBudgetControl({
  modelLoading = false,
  image = false,
}: { modelLoading?: boolean; image?: boolean }) {
  // macOS sizes Metal from _APPLE_UNIFIED_MEMORY_FRACTION and ignores this budget.
  const isMac = usePlatformStore((s) => s.deviceType === "mac");
  const [settings, setSettings] = useState<VramBudgetSettings | null>(null);
  const [percent, setPercent] = useState<number | null>(null);
  const [draftPercent, setDraftPercent] = useState<number | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const adviceId = useId();
  const wasModelLoading = useRef(false);
  // Locked while Run settles the budget, or an edit would be flushed with the load.
  const [locked, setLocked] = useState(isVramBudgetLocked);
  useEffect(() => subscribeVramBudgetLock(setLocked), []);

  // A staged edit outranks the publish; a dropped one does not, or the slider shows a value no load uses.
  useEffect(() => {
    if (isMac) {
      return;
    }
    return subscribeVramBudgetSettings((next) => {
      if (hasStagedVramBudgetSave()) {
        return;
      }
      setSettings(next);
      setPercent(vramFractionToPercent(next.fraction));
    });
  }, [isMac]);

  useEffect(() => {
    let cancelled = false;
    if (isMac) {
      return;
    }
    loadVramBudgetSettings().then((loaded) => {
      if (cancelled || !loaded) {
        return;
      }
      setSettings(loaded);
      setPercent(vramFractionToPercent(loaded.fraction));
    });
    return () => {
      cancelled = true;
    };
    // deviceType settles once /api/health answers, so re-run if the seed guess flips.
  }, [isMac]);

  // reloadRequired goes stale when a load finishes; refresh on the falling edge only.
  useEffect(() => {
    const finished = wasModelLoading.current && !modelLoading;
    wasModelLoading.current = modelLoading;
    if (isMac || !finished) {
      return;
    }
    let cancelled = false;
    // Forced: an earlier read describes the child being replaced.
    loadVramBudgetSettings({ force: true }).then((loaded) => {
      if (cancelled || !loaded) {
        return;
      }
      setSettings(loaded);
    });
    return () => {
      cancelled = true;
    };
  }, [isMac, modelLoading]);

  // Flush, do not drop: the fraction is held nowhere else.
  useEffect(
    () => () => {
      if (saveTimer.current) {
        clearTimeout(saveTimer.current);
        saveTimer.current = null;
      }
      flushVramBudgetSave()?.catch((error: unknown) => {
        toast.error(
          error instanceof Error ? error.message : "Failed to save VRAM budget",
        );
      });
    },
    [],
  );

  if (isMac || !settings || percent === null) {
    return null;
  }

  const defaultPercent = vramFractionToPercent(settings.defaultFraction);
  const shownPercent = draftPercent ?? percent;
  // Stored beats UNSLOTH_VRAM_FRACTION, so once the slider has been touched there is otherwise
  // no way back to inheriting it. Clearing is what the null the API accepts is for.
  const resetBudget = () => {
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    // Drop a queued drag, or its debounce would store back what this clears.
    stageVramBudgetSave(null);
    updateVramBudgetSettings(null)
      .then((next) => {
        setSettings(next);
        setPercent(vramFractionToPercent(next.fraction));
      })
      .catch((error: unknown) => {
        toast.error(
          error instanceof Error
            ? error.message
            : "Failed to reset VRAM budget",
        );
      });
  };
  const commit = (next: number) => {
    setPercent(next);
    // Debounced: a held arrow key commits every step, so one press would be dozens of writes, each
    // invalidating the read cache.
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
    }
    stageVramBudgetSave(vramPercentToFraction(next));
    saveTimer.current = setTimeout(() => {
      saveTimer.current = null;
      flushVramBudgetSave()
        ?.then(setSettings)
        .catch((error: unknown) => {
          // The client re-stages it, where the write generation can tell if it is still the newest.
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to save VRAM budget",
          );
        });
    }, 400);
  };

  return (
    <div className="space-y-1">
      <AdvancedGpuSlider
        label="VRAM Budget"
        value={percent}
        min={vramFractionToPercent(settings.minFraction)}
        max={vramFractionToPercent(settings.maxFraction)}
        step={VRAM_BUDGET_PERCENT_STEP}
        format={(v) => `${v}%`}
        onChange={commit}
        onDraft={setDraftPercent}
        disabled={locked || modelLoading}
        info={
          <div className="flex flex-col gap-1.5">
            <div>
              Shared across models. Changes apply on the next model load.
              Default {defaultPercent}%.
            </div>
            <div>
              Sets how much GPU memory Unsloth may use. Each engine also
              reserves memory for running the model, so 100% does not guarantee
              full GPU residency.
            </div>
          </div>
        }
      />
      {image && (
        <p className="text-ui-11 text-muted-foreground">
          Shared across models. Applies on the next load or Reapply. Image
          models keep additional memory headroom.
        </p>
      )}
      {shownPercent !== defaultPercent && (
        <p id={adviceId} className="text-ui-11 text-amber-500">
          {shownPercent > defaultPercent
            ? "Above the default leaves less headroom and may cause an out-of-memory error."
            : "Below the default leaves more memory free, but may offload more of the model to CPU and slow generation."}
        </p>
      )}
      {settings.isStored && (
        <button
          type="button"
          disabled={locked || modelLoading}
          onClick={resetBudget}
          className="text-ui-11 text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Reset to the server default
        </button>
      )}
      {settings.reloadRequired && (
        <p className="text-ui-11 text-muted-foreground">
          The loaded model was sized with a different budget. Reload it to apply
          this one.
        </p>
      )}
    </div>
  );
}
