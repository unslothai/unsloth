// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Picking loads the model right away through chat's runtime.

import { BenchModelPicker } from "@/features/benchmarks/components/bench-model-picker";
import { useT } from "@/i18n";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";
import type { EvalsModel } from "../use-evals-model";

export function EvalsModelPicker({ evals }: { evals: EvalsModel }) {
  const t = useT();
  const isRunning = useBenchmarkRuntimeStore((s) => s.isRunning);
  return (
    <BenchModelPicker
      models={evals.models}
      value={evals.selectedModel ?? undefined}
      ggufVariant={
        evals.selectedModel != null &&
        evals.selectedModel === evals.residentModel
          ? evals.activeGgufVariant
          : evals.selectedGgufVariant
      }
      loaded={
        !evals.selectedModel || evals.selectedModel === evals.residentModel
      }
      locked={isRunning}
      variant="ghost"
      className="!h-[calc(34px*var(--ui-space-scale,1))] max-w-full"
      placeholder={t("benchmark.pickModelPlaceholder")}
      onPick={evals.pick}
    />
  );
}
