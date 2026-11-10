// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { BenchModelPicker } from "@/features/benchmarks/components/bench-model-picker";
import type { EvalsModel } from "../use-evals-model";

export function EvalsModelPicker({ evals }: { evals: EvalsModel }) {
  return (
    <BenchModelPicker
      models={evals.models}
      loraModels={evals.loraModels}
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
      variant="ghost"
      className="!h-[calc(34px*var(--ui-space-scale,1))] max-w-full"
      placeholder="Pick a GGUF to benchmark"
      onPick={evals.pick}
    />
  );
}
