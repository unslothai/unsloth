// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DatasetSelector } from "@/features/dataset-picker";
import { hfApiToken, useHfTokenStore, useOnlineStatus } from "@/features/hub";
import { FieldLabel } from "@/features/recipe-studio/dialogs/shared/field-label";
import { HfDatasetSubsetSplitSelectors } from "@/features/training";
import { useCallback } from "react";
import { hfDatasetChoice } from "./dataset-choice";

export const HF_DATASET_NODE = "data.hf_dataset";

const noop = () => undefined;

/** The Train page's dataset picker and subset/split dropdowns over the node's repo, subset and split params. */
export function FlowHfDatasetFields({
  params,
  onChange,
}: {
  params: Record<string, unknown>;
  onChange: (patch: Record<string, unknown>) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const { repo, subset, split } = hfDatasetChoice(params);
  // Stable setters: the dropdowns' auto-select effect depends on them.
  const setSubset = useCallback(
    (v: string | null) => onChange({ subset: v ?? "" }),
    [onChange],
  );
  const setSplit = useCallback(
    (v: string | null) => onChange({ split: v ?? "" }),
    [onChange],
  );
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-1.5">
        <FieldLabel label="Dataset repo" />
        <DatasetSelector
          value={repo}
          // A new repo's subsets differ, so the dropdowns pick its default again.
          onValueChange={(next) => onChange({ repo: next, subset: "" })}
        />
      </div>
      <HfDatasetSubsetSplitSelectors
        variant="wizard"
        enabled={repo !== null}
        datasetName={repo}
        accessToken={hfApiToken(hfToken)}
        online={online}
        datasetSubset={subset}
        setDatasetSubset={setSubset}
        datasetSplit={split}
        setDatasetSplit={setSplit}
        datasetEvalSplit={null}
        setDatasetEvalSplit={noop}
        datasetStreaming={false}
        setManualDatasetOptionsValid={noop}
        markManualDatasetOptionsEdited={noop}
        hideEvalSplit={true}
      />
    </div>
  );
}
