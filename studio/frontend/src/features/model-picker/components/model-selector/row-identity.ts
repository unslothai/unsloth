// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure identity helpers for picker rows: id and quant comparison, and the selected/loaded state
// of a row that stands for one exact quant. No React/DOM deps so they stay easy to test.

export function normalizeModelIdForPicker(modelId: string): string {
  const trimmed = modelId.trim();
  const slashPath = trimmed.replace(/\\/g, "/").replace(/\/+$/, "");
  const caseInsensitive =
    !/^(\/|\.{1,2}\/|~\/)/.test(slashPath) ||
    /^[A-Za-z]:\//.test(slashPath) ||
    slashPath.startsWith("//") ||
    /^\/mnt\/[A-Za-z](?:\/|$)/.test(slashPath);
  return caseInsensitive ? slashPath.toLowerCase() : slashPath;
}

export function modelIdsMatchForPicker(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  return Boolean(
    left &&
      right &&
      normalizeModelIdForPicker(left) === normalizeModelIdForPicker(right),
  );
}

export function normalizeGgufVariantForPicker(
  variant: string | null | undefined,
) {
  return variant?.trim().toLowerCase() ?? "";
}

export function ggufVariantsMatchForPicker(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  return (
    normalizeGgufVariantForPicker(left) === normalizeGgufVariantForPicker(right)
  );
}

/** Selected and loaded state for a row that names one quant, such as a pinned quant or an On
 *  Device repo holding a single quant. Loaded is exact: only the running quant wears the badge.
 *  Selected follows the picker's value, minus the case where the repo is running a different
 *  quant. Compare panes hold a staged value while another pane's model is resident, so a repo
 *  that is not the loaded one stays selected on its value alone. */
export function soleQuantRowState({
  pickerValue,
  repoId,
  quant,
  loadedModelId,
  activeGgufVariant,
}: {
  pickerValue: string | null | undefined;
  repoId: string;
  quant: string;
  loadedModelId: string | null | undefined;
  activeGgufVariant: string | null | undefined;
}): { selected: boolean; loaded: boolean } {
  const repoIsLoaded = modelIdsMatchForPicker(loadedModelId, repoId);
  const quantIsLoaded =
    repoIsLoaded && ggufVariantsMatchForPicker(activeGgufVariant, quant);
  return {
    selected: pickerValue === repoId && (!repoIsLoaded || quantIsLoaded),
    loaded: quantIsLoaded,
  };
}
