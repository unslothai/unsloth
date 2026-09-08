// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ManualDatasetOptionError = "invalid" | "required" | "too_long";

export type ManualDatasetOptionDraft = {
  committedValue: string | null;
  value: string;
  error: ManualDatasetOptionError | null;
};

export type ManualDatasetOptionDrafts = {
  subset: ManualDatasetOptionDraft;
  split: ManualDatasetOptionDraft;
  evalSplit: ManualDatasetOptionDraft;
};

type ManualDatasetOptionDraftSources = {
  datasetSubset: string | null;
  datasetSplit: string | null;
  datasetEvalSplit: string | null;
  defaultSplit: string;
};

const MAX_OPTION_LENGTH = 128;
const CONFIG_FORBIDDEN_PATTERN = /[<>:/\\|?*]/;
const PATH_SEPARATOR_PATTERN = /[/\\]/;
const UNSAFE_UNICODE_PATTERN = /[\p{Cc}\p{Cf}\p{Cs}]/u;
const SPLIT_NAME_PATTERN = String.raw`[\p{L}\p{N}_]+(?:\.[\p{L}\p{N}_]+)*`;
const BARE_SPLIT_PATTERN = new RegExp(String.raw`^${SPLIT_NAME_PATTERN}$`, "u");
const SPLIT_BOUNDARY_PATTERN = String.raw`-?\d(?:_?\d)*%?`;
const SPLIT_PART_PATTERN = new RegExp(
  String.raw`^(${SPLIT_NAME_PATTERN})(?:\[(${SPLIT_BOUNDARY_PATTERN})?:(${SPLIT_BOUNDARY_PATTERN})?\])?(?:\((closest|pct1_dropremainder)\))?$`,
  "u",
);

export function manualDatasetSplitDefault(
  requireExplicitSplit: boolean,
): string {
  return requireExplicitSplit ? "" : "train";
}

export function manualDatasetOptionsFormKey(
  datasetName: string,
  localPath: string | null | undefined,
  preferLocalCache: boolean,
): string {
  return JSON.stringify([datasetName, localPath, preferLocalCache]);
}

function createManualDatasetOptionDraft(
  committedValue: string | null,
  fallback: string,
): ManualDatasetOptionDraft {
  return {
    committedValue,
    value: committedValue ?? fallback,
    error: null,
  };
}

export function createManualDatasetOptionDrafts({
  datasetSubset,
  datasetSplit,
  datasetEvalSplit,
  defaultSplit,
}: ManualDatasetOptionDraftSources): ManualDatasetOptionDrafts {
  return {
    subset: createManualDatasetOptionDraft(datasetSubset, ""),
    split: createManualDatasetOptionDraft(datasetSplit, defaultSplit),
    evalSplit: createManualDatasetOptionDraft(datasetEvalSplit, ""),
  };
}

function synchronizeManualDatasetOptionDraft(
  draft: ManualDatasetOptionDraft,
  committedValue: string | null,
  fallback: string,
): ManualDatasetOptionDraft {
  return draft.committedValue === committedValue
    ? draft
    : createManualDatasetOptionDraft(committedValue, fallback);
}

export function synchronizeManualDatasetOptionDrafts(
  drafts: ManualDatasetOptionDrafts,
  {
    datasetSubset,
    datasetSplit,
    datasetEvalSplit,
    defaultSplit,
  }: ManualDatasetOptionDraftSources,
): ManualDatasetOptionDrafts {
  const subset = synchronizeManualDatasetOptionDraft(
    drafts.subset,
    datasetSubset,
    "",
  );
  const split = synchronizeManualDatasetOptionDraft(
    drafts.split,
    datasetSplit,
    defaultSplit,
  );
  const evalSplit = synchronizeManualDatasetOptionDraft(
    drafts.evalSplit,
    datasetEvalSplit,
    "",
  );
  return subset === drafts.subset &&
    split === drafts.split &&
    evalSplit === drafts.evalSplit
    ? drafts
    : { subset, split, evalSplit };
}

function commonError(value: string): ManualDatasetOptionError | null {
  if (Array.from(value).length > MAX_OPTION_LENGTH) {
    return "too_long";
  }
  if (UNSAFE_UNICODE_PATTERN.test(value)) {
    return "invalid";
  }
  if (value === "." || value === ".." || PATH_SEPARATOR_PATTERN.test(value)) {
    return "invalid";
  }
  return null;
}

export function normalizeManualDatasetOption(value: string): string {
  return value.trim();
}

function validPercentBoundary(
  value: string | undefined,
  percentMode: boolean,
): boolean {
  if (!value || !percentMode) {
    return true;
  }
  const parsed = Number(value.replace(/%$/u, "").replaceAll("_", ""));
  return Number.isInteger(parsed) && Math.abs(parsed) <= 100;
}

function validSplitInstruction(value: string): boolean {
  let percentRounding: string | null = null;
  const parts = value.split(/\s*\+\s*/u);
  for (const part of parts) {
    const match = SPLIT_PART_PATTERN.exec(part);
    if (!match) {
      return false;
    }
    const [, , from, to, rounding] = match;
    const usesPercent = from?.endsWith("%") || to?.endsWith("%");
    if (
      !validPercentBoundary(from, Boolean(usesPercent)) ||
      !validPercentBoundary(to, Boolean(usesPercent)) ||
      (rounding !== undefined && !usesPercent)
    ) {
      return false;
    }
    if (usesPercent) {
      const effectiveRounding = rounding ?? "closest";
      if (percentRounding !== null && percentRounding !== effectiveRounding) {
        return false;
      }
      percentRounding = effectiveRounding;
    }
  }
  return true;
}

export function validateManualDatasetSubset(
  value: string,
): ManualDatasetOptionError | null {
  const normalized = normalizeManualDatasetOption(value);
  const error = commonError(normalized);
  if (error) {
    return error;
  }
  return CONFIG_FORBIDDEN_PATTERN.test(normalized) ? "invalid" : null;
}

export function validateManualDatasetSplit(
  value: string,
  required: boolean,
  allowInstructions = true,
): ManualDatasetOptionError | null {
  const normalized = normalizeManualDatasetOption(value);
  if (!normalized) {
    return required ? "required" : null;
  }
  const error = commonError(normalized);
  if (error) {
    return error;
  }
  const valid = allowInstructions
    ? validSplitInstruction(normalized)
    : BARE_SPLIT_PATTERN.test(normalized);
  return valid ? null : "invalid";
}
