// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Field, FieldError, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useHfDatasetSplits } from "@/hooks";
import { useT } from "@/i18n";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type KeyboardEvent, useEffect, useState } from "react";
import { nextHfDatasetOptionSelection } from "../lib/hf-dataset-option-selection";
import {
  type ManualDatasetOptionDrafts,
  type ManualDatasetOptionError,
  createManualDatasetOptionDrafts,
  manualDatasetOptionsFormKey,
  manualDatasetSplitDefault,
  normalizeManualDatasetOption,
  synchronizeManualDatasetOptionDrafts,
  validateManualDatasetSplit,
  validateManualDatasetSubset,
} from "../lib/manual-dataset-options";

type Props = {
  variant: "wizard" | "studio";
  enabled: boolean;
  datasetName: string | null;
  accessToken?: string;
  localPath?: string | null;
  online?: boolean;
  preferLocalCache?: boolean;
  datasetSubset: string | null;
  setDatasetSubset: (v: string | null) => void;
  datasetSplit: string | null;
  setDatasetSplit: (v: string | null) => void;
  datasetEvalSplit: string | null;
  setDatasetEvalSplit: (v: string | null) => void;
  datasetStreaming: boolean;
  setManualDatasetOptionsValid: (v: boolean) => void;
  markManualDatasetOptionsEdited: (optionsValid: boolean) => void;
};

function manualDatasetDraftsAreValid(
  subset: string,
  split: string,
  evalSplit: string,
  requireExplicitSplit: boolean,
  allowSplitInstructions: boolean,
) {
  return (
    validateManualDatasetSubset(subset) === null &&
    validateManualDatasetSplit(
      split,
      requireExplicitSplit,
      allowSplitInstructions,
    ) === null &&
    validateManualDatasetSplit(evalSplit, false, allowSplitInstructions) ===
      null
  );
}

export function HfDatasetSubsetSplitSelectors({
  variant,
  enabled,
  datasetName,
  accessToken,
  localPath,
  online = true,
  preferLocalCache = false,
  datasetSubset,
  setDatasetSubset,
  datasetSplit,
  setDatasetSplit,
  datasetEvalSplit,
  setDatasetEvalSplit,
  datasetStreaming,
  setManualDatasetOptionsValid,
  markManualDatasetOptionsEdited,
}: Props) {
  const t = useT();
  const {
    subsets: hfSubsets,
    splits: hfSplits,
    isLoading,
    error,
    requiresManualEntry,
  } = useHfDatasetSplits(enabled ? datasetName : null, datasetSubset, {
    accessToken,
    localPath,
    online,
    preferLocalCache,
  });
  const showPlaceholderDropdowns =
    variant === "studio" && !enabled && !datasetName;

  // Auto-select subset and split in one pass to avoid racing effects
  useEffect(() => {
    const next = nextHfDatasetOptionSelection({
      subsets: hfSubsets,
      splits: hfSplits,
      selectedSubset: datasetSubset,
      selectedSplit: datasetSplit,
    });
    if (next?.type === "subset") {
      setDatasetSubset(next.value);
    } else if (next?.type === "split") {
      setDatasetSplit(next.value);
    }
  }, [
    hfSubsets,
    hfSplits,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
  ]);

  const showDropdowns =
    isLoading === false && error === null && hfSubsets.length > 0;

  const selectorFields = (disabled = false) => (
    <>
      <SelectorDropdown
        variant={variant}
        label={t("studio.dataset.selectors.subset")}
        tooltip={t("studio.dataset.selectors.subsetTooltip")}
        value={disabled ? null : datasetSubset}
        onChange={setDatasetSubset}
        options={disabled ? [] : hfSubsets}
        placeholder={t("studio.dataset.selectors.selectSubset")}
        disabled={disabled}
      />
      <SelectorDropdown
        variant={variant}
        label={t("studio.dataset.selectors.trainSplit")}
        tooltip={t("studio.dataset.selectors.trainSplitTooltip")}
        value={disabled ? null : datasetSplit}
        onChange={setDatasetSplit}
        options={disabled ? [] : hfSplits}
        placeholder={t("studio.dataset.selectors.selectSplit")}
        disabled={disabled}
      />
      <SelectorDropdown
        variant={variant}
        label={t("studio.dataset.selectors.evaluationSplit")}
        tooltip={t("studio.dataset.selectors.evaluationSplitTooltip")}
        value={disabled ? null : datasetEvalSplit}
        onChange={setDatasetEvalSplit}
        options={disabled ? [] : hfSplits}
        placeholder={t("studio.dataset.selectors.none")}
        noneLabel={t("studio.dataset.selectors.none")}
        allowNone={true}
        disabled={disabled}
      />
    </>
  );

  return (
    <>
      {showPlaceholderDropdowns && (
        <div className="grid min-w-0 gap-3 sm:grid-cols-3">
          {selectorFields(true)}
        </div>
      )}

      {isLoading && (
        <div
          className={
            variant === "wizard"
              ? "flex items-center gap-2 text-xs text-muted-foreground py-1"
              : "flex min-w-0 items-center gap-2 rounded-lg border bg-muted/20 px-3.5 py-3 text-xs text-muted-foreground"
          }
        >
          <Spinner className="size-3.5" />
          {t("studio.dataset.selectors.loading")}
        </div>
      )}

      {error && (
        <div
          className={
            variant === "wizard"
              ? "rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
              : "min-w-0 rounded-lg border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
          }
        >
          {error}
        </div>
      )}

      {showDropdowns &&
        (variant === "studio" ? (
          <div className="grid min-w-0 gap-3 sm:grid-cols-3">
            {selectorFields()}
          </div>
        ) : (
          selectorFields()
        ))}

      {requiresManualEntry && datasetName && (
        <ManualDatasetOptions
          key={manualDatasetOptionsFormKey(
            datasetName,
            localPath,
            preferLocalCache,
          )}
          variant={variant}
          datasetSubset={datasetSubset}
          setDatasetSubset={setDatasetSubset}
          datasetSplit={datasetSplit}
          setDatasetSplit={setDatasetSplit}
          datasetEvalSplit={datasetEvalSplit}
          setDatasetEvalSplit={setDatasetEvalSplit}
          datasetStreaming={datasetStreaming}
          setManualDatasetOptionsValid={setManualDatasetOptionsValid}
          markManualDatasetOptionsEdited={markManualDatasetOptionsEdited}
          requireExplicitSplit={preferLocalCache}
        />
      )}
    </>
  );
}

function ManualDatasetOptions({
  variant,
  datasetSubset,
  setDatasetSubset,
  datasetSplit,
  setDatasetSplit,
  datasetEvalSplit,
  setDatasetEvalSplit,
  datasetStreaming,
  setManualDatasetOptionsValid,
  markManualDatasetOptionsEdited,
  requireExplicitSplit,
}: Pick<
  Props,
  | "variant"
  | "datasetSubset"
  | "setDatasetSubset"
  | "datasetSplit"
  | "setDatasetSplit"
  | "datasetEvalSplit"
  | "setDatasetEvalSplit"
  | "datasetStreaming"
  | "setManualDatasetOptionsValid"
  | "markManualDatasetOptionsEdited"
> & {
  requireExplicitSplit: boolean;
}) {
  const t = useT();
  const defaultSplit = manualDatasetSplitDefault(requireExplicitSplit);
  const allowSplitInstructions = !datasetStreaming;
  const draftSources = {
    datasetSubset,
    datasetSplit,
    datasetEvalSplit,
    defaultSplit,
  };
  const [storedDrafts, setStoredDrafts] = useState<ManualDatasetOptionDrafts>(
    () => createManualDatasetOptionDrafts(draftSources),
  );
  const drafts = synchronizeManualDatasetOptionDrafts(
    storedDrafts,
    draftSources,
  );
  if (drafts !== storedDrafts) {
    // Persist observed prop transitions so an A -> B -> A cycle cannot revive
    // an obsolete draft. The identity guard makes this a single adjustment.
    setStoredDrafts(drafts);
  }
  const subsetDraft = drafts.subset.value;
  const splitDraft = drafts.split.value;
  const evalDraft = drafts.evalSplit.value;
  const subsetError = drafts.subset.error;
  const splitError = drafts.split.error;
  const evalError = drafts.evalSplit.error;

  useEffect(() => {
    setManualDatasetOptionsValid(
      manualDatasetDraftsAreValid(
        subsetDraft,
        splitDraft,
        evalDraft,
        requireExplicitSplit,
        allowSplitInstructions,
      ),
    );
  }, [
    subsetDraft,
    splitDraft,
    evalDraft,
    requireExplicitSplit,
    allowSplitInstructions,
    setManualDatasetOptionsValid,
  ]);

  useEffect(
    () => () => setManualDatasetOptionsValid(true),
    [setManualDatasetOptionsValid],
  );

  const errorMessage = (error: ManualDatasetOptionError | null) => {
    if (error === "required") {
      return t("studio.dataset.selectors.manualRequired");
    }
    if (error === "too_long") {
      return t("studio.dataset.selectors.manualTooLong");
    }
    return error === "invalid"
      ? t("studio.dataset.selectors.manualInvalid")
      : null;
  };
  const blurOnEnter = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      event.currentTarget.blur();
    }
  };
  const commitSubset = () => {
    const nextError = validateManualDatasetSubset(subsetDraft);
    if (nextError) {
      setStoredDrafts({
        ...drafts,
        subset: { ...drafts.subset, error: nextError },
      });
      setManualDatasetOptionsValid(false);
      return;
    }
    const normalized = normalizeManualDatasetOption(subsetDraft);
    const value = normalized || null;
    if (value === datasetSubset) {
      setStoredDrafts({
        ...drafts,
        subset: {
          committedValue: value,
          value: normalized,
          error: null,
        },
      });
      return;
    }
    const nextDrafts: ManualDatasetOptionDrafts = {
      subset: {
        committedValue: value,
        value: normalized,
        error: null,
      },
      split: {
        committedValue: null,
        value: defaultSplit,
        error: null,
      },
      evalSplit: {
        committedValue: null,
        value: "",
        error: null,
      },
    };
    setDatasetSubset(value);
    // It nulls datasetEvalSplit but not evalSteps, so this is what stops evaluation staying
    // armed with no split (a 422 once streaming is on). setDatasetSplit(null) is not needed:
    // it only adds a runDatasetCheck against an assumed "train" split.
    setDatasetEvalSplit(null);
    setStoredDrafts(nextDrafts);
    setManualDatasetOptionsValid(
      manualDatasetDraftsAreValid(
        nextDrafts.subset.value,
        nextDrafts.split.value,
        nextDrafts.evalSplit.value,
        requireExplicitSplit,
        allowSplitInstructions,
      ),
    );
  };
  const commitSplit = () => {
    const nextError = validateManualDatasetSplit(
      splitDraft,
      requireExplicitSplit,
      allowSplitInstructions,
    );
    if (nextError) {
      setStoredDrafts({
        ...drafts,
        split: { ...drafts.split, error: nextError },
      });
      setManualDatasetOptionsValid(false);
      return;
    }
    const normalized = normalizeManualDatasetOption(splitDraft);
    const value = normalized || null;
    const nextSplitDraft = value ?? defaultSplit;
    const nextDrafts: ManualDatasetOptionDrafts = {
      ...drafts,
      split: {
        committedValue: value,
        value: nextSplitDraft,
        error: null,
      },
    };
    if (value !== datasetSplit) {
      setDatasetSplit(value);
    }
    setStoredDrafts(nextDrafts);
    setManualDatasetOptionsValid(
      manualDatasetDraftsAreValid(
        nextDrafts.subset.value,
        nextDrafts.split.value,
        nextDrafts.evalSplit.value,
        requireExplicitSplit,
        allowSplitInstructions,
      ),
    );
  };
  const commitEvalSplit = () => {
    const nextError = validateManualDatasetSplit(
      evalDraft,
      false,
      allowSplitInstructions,
    );
    if (nextError) {
      setStoredDrafts({
        ...drafts,
        evalSplit: { ...drafts.evalSplit, error: nextError },
      });
      setManualDatasetOptionsValid(false);
      return;
    }
    const normalized = normalizeManualDatasetOption(evalDraft);
    const value = normalized || null;
    const nextDrafts: ManualDatasetOptionDrafts = {
      ...drafts,
      evalSplit: {
        committedValue: value,
        value: normalized,
        error: null,
      },
    };
    if (value !== datasetEvalSplit) {
      setDatasetEvalSplit(value);
    }
    setStoredDrafts(nextDrafts);
    setManualDatasetOptionsValid(
      manualDatasetDraftsAreValid(
        nextDrafts.subset.value,
        nextDrafts.split.value,
        nextDrafts.evalSplit.value,
        requireExplicitSplit,
        allowSplitInstructions,
      ),
    );
  };

  const fields = (
    <>
      <Field className="gap-1.5" data-invalid={subsetError !== null}>
        <FieldLabel htmlFor={`manual-dataset-subset-${variant}`}>
          {t("studio.dataset.selectors.subset")}
        </FieldLabel>
        <Input
          id={`manual-dataset-subset-${variant}`}
          value={subsetDraft}
          placeholder={t("studio.dataset.selectors.manualSubsetPlaceholder")}
          aria-invalid={subsetError !== null}
          onChange={(event) => {
            const value = event.target.value;
            const nextDrafts: ManualDatasetOptionDrafts = {
              ...drafts,
              subset: { ...drafts.subset, value, error: null },
            };
            setStoredDrafts(nextDrafts);
            markManualDatasetOptionsEdited(
              manualDatasetDraftsAreValid(
                nextDrafts.subset.value,
                nextDrafts.split.value,
                nextDrafts.evalSplit.value,
                requireExplicitSplit,
                allowSplitInstructions,
              ),
            );
          }}
          onBlur={commitSubset}
          onKeyDown={blurOnEnter}
        />
        <FieldError className="text-xs">{errorMessage(subsetError)}</FieldError>
      </Field>
      <Field className="gap-1.5" data-invalid={splitError !== null}>
        <FieldLabel htmlFor={`manual-dataset-split-${variant}`}>
          {t("studio.dataset.selectors.trainSplit")}
        </FieldLabel>
        <Input
          id={`manual-dataset-split-${variant}`}
          value={splitDraft}
          placeholder={t("studio.dataset.selectors.selectSplit")}
          aria-invalid={splitError !== null}
          onChange={(event) => {
            const value = event.target.value;
            const nextDrafts: ManualDatasetOptionDrafts = {
              ...drafts,
              split: { ...drafts.split, value, error: null },
            };
            setStoredDrafts(nextDrafts);
            markManualDatasetOptionsEdited(
              manualDatasetDraftsAreValid(
                nextDrafts.subset.value,
                nextDrafts.split.value,
                nextDrafts.evalSplit.value,
                requireExplicitSplit,
                allowSplitInstructions,
              ),
            );
          }}
          onBlur={commitSplit}
          onKeyDown={blurOnEnter}
        />
        <FieldError className="text-xs">{errorMessage(splitError)}</FieldError>
      </Field>
      <Field className="gap-1.5" data-invalid={evalError !== null}>
        <FieldLabel htmlFor={`manual-dataset-eval-${variant}`}>
          {t("studio.dataset.selectors.evaluationSplit")}
        </FieldLabel>
        <Input
          id={`manual-dataset-eval-${variant}`}
          value={evalDraft}
          placeholder={t("studio.dataset.selectors.none")}
          aria-invalid={evalError !== null}
          onChange={(event) => {
            const value = event.target.value;
            const nextDrafts: ManualDatasetOptionDrafts = {
              ...drafts,
              evalSplit: { ...drafts.evalSplit, value, error: null },
            };
            setStoredDrafts(nextDrafts);
            markManualDatasetOptionsEdited(
              manualDatasetDraftsAreValid(
                nextDrafts.subset.value,
                nextDrafts.split.value,
                nextDrafts.evalSplit.value,
                requireExplicitSplit,
                allowSplitInstructions,
              ),
            );
          }}
          onBlur={commitEvalSplit}
          onKeyDown={blurOnEnter}
        />
        <FieldError className="text-xs">{errorMessage(evalError)}</FieldError>
      </Field>
    </>
  );

  return (
    <div
      className={
        variant === "studio"
          ? "min-w-0 rounded-lg border bg-muted/20 px-3.5 py-3"
          : "rounded-lg border bg-muted/20 px-3 py-3"
      }
    >
      <p className="text-xs font-medium">
        {t("studio.dataset.selectors.manualTitle")}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        {t("studio.dataset.selectors.manualDescription")}
      </p>
      <div className="mt-3 grid min-w-0 gap-3 sm:grid-cols-3">{fields}</div>
    </div>
  );
}

function SelectorDropdown({
  variant,
  label,
  tooltip,
  value,
  onChange,
  options,
  placeholder,
  noneLabel,
  allowNone = false,
  disabled = false,
}: {
  variant: "wizard" | "studio";
  label: string;
  tooltip: string;
  value: string | null;
  onChange: (v: string | null) => void;
  options: string[];
  placeholder: string;
  noneLabel?: string;
  allowNone?: boolean;
  disabled?: boolean;
}) {
  const selectValue = value ?? (allowNone && !disabled ? "_none" : undefined);

  if (variant === "wizard") {
    return (
      <Field>
        <FieldLabel className="flex items-center gap-1.5">
          {label}
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                className="text-muted-foreground/50 hover:text-muted-foreground"
              >
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  className="size-3.5"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
          </Tooltip>
        </FieldLabel>
        <Select
          value={selectValue}
          onValueChange={(v) => onChange(v === "_none" ? null : v)}
          disabled={disabled}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {allowNone && <SelectItem value="_none">{noneLabel}</SelectItem>}
            {options.map((opt) => (
              <SelectItem key={opt} value={opt}>
                {opt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent>{tooltip}</TooltipContent>
        </Tooltip>
      </span>
      <Select
        value={selectValue}
        onValueChange={(v) => onChange(v === "_none" ? null : v)}
        disabled={disabled}
      >
        <SelectTrigger className="w-full min-w-0">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {allowNone && <SelectItem value="_none">{noneLabel}</SelectItem>}
          {options.map((opt) => (
            <SelectItem key={opt} value={opt}>
              {opt}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
