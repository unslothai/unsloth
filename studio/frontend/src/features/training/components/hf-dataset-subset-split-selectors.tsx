// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Field, FieldLabel } from "@/components/ui/field";
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
import { useEffect } from "react";

type Props = {
  variant: "wizard" | "studio";
  enabled: boolean;
  datasetName: string | null;
  accessToken?: string;
  datasetSubset: string | null;
  setDatasetSubset: (v: string | null) => void;
  datasetSplit: string | null;
  setDatasetSplit: (v: string | null) => void;
  datasetEvalSplit: string | null;
  setDatasetEvalSplit: (v: string | null) => void;
};

export function HfDatasetSubsetSplitSelectors({
  variant,
  enabled,
  datasetName,
  accessToken,
  datasetSubset,
  setDatasetSubset,
  datasetSplit,
  setDatasetSplit,
  datasetEvalSplit,
  setDatasetEvalSplit,
}: Props) {
  const t = useT();
  const {
    subsets: hfSubsets,
    splits: hfSplits,
    isLoading,
    error,
  } = useHfDatasetSplits(enabled ? datasetName : null, datasetSubset, {
    accessToken,
  });
  const showPlaceholderDropdowns =
    variant === "studio" && !enabled && !datasetName;

  // Auto-select subset and split in one pass to avoid racing effects
  useEffect(() => {
    if (hfSubsets.length === 0) return;

    if (!datasetSubset || !hfSubsets.includes(datasetSubset)) {
      const pick = hfSubsets.includes("default") ? "default" : hfSubsets[0];
      setDatasetSubset(pick);
      return;
    }

    // Split, only once the subset is settled.
    if (hfSplits.length === 0) return;
    if (!datasetSplit || !hfSplits.includes(datasetSplit)) {
      const pick = hfSplits.includes("train") ? "train" : hfSplits[0];
      setDatasetSplit(pick);
    }
  }, [
    hfSubsets,
    hfSplits,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
  ]);

  const showDropdowns = !isLoading && !error && hfSubsets.length > 0;

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
    </>
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
