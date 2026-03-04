import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Field,
  FieldLabel,
} from "@/components/ui/field";
import { useHfDatasetSplits } from "@/hooks";
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
  datasetSliceStart?: string | null;
  setDatasetSliceStart?: (v: string | null) => void;
  datasetSliceEnd?: string | null;
  setDatasetSliceEnd?: (v: string | null) => void;
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
  datasetSliceStart,
  setDatasetSliceStart,
  datasetSliceEnd,
  setDatasetSliceEnd,
}: Props) {
  const {
    subsets: hfSubsets,
    splits: hfSplits,
    hasMultipleSubsets,
    isLoading,
    error,
  } = useHfDatasetSplits(enabled ? datasetName : null, datasetSubset, {
    accessToken,
  });

  useEffect(() => {
    if (hfSubsets.length === 1 && datasetSubset !== hfSubsets[0]) {
      setDatasetSubset(hfSubsets[0]);
    }
  }, [hfSubsets, datasetSubset, setDatasetSubset]);

  useEffect(() => {
    if (hfSplits.length === 0) return;
    if (hasMultipleSubsets && !datasetSubset) return;
    if (hfSplits.length === 1 && datasetSplit !== hfSplits[0]) {
      setDatasetSplit(hfSplits[0]);
    } else if (!datasetSplit && hfSplits.includes("train")) {
      setDatasetSplit("train");
    } else if (!datasetSplit) {
      setDatasetSplit(hfSplits[0]);
    }
  }, [
    hfSplits,
    hasMultipleSubsets,
    datasetSubset,
    datasetSplit,
    setDatasetSplit,
  ]);

  if (!enabled || !datasetName) return null;

  const showDropdowns = !isLoading && !error && hfSubsets.length > 0;

  return (
    <>
      {isLoading && (
        <div
          className={
            variant === "wizard"
              ? "flex items-center gap-2 text-xs text-muted-foreground py-1"
              : "flex items-center gap-2 rounded-lg border bg-muted/20 px-3.5 py-3 text-xs text-muted-foreground"
          }
        >
          <Spinner className="size-3.5" />
          Loading dataset configs and splits...
        </div>
      )}

      {error && (
        <div
          className={
            variant === "wizard"
              ? "rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
              : "rounded-lg border border-amber-200 bg-amber-50 px-3.5 py-2.5 text-xs text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-400"
          }
        >
          {error}
        </div>
      )}

      {showDropdowns && (
        <>
          {variant === "studio" ? (
            <div className="grid gap-3 sm:grid-cols-2">
              <SelectorDropdown
                variant={variant}
                label="Subset"
                tooltip="Select which subset (config) of the dataset to use."
                value={datasetSubset}
                onChange={setDatasetSubset}
                options={hfSubsets}
                placeholder="Select a subset..."
              />
              <SelectorDropdown
                variant={variant}
                label="Train Split"
                tooltip="Select which split to use for training."
                value={datasetSplit}
                onChange={setDatasetSplit}
                options={hfSplits}
                placeholder="Select a split..."
              />
            </div>
          ) : (
            <>
              <SelectorDropdown
                variant={variant}
                label="Subset"
                tooltip="Select which subset (config) of the dataset to use."
                value={datasetSubset}
                onChange={setDatasetSubset}
                options={hfSubsets}
                placeholder="Select a subset..."
              />
              <SelectorDropdown
                variant={variant}
                label="Train Split"
                tooltip="Select which split to use for training."
                value={datasetSplit}
                onChange={setDatasetSplit}
                options={hfSplits}
                placeholder="Select a split..."
              />
            </>
          )}
          {variant === "studio" && setDatasetSliceStart && setDatasetSliceEnd ? (
            <div className="grid grid-cols-3 gap-3">
              <SelectorDropdown
                variant={variant}
                label="Eval Split"
                tooltip="Select which split to use for evaluation. None means no evaluation during training."
                value={datasetEvalSplit}
                onChange={setDatasetEvalSplit}
                options={hfSplits}
                placeholder="None"
                allowNone
              />
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  Slice Start
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-foreground/70 hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Inclusive start row index. Leave empty to start from the beginning.
                    </TooltipContent>
                  </Tooltip>
                </span>
                <Input
                  inputMode="numeric"
                  placeholder="0"
                  value={datasetSliceStart ?? ""}
                  onChange={(e) =>
                    setDatasetSliceStart(e.target.value || null)
                  }
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  Slice End
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        className="text-foreground/70 hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Inclusive end row index. Leave empty to use all remaining rows.
                    </TooltipContent>
                  </Tooltip>
                </span>
                <Input
                  inputMode="numeric"
                  placeholder="End"
                  value={datasetSliceEnd ?? ""}
                  onChange={(e) =>
                    setDatasetSliceEnd(e.target.value || null)
                  }
                />
              </div>
            </div>
          ) : (
            <SelectorDropdown
              variant={variant}
              label="Eval Split"
              tooltip="Select which split to use for evaluation. None means no evaluation during training."
              value={datasetEvalSplit}
              onChange={setDatasetEvalSplit}
              options={hfSplits}
              placeholder="None"
              allowNone
            />
          )}
        </>
      )}
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
  allowNone = false,
}: {
  variant: "wizard" | "studio";
  label: string;
  tooltip: string;
  value: string | null;
  onChange: (v: string | null) => void;
  options: string[];
  placeholder: string;
  allowNone?: boolean;
}) {
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
            <TooltipContent className="max-w-xs">
              {tooltip}
            </TooltipContent>
          </Tooltip>
        </FieldLabel>
        <Select
          value={value ?? "_none"}
          onValueChange={(v) => onChange(v === "_none" ? null : v)}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {allowNone && (
              <SelectItem value="_none">None</SelectItem>
            )}
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
    <div className="flex flex-col gap-1.5">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon
                icon={InformationCircleIcon}
                className="size-3"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent>
            {tooltip}
          </TooltipContent>
        </Tooltip>
      </span>
      <Select
        value={value ?? "_none"}
        onValueChange={(v) => onChange(v === "_none" ? null : v)}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {allowNone && (
            <SelectItem value="_none">None</SelectItem>
          )}
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
