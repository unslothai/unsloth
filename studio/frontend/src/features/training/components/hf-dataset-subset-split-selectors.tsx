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
}: Props) {
  const {
    subsets: hfSubsets,
    splits: hfSplits,
    hasMultipleSubsets,
    hasMultipleSplits,
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
          Could not fetch dataset splits: {error}
        </div>
      )}

      {!isLoading && !error && hasMultipleSubsets && (
        <>
          {variant === "wizard" ? (
            <Field>
              <FieldLabel className="flex items-center gap-1.5">
                Subset
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
                    This dataset has multiple subsets. Select which one to use
                    for training.
                  </TooltipContent>
                </Tooltip>
              </FieldLabel>
              <Select
                value={datasetSubset ?? ""}
                onValueChange={(v) => setDatasetSubset(v || null)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a subset..." />
                </SelectTrigger>
                <SelectContent>
                  {hfSubsets.map((subset) => (
                    <SelectItem key={subset} value={subset}>
                      {subset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          ) : (
            <div className="flex flex-col gap-1.5">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                Subset
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
                    This dataset has multiple subsets. Select which one to use
                    for training.
                  </TooltipContent>
                </Tooltip>
              </span>
              <Select
                value={datasetSubset ?? ""}
                onValueChange={(v) => setDatasetSubset(v || null)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a subset..." />
                </SelectTrigger>
                <SelectContent>
                  {hfSubsets.map((subset) => (
                    <SelectItem key={subset} value={subset}>
                      {subset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </>
      )}

      {!isLoading && !error && hasMultipleSplits && (
        <>
          {variant === "wizard" ? (
            <Field>
              <FieldLabel className="flex items-center gap-1.5">
                Split
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
                    Select which split of the dataset to use for training.
                  </TooltipContent>
                </Tooltip>
              </FieldLabel>
              <Select
                value={datasetSplit ?? ""}
                onValueChange={(v) => setDatasetSplit(v || null)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a split..." />
                </SelectTrigger>
                <SelectContent>
                  {hfSplits.map((split) => (
                    <SelectItem key={split} value={split}>
                      {split}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
          ) : (
            <div className="flex flex-col gap-1.5">
              <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                Split
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
                    Select which split of the dataset to use for training.
                  </TooltipContent>
                </Tooltip>
              </span>
              <Select
                value={datasetSplit ?? ""}
                onValueChange={(v) => setDatasetSplit(v || null)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a split..." />
                </SelectTrigger>
                <SelectContent>
                  {hfSplits.map((split) => (
                    <SelectItem key={split} value={split}>
                      {split}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </>
      )}
    </>
  );
}
