


export type HfDatasetOptionSelection =
  | { type: "subset"; value: string }
  | { type: "split"; value: string };

export function nextHfDatasetOptionSelection({
  subsets,
  splits,
  selectedSubset,
  selectedSplit,
}: {
  subsets: string[];
  splits: string[];
  selectedSubset: string | null;
  selectedSplit: string | null;
}): HfDatasetOptionSelection | null {
  if (subsets.length === 0) {
    return null;
  }
  const hasSelectedSubset =
    selectedSubset !== null && subsets.includes(selectedSubset);
  if (!hasSelectedSubset) {
    return {
      type: "subset",
      value: subsets.includes("default") ? "default" : subsets[0],
    };
  }
  if (
    splits.length === 0 ||
    (selectedSplit && splits.includes(selectedSplit))
  ) {
    return null;
  }
  return {
    type: "split",
    value: splits.includes("train") ? "train" : splits[0],
  };
}
