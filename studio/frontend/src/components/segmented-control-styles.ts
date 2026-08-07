
export type SegmentedSize = "compact" | "default";

export const segmentedTrackHeight: Record<SegmentedSize, string> = {
  compact: "h-8",
  default: "h-9",
};

export const segmentedSegmentLabel =
  "whitespace-nowrap px-3 text-ui-12p5" as const;
