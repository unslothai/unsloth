export type ScaleMode = "linear" | "log";
export type OutlierMode = "none" | "p99" | "p95";

export type LossHistoryItem = { step: number; loss: number };
export type SmoothedLossItem = LossHistoryItem & { smoothed: number };

export interface TrainingChartSeries {
  lossHistory: LossHistoryItem[];
  lrHistory: { step: number; lr: number }[];
  gradNormHistory: { step: number; gradNorm: number }[];
  evalLossHistory: { step: number; loss: number }[];
}

export interface ViewSettingsState {
  effectiveWindowSize: number;
  minWindow: number;
  allStepsLength: number;
  setWindowSize: (value: number) => void;
}
