export const HANDLE_IDS = {
  // data flow lanes
  dataIn: "data-in",
  dataOut: "data-out",
  // semantic dependency lanes
  semanticIn: "semantic-in",
  semanticOut: "semantic-out",
} as const;

export type CanvasHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];
