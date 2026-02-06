export const HANDLE_IDS = {
  // data flow lanes
  dataIn: "data-in",
  dataOut: "data-out",
  // semantic dependency lanes
  semanticIn: "semantic-in",
  semanticOut: "semantic-out",
  // llm prompt/scorer lanes
  llmPromptIn: "llm-prompt-in",
  llmSystemIn: "llm-system-in",
  llmInputOut: "llm-input-out",
} as const;

export type CanvasHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];

export function getLlmJudgeScoreHandleId(index: number): string {
  return `llm-judge-score-in-${index}`;
}
