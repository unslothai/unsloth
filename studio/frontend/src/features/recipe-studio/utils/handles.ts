import type { Connection } from "@xyflow/react";

export const HANDLE_IDS = {
  // data flow lanes
  dataIn: "data-in",
  dataInTop: "data-in-top",
  dataOut: "data-out",
  dataOutBottom: "data-out-bottom",
  // semantic dependency lanes
  semanticIn: "semantic-in",
  semanticInLeft: "semantic-in-left",
  semanticOut: "semantic-out",
  semanticOutRight: "semantic-out-right",
  // llm prompt/scorer lanes
  llmPromptIn: "llm-prompt-in",
  llmSystemIn: "llm-system-in",
  llmInputOut: "llm-input-out",
} as const;

export type RecipeHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];

export function getLlmJudgeScoreHandleId(index: number): string {
  return `llm-judge-score-in-${index}`;
}

const HANDLE_CANONICAL_MAP: Record<string, string> = {
  [HANDLE_IDS.dataIn]: HANDLE_IDS.dataIn,
  [HANDLE_IDS.dataInTop]: HANDLE_IDS.dataIn,
  [HANDLE_IDS.dataOut]: HANDLE_IDS.dataOut,
  [HANDLE_IDS.dataOutBottom]: HANDLE_IDS.dataOut,
  [HANDLE_IDS.semanticIn]: HANDLE_IDS.semanticIn,
  [HANDLE_IDS.semanticInLeft]: HANDLE_IDS.semanticIn,
  [HANDLE_IDS.semanticOut]: HANDLE_IDS.semanticOut,
  [HANDLE_IDS.semanticOutRight]: HANDLE_IDS.semanticOut,
};

export function normalizeRecipeHandleId(
  handleId: string | null | undefined,
): string | null {
  if (!handleId) {
    return null;
  }
  return HANDLE_CANONICAL_MAP[handleId] ?? handleId;
}

export function normalizeRecipeConnectionHandles(
  connection: Connection,
): Connection {
  return {
    ...connection,
    sourceHandle: normalizeRecipeHandleId(connection.sourceHandle),
    targetHandle: normalizeRecipeHandleId(connection.targetHandle),
  };
}
