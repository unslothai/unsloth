export interface PlatformChatCompletionRequest {
  chatId: string;
  sessionId: string;
  question: string;
  legacy?: boolean;
  thinking?: "enabled" | "disabled" | "default";
}

export interface PlatformChatCitation {
  id: string;
  chunkId: string | null;
  documentId: string | null;
  datasetId: string | null;
  filename: string;
  text: string;
  page: number | null;
  score: number | null;
}

export interface PlatformChatReference {
  chunks: PlatformChatCitation[];
  documentAggregations: Array<{
    documentId: string | null;
    filename: string;
    count: number | null;
  }>;
}

export interface PlatformChatUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

export type PlatformChatStreamEvent =
  | { type: "text-delta"; delta: string; text: string }
  | { type: "reasoning-start" }
  | { type: "reasoning-delta"; delta: string; text: string }
  | { type: "reasoning-end"; text: string }
  | { type: "reference-update"; reference: PlatformChatReference }
  | { type: "usage"; usage: PlatformChatUsage }
  | {
      type: "final";
      terminal: boolean;
      messageId: string | null;
      chatId: string | null;
      sessionId: string | null;
      text: string;
      reasoning: string;
      reference: PlatformChatReference | null;
      usage: PlatformChatUsage | null;
    }
  | { type: "error"; code: number | string; message: string };

export interface PlatformMindMapNode {
  id: string;
  label: string;
  children: PlatformMindMapNode[];
}

export interface PlatformFeedbackRequest {
  thumbup: boolean;
  feedback?: string;
}
