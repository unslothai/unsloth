export type ModelType = "base" | "lora";

export type ChatView =
  | { mode: "single"; threadId?: string }
  | { mode: "compare"; pairId: string };

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  pairId?: string;
  archived: boolean;
  createdAt: number;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}

export interface RuntimeBridge {
  switchToThread: (threadId: string) => void;
  switchToNewThread: () => void;
}
