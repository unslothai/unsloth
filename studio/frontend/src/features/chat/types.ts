// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

export type ModelType = "base" | "lora";

export type ChatView =
  | { mode: "single"; threadId?: string; newThreadNonce?: string }
  | { mode: "compare"; pairId: string };

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  modelId?: string;
  pairId?: string;
  archived: boolean;
  createdAt: number;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}
