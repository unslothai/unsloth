// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "base" | "lora" | "model1" | "model2";

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
  /**
   * OpenAI shell tool container id captured from a prior response on
   * this thread. When set, the next turn reuses it via
   * `environment.type="container_reference"` so the model can read
   * files it wrote earlier in the conversation. When null/undefined,
   * the next turn auto-creates a fresh container.
   *
   * OpenAI containers expire after ~20 min of inactivity by default;
   * if a stale id is sent, the backend surfaces an
   * `_toolEvent.type="container_invalidated"` and the chat-adapter
   * clears this field so the following turn falls back to auto-create.
   *
   * Anthropic's code-execution path doesn't need this — each turn
   * gets a fresh container server-side.
   */
  openaiCodeExecContainerId?: string | null;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  parentId?: string | null;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}
