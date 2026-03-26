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
  /** First ~500 chars of first user message for search indexing */
  searchText?: string;
  /** Folder this thread belongs to */
  folderId?: string;
  /** Pin thread to top of sidebar */
  pinned?: boolean;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
  /** User feedback on assistant messages */
  feedback?: "thumbs_up" | "thumbs_down";
}

export interface FolderRecord {
  id: string;
  name: string;
  createdAt: number;
}

export interface PromptRecord {
  id: string;
  name: string;
  content: string;
  variables: string[];
  tags: string[];
  createdAt: number;
}

export interface MemoryRecord {
  id: string;
  content: string;
  enabled: boolean;
  createdAt: number;
  updatedAt: number;
}
