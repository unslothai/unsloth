// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ModelType = "base" | "lora" | "model1" | "model2";

export type ChatView =
  | {
      mode: "project";
      projectId: string;
    }
  | {
      mode: "single";
      threadId?: string;
      newThreadNonce?: string;
      projectId?: string | null;
    }
  | { mode: "compare"; pairId: string; projectId?: string | null };

export interface ProjectRecord {
  id: string;
  name: string;
  instructions?: string;
  rootPath?: string | null;
  sandboxPath?: string | null;
  archived: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  modelId?: string;
  modelGgufVariant?: string | null;
  pairId?: string;
  projectId?: string | null;
  archived: boolean;
  createdAt: number;
  updatedAt?: number;
  /** OpenAI shell container from a prior response; cleared on `container_invalidated`. */
  openaiCodeExecContainerId?: string | null;
  /** Anthropic code_execution container from a prior response; cleared on `container_invalidated`. */
  anthropicCodeExecContainerId?: string | null;
  forkedFromThreadId?: string | null;
  forkedFromMessageId?: string | null;
  /** Null on forks made before the column existed, which show no divider. */
  forkBoundaryMessageId?: string | null;
  forkTitleBase?: string | null;
  modifiedAt?: number | null;
  settings?:
    | import("./utils/thread-scoped-settings").ThreadScopedSettings
    | null;
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

export interface ParsedConversation {
  title: string;
  threadId: string;
  messages: MessageRecord[];
  archived?: boolean;
  createdAt?: number;
  thread?: Partial<Omit<ThreadRecord, "id" | "title" | "archived">>;
}
