// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type LorebookPosition = "before_char" | "after_char" | "at_depth";
export type LorebookLogic = "and_any" | "not_all" | "not_any" | "and_all";

export interface LorebookEntry {
  id: string | number;
  keys: string[];
  secondary_keys: string[];
  content: string;
  comment: string;
  enabled: boolean;
  constant: boolean;
  selective: boolean;
  case_sensitive: boolean;
  /** V3: every key is a regex even without /slashes/. */
  use_regex?: boolean;
  insertion_order: number;
  position: LorebookPosition;
  depth: number;
  selective_logic: LorebookLogic;
  extensions?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Lorebook {
  name: string;
  scan_depth: number;
  token_budget: number;
  recursive_scanning: boolean;
  extensions: Record<string, unknown>;
  entries: LorebookEntry[];
  [key: string]: unknown;
}

export interface DepthPrompt {
  prompt: string;
  depth: number;
  role: "system" | "user" | "assistant";
}

export interface Character {
  id: string;
  name: string;
  description: string;
  personality: string;
  scenario: string;
  firstMes: string;
  alternateGreetings: string[];
  mesExample: string;
  systemPrompt: string;
  postHistoryInstructions: string;
  creatorNotes: string;
  creator: string;
  characterVersion: string;
  tags: string[];
  talkativeness: number;
  favorite: boolean;
  depthPrompt: DepthPrompt | null;
  characterBook: Lorebook | null;
  extensions: Record<string, unknown>;
  hasAvatar: boolean;
  avatarType: string | null;
  avatarRev: number;
  createdAt: number;
  updatedAt: number;
  lastChatAt: number | null;
  chatCount: number;
  cardSpec: string | null;
}

export type CharacterDraft = Omit<
  Character,
  | "id"
  | "hasAvatar"
  | "avatarType"
  | "avatarRev"
  | "createdAt"
  | "updatedAt"
  | "lastChatAt"
  | "chatCount"
  | "cardSpec"
  | "extensions"
>;

export interface Persona {
  id: string;
  name: string;
  description: string;
  hasAvatar: boolean;
  avatarType: string | null;
  avatarRev: number;
  isDefault: boolean;
  createdAt: number;
  updatedAt: number;
}

export type ReplyOrder = "natural" | "list" | "manual";

/** How a character chat is laid out: the plain chat, the speaker's picture beside it, or a visual novel scene. */
export type ChatView = "chat" | "stage" | "novel";

export interface SceneSettings {
  scenario?: string;
  /** Fields merge one by one on the server, so either can be missing. */
  authorsNote?: { text?: string; depth?: number };
  replyOrder?: ReplyOrder;
  allowSelfReplies?: boolean;
  view?: ChatView;
}

export interface CastMember {
  characterId: string;
  position: number;
  muted: boolean;
}

export interface ChatScene {
  threadId: string;
  personaId: string | null;
  settings: SceneSettings;
  cast: CastMember[];
}

export interface CharacterChatSummary {
  threadId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  archived: boolean;
  castSize: number;
  messageCount: number;
}

/** What a character chat message carries in `metadata.custom.roleplay`. */
export type RoleplayMessageMeta =
  | { kind: "opening" }
  | { kind: "turn"; speakerId: string }
  | { kind: "reply"; speakerId: string; speakerName: string; greeting?: number }
  | { kind: "user"; personaId: string | null; personaName: string };

/** A persona as a chat uses it: a stored persona, or the account's own name when there is none. */
export interface ResolvedPersona {
  id: string | null;
  name: string;
  description: string;
  hasAvatar: boolean;
  avatarRev: number;
}
