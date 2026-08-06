


import { create } from "zustand";

export type PromptQueueUIEntry = {
  runId: string;
  current: number;
  total: number;
  local: boolean;
  temporary: boolean;
  dispatched: boolean;
};

export type PromptQueueUIItemStatus =
  | "queued"
  | "next"
  | "waiting"
  | "running";

export type PromptQueueUIItem = {
  id: string;
  runId: string;
  prompt: string;
  position: number;
  total: number;
  status: PromptQueueUIItemStatus;
  threadIds: string[];
  canEdit: boolean;
  canRemove: boolean;
};

export interface PromptQueueUIState {
  byThreadId: Record<string, PromptQueueUIEntry>;
  current: number;
  total: number;
  items: PromptQueueUIItem[];
  isRunning: boolean;
}

export const usePromptQueueUI = create<PromptQueueUIState>(() => ({
  byThreadId: {},
  current: 0,
  total: 0,
  items: [],
  isRunning: false,
}));
