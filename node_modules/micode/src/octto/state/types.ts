// src/octto/state/types.ts
import type { Answer, BaseConfig, QuestionType } from "../session";

export const BRANCH_STATUSES = {
  EXPLORING: "exploring",
  DONE: "done",
} as const;

export type BranchStatus = (typeof BRANCH_STATUSES)[keyof typeof BRANCH_STATUSES];

export interface BranchQuestion {
  id: string;
  type: QuestionType;
  text: string;
  config: BaseConfig;
  answer?: Answer;
  answeredAt?: number;
}

export interface Branch {
  id: string;
  scope: string;
  status: BranchStatus;
  questions: BranchQuestion[];
  finding: string | null;
}

export interface BrainstormState {
  session_id: string;
  browser_session_id: string | null;
  request: string;
  created_at: number;
  updated_at: number;
  branches: Record<string, Branch>;
  branch_order: string[];
}

export interface CreateBranchInput {
  id: string;
  scope: string;
}

export interface BranchProbeResult {
  done: boolean;
  reason: string;
  finding?: string;
  question?: {
    type: QuestionType;
    config: BaseConfig;
  };
}
