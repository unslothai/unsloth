// src/octto/state/store.ts

import { STATE_DIR } from "../constants";
import type { Answer } from "../session";
import { createStatePersistence } from "./persistence";
import {
  BRANCH_STATUSES,
  type BrainstormState,
  type Branch,
  type BranchQuestion,
  type CreateBranchInput,
} from "./types";

export interface StateStore {
  createSession: (sessionId: string, request: string, branches: CreateBranchInput[]) => Promise<BrainstormState>;
  getSession: (sessionId: string) => Promise<BrainstormState | null>;
  setBrowserSessionId: (sessionId: string, browserSessionId: string) => Promise<void>;
  addQuestionToBranch: (sessionId: string, branchId: string, question: BranchQuestion) => Promise<BranchQuestion>;
  recordAnswer: (sessionId: string, questionId: string, answer: Answer) => Promise<void>;
  completeBranch: (sessionId: string, branchId: string, finding: string) => Promise<void>;
  getNextExploringBranch: (sessionId: string) => Promise<Branch | null>;
  isSessionComplete: (sessionId: string) => Promise<boolean>;
  deleteSession: (sessionId: string) => Promise<void>;
}

export function createStateStore(baseDir = STATE_DIR): StateStore {
  const persistence = createStatePersistence(baseDir);

  // Operation queue per session to prevent concurrent read-modify-write races
  const operationQueues = new Map<string, Promise<void>>();

  // Serialize operations for a given session
  function withSessionLock<T>(sessionId: string, operation: () => Promise<T>): Promise<T> {
    const currentQueue = operationQueues.get(sessionId) ?? Promise.resolve();
    const newOperation = currentQueue.then(operation, operation); // Run even if previous failed
    operationQueues.set(
      sessionId,
      newOperation.then(
        () => {},
        () => {},
      ),
    ); // Ignore result for queue
    return newOperation;
  }

  return {
    async createSession(
      sessionId: string,
      request: string,
      branchInputs: CreateBranchInput[],
    ): Promise<BrainstormState> {
      const branches: Record<string, Branch> = {};
      const order: string[] = [];

      for (const input of branchInputs) {
        branches[input.id] = {
          id: input.id,
          scope: input.scope,
          status: BRANCH_STATUSES.EXPLORING,
          questions: [],
          finding: null,
        };
        order.push(input.id);
      }

      const state: BrainstormState = {
        session_id: sessionId,
        browser_session_id: null,
        request,
        created_at: Date.now(),
        updated_at: Date.now(),
        branches,
        branch_order: order,
      };

      await persistence.save(state);
      return state;
    },

    async getSession(sessionId: string): Promise<BrainstormState | null> {
      return persistence.load(sessionId);
    },

    async setBrowserSessionId(sessionId: string, browserSessionId: string): Promise<void> {
      return withSessionLock(sessionId, async () => {
        const state = await persistence.load(sessionId);
        if (!state) throw new Error(`Session not found: ${sessionId}`);
        state.browser_session_id = browserSessionId;
        await persistence.save(state);
      });
    },

    async addQuestionToBranch(sessionId: string, branchId: string, question: BranchQuestion): Promise<BranchQuestion> {
      return withSessionLock(sessionId, async () => {
        const state = await persistence.load(sessionId);
        if (!state) throw new Error(`Session not found: ${sessionId}`);
        if (!state.branches[branchId]) throw new Error(`Branch not found: ${branchId}`);

        state.branches[branchId].questions.push(question);
        await persistence.save(state);
        return question;
      });
    },

    async recordAnswer(sessionId: string, questionId: string, answer: Answer): Promise<void> {
      return withSessionLock(sessionId, async () => {
        const state = await persistence.load(sessionId);
        if (!state) throw new Error(`Session not found: ${sessionId}`);

        for (const branch of Object.values(state.branches)) {
          const question = branch.questions.find((q) => q.id === questionId);
          if (question) {
            question.answer = answer;
            question.answeredAt = Date.now();
            await persistence.save(state);
            return;
          }
        }
        throw new Error(`Question not found: ${questionId}`);
      });
    },

    async completeBranch(sessionId: string, branchId: string, finding: string): Promise<void> {
      return withSessionLock(sessionId, async () => {
        const state = await persistence.load(sessionId);
        if (!state) throw new Error(`Session not found: ${sessionId}`);
        if (!state.branches[branchId]) throw new Error(`Branch not found: ${branchId}`);

        state.branches[branchId].status = BRANCH_STATUSES.DONE;
        state.branches[branchId].finding = finding;
        await persistence.save(state);
      });
    },

    async getNextExploringBranch(sessionId: string): Promise<Branch | null> {
      const state = await persistence.load(sessionId);
      if (!state) return null;

      for (const branchId of state.branch_order) {
        const branch = state.branches[branchId];
        if (branch.status === BRANCH_STATUSES.EXPLORING) {
          return branch;
        }
      }
      return null;
    },

    async isSessionComplete(sessionId: string): Promise<boolean> {
      const state = await persistence.load(sessionId);
      if (!state) return false;

      return Object.values(state.branches).every((b) => b.status === BRANCH_STATUSES.DONE);
    },

    async deleteSession(sessionId: string): Promise<void> {
      await withSessionLock(sessionId, async () => {
        await persistence.delete(sessionId);
      });
    },
  };
}
