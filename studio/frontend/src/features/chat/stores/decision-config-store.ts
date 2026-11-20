// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The one question configured for the loaded decision-only GGUF.
 *
 * A decision model answers one typed question per message, so the composer keeps
 * a single editable question. The answer comes back under the fixed key
 * "answer": the user never has to name it.
 */

// eslint-disable-next-line no-restricted-imports -- Leaf module: request types, no React.
import {
  type DecisionDraft,
  type DecisionType,
  type WireQuestion,
  buildQuestion,
} from "@/features/settings/lib/decision-request";
import { create } from "zustand";

export type DecisionQuestion = {
  id: string;
  type: DecisionType;
  draft: DecisionDraft;
};

export type DecisionQuestionPatch = {
  type?: DecisionType;
  draft?: Partial<DecisionDraft>;
};

/** The answer key llama-server gets; nothing in the UI exposes it. */
export const DECISION_ANSWER_KEY = "answer";

let questionSeq = 0;
const nextQuestionId = () => `decision-question-${++questionSeq}`;

/** A usable starting point per type, so the first send needs no setup. */
export function defaultDraft(type: DecisionType): DecisionDraft {
  const base = {
    name: DECISION_ANSWER_KEY,
    state: "",
    yesWhen: "",
    noWhen: "",
  };
  if (type === "choice") {
    return {
      ...base,
      instructions: "Which option best describes this?",
      options: [
        { name: "", description: "" },
        { name: "", description: "" },
      ],
      levels: [],
    };
  }
  if (type === "score") {
    return {
      ...base,
      instructions: "How would you rate this?",
      options: [],
      levels: ["Low", "Medium", "High"],
    };
  }
  return {
    ...base,
    instructions: "Is this statement true?",
    options: [],
    levels: [],
  };
}

export function newDecisionQuestion(type: DecisionType): DecisionQuestion {
  return { id: nextQuestionId(), type, draft: defaultDraft(type) };
}

/** A question is sendable only with instructions: llama-server rejects it otherwise. */
export function decisionQuestionReady(question: DecisionQuestion): boolean {
  if (!question.draft.instructions.trim()) return false;
  if (question.type === "choice") {
    return (
      question.draft.options.filter((option) => option.name.trim()).length >= 2
    );
  }
  if (question.type === "score") {
    return question.draft.levels.some((level) => level.trim());
  }
  return true;
}

/** The question in /v1/systemone's wire shape. */
export function decisionQuestionToWire(question: DecisionQuestion): {
  name: string;
  body: WireQuestion;
} {
  if (question.type === "choice") {
    // The chat composer numbers its options, so the wire keys are "1", "2", ...
    // and the option text travels as the criterion's value.
    return {
      name: DECISION_ANSWER_KEY,
      body: {
        type: "choice",
        instructions: question.draft.instructions.trim(),
        criteria: Object.fromEntries(
          question.draft.options
            .map(
              (option, index) =>
                [String(index + 1), option.name.trim()] as const,
            )
            .filter(([, text]) => text.length > 0),
        ),
      },
    };
  }
  return {
    name: DECISION_ANSWER_KEY,
    body: buildQuestion(question.type, question.draft),
  };
}

type DecisionConfigState = {
  question: DecisionQuestion;
  /** Which inline composer panel is open; only one at a time. */
  openPanel: DecisionPanel;
  update: (patch: DecisionQuestionPatch) => void;
  setPanel: (panel: DecisionPanel) => void;
  togglePanel: (panel: Exclude<DecisionPanel, "none">) => void;
  reset: () => void;
};

export type DecisionPanel = "none" | "instructions" | "criteria";

export const useDecisionConfigStore = create<DecisionConfigState>((set) => ({
  question: newDecisionQuestion("noul"),
  openPanel: "none",

  setPanel: (panel) => set({ openPanel: panel }),

  togglePanel: (panel) =>
    set((state) => ({ openPanel: state.openPanel === panel ? "none" : panel })),

  update: (patch) =>
    set((state) => {
      const question = state.question;
      if (!patch.type || patch.type === question.type) {
        return {
          question: {
            ...question,
            draft: patch.draft
              ? { ...question.draft, ...patch.draft }
              : question.draft,
          },
        };
      }
      // A new type starts from its own defaults, but a user's own instructions
      // survive the switch; only an untouched default is replaced.
      const base = defaultDraft(patch.type);
      const previousDefault = defaultDraft(question.type).instructions;
      const keepInstructions =
        question.draft.instructions.trim() !== "" &&
        question.draft.instructions.trim() !== previousDefault;
      return {
        question: {
          ...question,
          type: patch.type,
          draft: {
            ...base,
            ...(keepInstructions
              ? { instructions: question.draft.instructions }
              : {}),
            ...patch.draft,
          },
        },
      };
    }),

  reset: () => set({ question: newDecisionQuestion("noul"), openPanel: "none" }),
}));
