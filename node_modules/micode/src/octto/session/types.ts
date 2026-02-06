// src/octto/session/types.ts
// Session and Question types for the octto module
import type { ServerWebSocket } from "bun";

import type {
  AskCodeConfig,
  AskFileConfig,
  AskImageConfig,
  AskTextConfig,
  ConfirmConfig,
  EmojiReactConfig,
  PickManyConfig,
  PickOneConfig,
  RankConfig,
  RateConfig,
  ReviewSectionConfig,
  ShowDiffConfig,
  ShowOptionsConfig,
  ShowPlanConfig,
  SliderConfig,
  ThumbsConfig,
} from "../types";

export const STATUSES = {
  PENDING: "pending",
  ANSWERED: "answered",
  CANCELLED: "cancelled",
  TIMEOUT: "timeout",
  NONE_PENDING: "none_pending",
} as const;

export type QuestionStatus = (typeof STATUSES)[Exclude<keyof typeof STATUSES, "NONE_PENDING">];

export interface Question {
  id: string;
  sessionId: string;
  type: QuestionType;
  config: BaseConfig;
  status: QuestionStatus;
  createdAt: Date;
  answeredAt?: Date;
  response?: Answer;
  /** True if this answer was already returned via get_next_answer */
  retrieved?: boolean;
}

export const QUESTIONS = {
  PICK_ONE: "pick_one",
  PICK_MANY: "pick_many",
  CONFIRM: "confirm",
  RANK: "rank",
  RATE: "rate",
  ASK_TEXT: "ask_text",
  ASK_IMAGE: "ask_image",
  ASK_FILE: "ask_file",
  ASK_CODE: "ask_code",
  SHOW_DIFF: "show_diff",
  SHOW_PLAN: "show_plan",
  SHOW_OPTIONS: "show_options",
  REVIEW_SECTION: "review_section",
  THUMBS: "thumbs",
  EMOJI_REACT: "emoji_react",
  SLIDER: "slider",
} as const;

export type QuestionType = (typeof QUESTIONS)[keyof typeof QUESTIONS];
export const QUESTION_TYPES = Object.values(QUESTIONS);

// --- Answer Types ---

export interface PickOneAnswer {
  selected: string;
}

export interface PickManyAnswer {
  selected: string[];
}

export interface ConfirmAnswer {
  choice: "yes" | "no" | "cancel";
}

export interface ThumbsAnswer {
  choice: "up" | "down";
}

export interface EmojiReactAnswer {
  emoji: string;
}

export interface AskTextAnswer {
  text: string;
}

export interface SliderAnswer {
  value: number;
}

export interface RankAnswer {
  ranking: Array<{ id: string; rank: number }>;
}

export interface RateAnswer {
  ratings: Record<string, number>;
}

export interface AskCodeAnswer {
  code: string;
}

export interface AskImageAnswer {
  images: Array<{ name: string; data: string; type: string }>;
}

export interface AskFileAnswer {
  files: Array<{ name: string; data: string; type: string }>;
}

export interface ReviewAnswer {
  decision: string;
  feedback?: string;
}

export interface ShowOptionsAnswer {
  selected: string;
  feedback?: string;
}

export type Answer =
  | PickOneAnswer
  | PickManyAnswer
  | ConfirmAnswer
  | ThumbsAnswer
  | EmojiReactAnswer
  | AskTextAnswer
  | SliderAnswer
  | RankAnswer
  | RateAnswer
  | AskCodeAnswer
  | AskImageAnswer
  | AskFileAnswer
  | ReviewAnswer
  | ShowOptionsAnswer;

export interface QuestionAnswers {
  [QUESTIONS.PICK_ONE]: PickOneAnswer;
  [QUESTIONS.PICK_MANY]: PickManyAnswer;
  [QUESTIONS.CONFIRM]: ConfirmAnswer;
  [QUESTIONS.THUMBS]: ThumbsAnswer;
  [QUESTIONS.EMOJI_REACT]: EmojiReactAnswer;
  [QUESTIONS.ASK_TEXT]: AskTextAnswer;
  [QUESTIONS.SLIDER]: SliderAnswer;
  [QUESTIONS.RANK]: RankAnswer;
  [QUESTIONS.RATE]: RateAnswer;
  [QUESTIONS.ASK_CODE]: AskCodeAnswer;
  [QUESTIONS.ASK_IMAGE]: AskImageAnswer;
  [QUESTIONS.ASK_FILE]: AskFileAnswer;
  [QUESTIONS.SHOW_DIFF]: ReviewAnswer;
  [QUESTIONS.SHOW_PLAN]: ReviewAnswer;
  [QUESTIONS.REVIEW_SECTION]: ReviewAnswer;
  [QUESTIONS.SHOW_OPTIONS]: ShowOptionsAnswer;
}

export type QuestionConfig =
  | PickOneConfig
  | PickManyConfig
  | ConfirmConfig
  | RankConfig
  | RateConfig
  | AskTextConfig
  | AskImageConfig
  | AskFileConfig
  | AskCodeConfig
  | ShowDiffConfig
  | ShowPlanConfig
  | ShowOptionsConfig
  | ReviewSectionConfig
  | ThumbsConfig
  | EmojiReactConfig
  | SliderConfig;

/** Config type for transit - accepts both strict QuestionConfig and loose objects */
export type BaseConfig =
  | QuestionConfig
  | {
      question?: string;
      context?: string;
      [key: string]: unknown;
    };

export interface Session {
  id: string;
  title?: string;
  port: number;
  url: string;
  createdAt: Date;
  questions: Map<string, Question>;
  wsConnected: boolean;
  server?: ReturnType<typeof Bun.serve>;
  wsClient?: ServerWebSocket<unknown>;
}

export interface InitialQuestion {
  type: QuestionType;
  config: BaseConfig;
}

export interface StartSessionInput {
  title?: string;
  /** Initial questions to display immediately when browser opens */
  questions?: InitialQuestion[];
}

export interface StartSessionOutput {
  session_id: string;
  url: string;
  /** IDs of initial questions if any were provided */
  question_ids?: string[];
}

export interface EndSessionOutput {
  ok: boolean;
}

export interface PushQuestionOutput {
  question_id: string;
}

export interface GetAnswerInput {
  question_id: string;
  block?: boolean;
  timeout?: number;
}

export interface GetAnswerOutput {
  completed: boolean;
  status: QuestionStatus;
  response?: Answer;
  reason?: "timeout" | "cancelled" | "pending";
}

export interface GetNextAnswerInput {
  session_id: string;
  block?: boolean;
  timeout?: number;
}

export type AnswerStatus = (typeof STATUSES)[keyof typeof STATUSES];

export interface GetNextAnswerOutput {
  completed: boolean;
  question_id?: string;
  question_type?: QuestionType;
  status: AnswerStatus;
  response?: Answer;
  reason?: typeof STATUSES.TIMEOUT | typeof STATUSES.NONE_PENDING;
}

export interface ListQuestionsOutput {
  questions: Array<{
    id: string;
    type: QuestionType;
    status: QuestionStatus;
    createdAt: string;
    answeredAt?: string;
  }>;
}

// WebSocket message types
export const WS_MESSAGES = {
  QUESTION: "question",
  CANCEL: "cancel",
  END: "end",
  RESPONSE: "response",
  CONNECTED: "connected",
} as const;

export interface WsQuestionMessage {
  type: "question";
  id: string;
  questionType: QuestionType;
  config: BaseConfig;
}

export interface WsCancelMessage {
  type: "cancel";
  id: string;
}

export interface WsEndMessage {
  type: "end";
}

export interface WsResponseMessage {
  type: "response";
  id: string;
  answer: Answer;
}

export interface WsConnectedMessage {
  type: "connected";
}

export type WsServerMessage = WsQuestionMessage | WsCancelMessage | WsEndMessage;
export type WsClientMessage = WsResponseMessage | WsConnectedMessage;
