// src/octto/constants.ts
// Re-exports from centralized config for backward compatibility
// Single source of truth is in src/utils/config.ts

import { config } from "../utils/config";

/** Default timeout for waiting for user answers (5 minutes) */
export const DEFAULT_ANSWER_TIMEOUT_MS = config.octto.answerTimeoutMs;

/** Default maximum number of follow-up questions per branch */
export const DEFAULT_MAX_QUESTIONS = config.octto.maxQuestions;

/** Default timeout for brainstorm review (10 minutes) */
export const DEFAULT_REVIEW_TIMEOUT_MS = config.octto.reviewTimeoutMs;

/** Maximum number of brainstorm iterations */
export const MAX_ITERATIONS = config.octto.maxIterations;

/** Directory for persisting brainstorm state files */
export const STATE_DIR = config.octto.stateDir;
