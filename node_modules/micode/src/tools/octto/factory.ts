// src/tools/octto/factory.ts

import { tool } from "@opencode-ai/plugin/tool";

import type { BaseConfig, QuestionType, SessionStore } from "../../octto/session";
import type { OcttoTool, OcttoTools } from "./types";

type ArgsSchema = Parameters<typeof tool>[0]["args"];

interface QuestionToolConfig<T> {
  type: QuestionType;
  description: string;
  args: ArgsSchema;
  validate?: (args: T) => string | null;
  toConfig: (args: T) => BaseConfig;
}

export function createQuestionToolFactory(sessions: SessionStore) {
  return function createQuestionTool<T extends { session_id: string }>(config: QuestionToolConfig<T>): OcttoTool {
    return tool({
      description: `${config.description}
Returns immediately with question_id. Use get_answer to retrieve response.`,
      args: {
        session_id: tool.schema.string().describe("Session ID from start_session"),
        ...config.args,
      },
      execute: async (args) => {
        const validationError = config.validate?.(args as unknown as T);
        if (validationError) return `Failed: ${validationError}`;

        try {
          const questionConfig = config.toConfig(args as unknown as T);
          const result = sessions.pushQuestion(args.session_id, config.type, questionConfig);
          return `Question pushed: ${result.question_id}\nUse get_answer("${result.question_id}") to retrieve response.`;
        } catch (error) {
          return `Failed: ${error instanceof Error ? error.message : String(error)}`;
        }
      },
    });
  };
}

export function createPushQuestionTool(sessions: SessionStore): OcttoTools {
  const push_question = tool({
    description: `Push a question to the session queue. This is the generic tool for adding any question type.
The question will appear in the browser for the user to answer.`,
    args: {
      session_id: tool.schema.string().describe("Session ID from start_session"),
      type: tool.schema
        .enum([
          "pick_one",
          "pick_many",
          "confirm",
          "ask_text",
          "ask_image",
          "ask_file",
          "ask_code",
          "show_diff",
          "show_plan",
          "show_options",
          "review_section",
          "thumbs",
          "slider",
          "rank",
          "rate",
          "emoji_react",
        ])
        .describe("Question type"),
      config: tool.schema
        .looseObject({
          question: tool.schema.string().optional(),
          context: tool.schema.string().optional(),
        })
        .describe("Question configuration (varies by type)"),
    },
    execute: async (args) => {
      try {
        const result = sessions.pushQuestion(args.session_id, args.type, args.config);
        return `Question pushed: ${result.question_id}
Type: ${args.type}
Use get_next_answer(session_id, block=true) to wait for the user's response.`;
      } catch (error) {
        return `Failed to push question: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  });

  return { push_question };
}
