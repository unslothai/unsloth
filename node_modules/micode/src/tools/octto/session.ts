// src/tools/octto/session.ts
import { tool } from "@opencode-ai/plugin/tool";

import type { SessionStore } from "../../octto/session";
import type { OcttoSessionTracker, OcttoTools } from "./types";

export function createSessionTools(sessions: SessionStore, tracker?: OcttoSessionTracker): OcttoTools {
  const start_session = tool({
    description: `Start an interactive octto session with initial questions.
Opens a browser window with questions already displayed - no waiting.
REQUIRED: You MUST provide at least 1 question. Will fail without questions.`,
    args: {
      title: tool.schema.string().optional().describe("Session title (shown in browser)"),
      questions: tool.schema
        .array(
          tool.schema.object({
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
              .describe("Question config (varies by type)"),
          }),
        )
        .describe("REQUIRED: Initial questions to display when browser opens. Must have at least 1."),
    },
    execute: async (args, context) => {
      // ENFORCE: questions are required
      if (!args.questions || args.questions.length === 0) {
        return `## ERROR: questions parameter is REQUIRED

start_session MUST include questions. Browser should open with questions ready.

Example:
\`\`\`
start_session(
  title="Design Session",
  questions=[
    {type: "pick_one", config: {question: "What language?", options: [{id: "go", label: "Go"}]}},
    {type: "ask_text", config: {question: "Any constraints?"}}
  ]
)
\`\`\`

Please call start_session again WITH your prepared questions.`;
      }

      try {
        const result = await sessions.startSession({ title: args.title, questions: args.questions });
        tracker?.onCreated?.(context.sessionID, result.session_id);

        let output = `## Session Started

| Field | Value |
|-------|-------|
| Session ID | ${result.session_id} |
| URL | ${result.url} |
`;

        if (result.question_ids && result.question_ids.length > 0) {
          output += `| Questions | ${result.question_ids.length} loaded |\n\n`;
          output += `**Question IDs:** ${result.question_ids.join(", ")}\n\n`;
          output += `Browser opened with ${result.question_ids.length} questions ready.\n`;
          output += `Use get_next_answer(session_id, block=true) to get answers as user responds.`;
        } else {
          output += `\nBrowser opened. Use question tools to push questions.`;
        }

        return output;
      } catch (error) {
        return `Failed to start session: ${error instanceof Error ? error.message : String(error)}`;
      }
    },
  });

  const end_session = tool({
    description: `End an interactive octto session.
Closes the browser window and cleans up resources.`,
    args: {
      session_id: tool.schema.string().describe("Session ID to end"),
    },
    execute: async (args, context) => {
      const result = await sessions.endSession(args.session_id);
      if (result.ok) {
        tracker?.onEnded?.(context.sessionID, args.session_id);
        return `Session ${args.session_id} ended successfully.`;
      }
      return `Failed to end session ${args.session_id}. It may not exist.`;
    },
  });

  return { start_session, end_session };
}
