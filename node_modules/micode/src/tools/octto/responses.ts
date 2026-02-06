// src/tools/octto/responses.ts
import { tool } from "@opencode-ai/plugin/tool";

import { type SessionStore, STATUSES } from "../../octto/session";

import type { OcttoTools } from "./types";

export function createResponseTools(sessions: SessionStore): OcttoTools {
  const get_answer = tool({
    description: `Get the answer to a SPECIFIC question.
By default returns immediately with current status.
Set block=true to wait for user response (with optional timeout).
NOTE: Prefer get_next_answer for better flow - it returns whichever question user answers first.`,
    args: {
      question_id: tool.schema.string().describe("Question ID from a question tool"),
      block: tool.schema.boolean().optional().describe("Wait for response (default: false)"),
      timeout: tool.schema
        .number()
        .optional()
        .describe("Max milliseconds to wait if blocking (default: 300000 = 5 min)"),
    },
    execute: async (args) => {
      const result = await sessions.getAnswer({
        question_id: args.question_id,
        block: args.block,
        timeout: args.timeout,
      });

      if (result.completed) {
        return `## Answer Received

**Status:** ${result.status}

**Response:**
\`\`\`json
${JSON.stringify(result.response, null, 2)}
\`\`\``;
      }

      return `## Waiting for Answer

**Status:** ${result.status}
**Reason:** ${result.reason}

${result.status === STATUSES.PENDING ? "User has not answered yet. Call again with block=true to wait." : ""}`;
    },
  });

  const get_next_answer = tool({
    description: `Wait for ANY question to be answered. Returns whichever question the user answers first.
This is the PREFERRED way to get answers - lets user answer in any order.
Push multiple questions, then call this repeatedly to get answers as they come.`,
    args: {
      session_id: tool.schema.string().describe("Session ID from start_session"),
      block: tool.schema.boolean().optional().describe("Wait for response (default: false)"),
      timeout: tool.schema
        .number()
        .optional()
        .describe("Max milliseconds to wait if blocking (default: 300000 = 5 min)"),
    },
    execute: async (args) => {
      const result = await sessions.getNextAnswer({
        session_id: args.session_id,
        block: args.block,
        timeout: args.timeout,
      });

      if (result.completed) {
        return `## Answer Received

**Question ID:** ${result.question_id}
**Question Type:** ${result.question_type}
**Status:** ${result.status}

**Response:**
\`\`\`json
${JSON.stringify(result.response, null, 2)}
\`\`\``;
      }

      if (result.status === STATUSES.NONE_PENDING) {
        return `## No Pending Questions

All questions have been answered or there are no questions in the queue.
Push more questions or end the session.`;
      }

      return `## Waiting for Answer

**Status:** ${result.status}
${result.reason === STATUSES.TIMEOUT ? "Timed out waiting for response." : "No answer yet."}`;
    },
  });

  const list_questions = tool({
    description: `List all questions and their status for a session.`,
    args: {
      session_id: tool.schema.string().optional().describe("Session ID (omit for all sessions)"),
    },
    execute: async (args) => {
      const result = sessions.listQuestions(args.session_id);

      if (result.questions.length === 0) {
        return "No questions found.";
      }

      let output = "## Questions\n\n";
      output += "| ID | Type | Status | Created | Answered |\n";
      output += "|----|------|--------|---------|----------|\n";

      for (const q of result.questions) {
        output += `| ${q.id} | ${q.type} | ${q.status} | ${q.createdAt} | ${q.answeredAt || "-"} |\n`;
      }

      return output;
    },
  });

  const cancel_question = tool({
    description: `Cancel a pending question.
The question will be removed from the user's queue.`,
    args: {
      question_id: tool.schema.string().describe("Question ID to cancel"),
    },
    execute: async (args) => {
      const result = sessions.cancelQuestion(args.question_id);
      if (result.ok) {
        return `Question ${args.question_id} cancelled.`;
      }
      return `Could not cancel question ${args.question_id}. It may already be answered or not exist.`;
    },
  });

  return { get_answer, get_next_answer, list_questions, cancel_question };
}
