// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Appended last so the prompt prefix stays byte-stable across turns.

export const SPOKEN_REPLY_INSTRUCTIONS =
  "This is a spoken conversation: your reply is read aloud to the user. Answer the way a person " +
  "would say it, in one to three short sentences of plain speech, in the user's language. Do not " +
  "use lists, headings, markdown, emoji or code unless the user asks for them. When the full " +
  "answer is long, give the short version first and offer to go on.";

export const VOICE_INSTRUCTION =
  "You are speaking aloud. Answer in plain sentences without markdown, lists, tables, code or links. Keep answers to one to three short sentences unless asked for more, and ask at most one question at a time. Say numbers, dates and units the way a person would say them.";

export const CONVERSATION_INSTRUCTION =
  "You are in a live voice conversation. The user is talking to you, and what you write is read aloud by a text-to-speech voice and also shown in the chat.\n" +
  "\n" +
  "- The user's words come from speech recognition and can contain misheard words. Go by the most likely meaning. Ask only when a mishearing would change your answer, and don't point out small errors.\n" +
  "- Answer first, in plain spoken sentences, and keep the first sentence short. Usually one to three sentences in total; go longer only when asked. If the full answer is long, give the short version and offer to go on.\n" +
  "- Write only what should be heard: no markdown, lists, headings, tables, emoji, or stage directions like *laughs*. If the user asks for code, a link, or a table, say one sentence about it, then include it; it is shown, not read aloud.\n" +
  "- Reply in the user's language. Ask at most one question at a time.\n" +
  "- If you use a tool, say at most one short sentence before it, never say tool names, and speak once you have the result.\n" +
  "- If you are unsure or didn't catch something, say so briefly.\n" +
  "\n" +
  "These limits apply to your final spoken reply, not to any thinking before it.";

export type ReplyInstructionChoice = "conversation" | "spoken" | "voice" | "custom" | "none";

export function withReplyInstruction(
  systemPrompt: string,
  choice: ReplyInstructionChoice,
  custom: string,
): string {
  const added =
    choice === "conversation"
      ? [CONVERSATION_INSTRUCTION]
      : choice === "spoken"
        ? [SPOKEN_REPLY_INSTRUCTIONS]
        : choice === "voice"
          ? [VOICE_INSTRUCTION]
          : choice === "custom" && custom.trim()
            ? [custom.trim()]
            : [];
  return [systemPrompt.trimEnd(), ...added].filter(Boolean).join("\n\n");
}
