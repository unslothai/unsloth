// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The UI the chat renders. Lazy pieces (sidebar mark, chat panel, start picker, pages) stay out, and so do modules
// @/features/chat loads itself (chat-runtime, export-transcript, use-thread-scene): these message parts import it back.
export { CharacterStage } from "./chat/character-stage";
export { CastChip, ChatViewMenu } from "./chat/chat-chrome";
export { requestCastReveal } from "./chat/cast-reveal";
export { useInCharacterChat } from "./chat/character-chat-context";
export {
  CharacterReplyFrame,
  PersonaMessageFrame,
  useCharacterReplyText,
  useCharacterUserParts,
  useHiddenRoleplayTurn,
  useInCharacterReply,
} from "./chat/message-parts";
export { CharacterSpeakerBar, useCharacterComposerPlaceholder } from "./chat/speaker-bar";
export { groupRoundPending } from "./chat/group-round-controller";
export type { CharacterReplyMeta } from "./types";
