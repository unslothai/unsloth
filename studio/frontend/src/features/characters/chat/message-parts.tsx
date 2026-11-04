// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import { createContext, useContext, type ReactNode } from "react";
import { useMessagePartTextSafe } from "@/components/assistant-ui/stale-safe-scopes";
import { Portrait } from "../components/portrait";
import { RoleplayText } from "../components/roleplay-text";
import { showOnStage } from "../stage-focus";
import { roleplayMetaOf } from "../meta";
import { useCharactersStore } from "../store";
import { useThreadScene } from "./use-thread-scene";

/** The speaking character's name, so a reply's leading "Name:" can be hidden. */
export const RoleplaySpeakerNameContext = createContext<string | null>(null);

export function useRoleplaySpeakerName(): string | null {
  return useContext(RoleplaySpeakerNameContext);
}

/** Who said this reply: its stamped speaker, else the chat's only character. */
function useReplySpeaker() {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const thread = useThreadScene(threadId);
  // Strings, not the record: a selector must return a stable value or the store re-renders forever.
  const group = Boolean(thread?.group);
  const speakerId = useAuiState(({ message, thread: state }) => {
    const meta = roleplayMetaOf(message);
    if (meta?.kind === "reply") return meta.speakerId;
    if (!group) return null;
    // A reply saved without its speaker (an older run) answers the turn marker before it.
    const index = state.messages.findIndex((m) => m.id === message.id);
    const previous = index > 0 ? roleplayMetaOf(state.messages[index - 1]) : null;
    return previous?.kind === "turn" ? previous.speakerId : null;
  });
  const speakerName = useAuiState(({ message }) => {
    const meta = roleplayMetaOf(message);
    return meta?.kind === "reply" ? meta.speakerName : "";
  });
  const characters = useCharactersStore((s) => s.characters);
  if (!thread) return null;
  if (speakerId) {
    const character = characters[speakerId] ?? null;
    return { id: speakerId, name: character?.name ?? speakerName, character, group: thread.group };
  }
  if (thread.cast.length === 1) {
    const only = thread.cast[0];
    return { id: only.id, name: only.name, character: only, group: false };
  }
  return null;
}

/** Wraps a reply in a character chat: portrait and name above the text, prefix stripping below. */
export function RoleplayReplyFrame({ children }: { children: ReactNode }) {
  const t = useT();
  const speaker = useReplySpeaker();
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const aui = useAui();
  if (!speaker) return <>{children}</>;
  const character = speaker.character;
  return (
    <RoleplaySpeakerNameContext.Provider value={speaker.name}>
      <div className="roleplay-reply relative pl-[calc(2.75rem*var(--ui-space-scale,1))] [&_.aui-assistant-message-content_em]:text-[color-mix(in_oklab,currentColor_62%,transparent)]" data-speaker={speaker.id}>
        <Portrait
          id={character?.id ?? speaker.id}
          name={speaker.name}
          hasImage={Boolean(character?.hasAvatar)}
          rev={character?.avatarRev ?? 0}
          label={speaker.name}
          className="absolute left-0 top-0.5 size-[calc(2.125rem*var(--ui-space-scale,1))] rounded-[11px] text-[0.95rem] shadow-[0_1px_2px_rgb(0_0_0/0.12)]"
          onClick={
            character && threadId
              ? () => showOnStage(threadId, character.id, aui.thread().getState().messages.length)
              : undefined
          }
        />
        <div className="mb-1 flex h-[calc(1.25rem*var(--ui-space-scale,1))] items-center gap-2">
          <span
            className={cn(
              "truncate text-ui-14 font-semibold text-foreground",
              !character && "text-muted-foreground",
            )}
          >
            {character ? speaker.name : `${speaker.name} · ${t("characters.chat.deleted")}`}
          </span>
        </div>
        {children}
      </div>
    </RoleplaySpeakerNameContext.Provider>
  );
}

/** The persona's name over the user's own messages in a character chat. */
export function RoleplayPersonaTag() {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const thread = useThreadScene(threadId);
  if (!thread) return null;
  return (
    <span className="-mb-1 mr-1 text-ui-12 font-medium text-muted-foreground" data-testid="persona-tag">
      {thread.persona.name}
    </span>
  );
}

/** Hidden bookkeeping turns (the opening and group turn markers) render nothing. */
export function useIsHiddenRoleplayTurn(): boolean {
  return useAuiState(({ message }) => {
    if (message.role !== "user") return false;
    const kind = roleplayMetaOf(message)?.kind;
    return kind === "opening" || kind === "turn";
  });
}

function RoleplayUserText() {
  const { text } = useMessagePartTextSafe();
  return <RoleplayText text={text} className="whitespace-pre-wrap" />;
}

const ROLEPLAY_USER_PARTS = { Text: RoleplayUserText };

/** In a character chat, *actions* in the user's own messages read as italics too. */
export function useRoleplayUserParts(): typeof ROLEPLAY_USER_PARTS | undefined {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  return useThreadScene(threadId) ? ROLEPLAY_USER_PARTS : undefined;
}
