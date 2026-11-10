// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useLocale, useT } from "@/i18n";
import { withTimeFormat } from "@/lib/time-format";
import { cn } from "@/lib/utils";
import { useUserProfileStore } from "@/features/profile";
import { useAui, useAuiState } from "@assistant-ui/react";
import { createContext, useContext, type ReactNode } from "react";
import { MessageResponseModelBadge } from "@/components/assistant-ui/message-response-details-sheet";
import { useMessagePartTextSafe } from "@/components/assistant-ui/stale-safe-scopes";
import { Portrait } from "../components/portrait";
import { RoleplayText } from "../components/roleplay-text";
import { toggleOnStage } from "../stage-focus";
import { roleplayMetaOf } from "../meta";
import { messageTime } from "../time";
import { useCharactersStore } from "../store";
import { useThreadScene } from "./use-thread-scene";

/** The speaking character's name, so a reply's leading "Name:" can be hidden. */
export const RoleplaySpeakerNameContext = createContext<string | null>(null);

export function useRoleplaySpeakerName(): string | null {
  return useContext(RoleplaySpeakerNameContext);
}

/** When this message was written, with the full date on hover. */
function MessageTime({ className }: { className?: string }) {
  const locale = useLocale();
  // A number, not the Date: a selector must return a stable value.
  const at = useAuiState(({ message }) => (message.createdAt instanceof Date ? message.createdAt.getTime() : 0));
  if (!at) return null;
  const full = new Date(at).toLocaleString(
    locale,
    withTimeFormat({ weekday: "long", year: "numeric", month: "long", day: "numeric", hour: "numeric", minute: "2-digit" }, locale),
  );
  return (
    <time
      dateTime={new Date(at).toISOString()}
      title={full}
      className={cn("shrink-0 text-ui-12 font-normal tabular-nums text-muted-foreground", className)}
      data-testid="roleplay-message-time"
    >
      {messageTime(locale, at)}
    </time>
  );
}

/** The chat's cast ids, read when needed rather than subscribed to by every reply. */
function castIdsOf(threadId: string): string[] {
  const scene = useCharactersStore.getState().scenes[threadId];
  return scene ? scene.cast.map((member) => member.characterId) : [];
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
  // Only this reply's speaker, so an edit to another character does not re-render every reply.
  const speaker = useCharactersStore((s) => (speakerId ? s.characters[speakerId] : undefined));
  if (!thread) return null;
  if (speakerId) {
    const character = speaker ?? null;
    return { id: speakerId, name: character?.name ?? speakerName, character, group: thread.group };
  }
  if (thread.cast.length === 1) {
    const only = thread.cast[0];
    return { id: only.id, name: only.name, character: only, group: false };
  }
  return null;
}

/** Who wrote a message: picture on the left, then name and time, then the message. Replies and the user's own messages share it. */
function SpeakerFrame({
  className,
  portrait,
  name,
  speakerId,
  children,
}: {
  className: string;
  portrait: ReactNode;
  name: ReactNode;
  speakerId?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "relative min-h-[calc(4rem*var(--ui-space-scale,1))] pl-[calc(3.75rem*var(--ui-space-scale,1))] max-sm:pl-[calc(3.25rem*var(--ui-space-scale,1))]",
        "[&_.aui-assistant-message-content_em]:text-[color-mix(in_oklab,currentColor_62%,transparent)] [&_.aui-user-message-content_em]:text-[color-mix(in_oklab,currentColor_62%,transparent)]",
        className,
      )}
      data-speaker={speakerId}
    >
      {portrait}
      <div className="mb-1 flex h-[calc(1.25rem*var(--ui-space-scale,1))] items-center gap-2">{name}</div>
      {children}
    </div>
  );
}

const FRAME_PORTRAIT =
  "absolute left-0 top-0 aspect-[3/4] w-[calc(3rem*var(--ui-space-scale,1))] rounded-[12px] text-[1.05rem] shadow-[0_1px_3px_rgb(0_0_0/0.14)] max-sm:w-[calc(2.5rem*var(--ui-space-scale,1))] max-sm:rounded-[10px]";

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
      <SpeakerFrame
        className="roleplay-reply"
        speakerId={speaker.id}
        portrait={
          <Portrait
            id={character?.id ?? speaker.id}
            name={speaker.name}
            hasImage={Boolean(character?.hasAvatar)}
            rev={character?.avatarRev ?? 0}
            label={speaker.name}
            className={FRAME_PORTRAIT}
            onClick={
              character && threadId
                ? () =>
                    toggleOnStage(
                      threadId,
                      character.id,
                      aui.thread().getState().messages,
                      castIdsOf(threadId),
                    )
                : undefined
            }
          />
        }
        name={
          <>
            <span
              className={cn(
                "truncate text-ui-14 font-semibold text-foreground",
                !character && "text-muted-foreground",
              )}
            >
              {character ? speaker.name : `${speaker.name} · ${t("characters.chat.deleted")}`}
            </span>
            <MessageTime />
            {/* With "Response model" on, the model shows after the time; the floating badge would sit on the name. */}
            <MessageResponseModelBadge className="roleplay-model-badge min-w-0" />
          </>
        }
      >
        {children}
      </SpeakerFrame>
    </RoleplaySpeakerNameContext.Provider>
  );
}

/** Wraps the user's own message in a character chat the same way: the persona's picture, name and time. */
export function RoleplayUserFrame({ children }: { children: ReactNode }) {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const thread = useThreadScene(threadId);
  // Speaking as the account itself shows the account's own picture.
  const profileAvatar = useUserProfileStore((s) => s.avatarDataUrl);
  if (!thread) return <>{children}</>;
  const { persona } = thread;
  return (
    <SpeakerFrame
      className="roleplay-user"
      portrait={
        <Portrait
          kind="persona"
          id={persona.id}
          name={persona.name}
          hasImage={persona.hasAvatar}
          rev={persona.avatarRev}
          imageUrl={persona.id ? undefined : profileAvatar}
          label={persona.name}
          className={FRAME_PORTRAIT}
        />
      }
      name={
        <>
          <span className="truncate text-ui-14 font-semibold text-foreground" data-testid="persona-tag">
            {persona.name}
          </span>
          <MessageTime />
        </>
      }
    >
      {children}
    </SpeakerFrame>
  );
}

export function useIsRoleplayThread(threadId: string | null | undefined): boolean {
  return useThreadScene(threadId) !== null;
}

/** The hidden bookkeeping turn this message is (the opening or a group turn marker), else null. */
export function useHiddenRoleplayTurn(): "opening" | "turn" | null {
  return useAuiState(({ message }) => {
    if (message.role !== "user") return null;
    const kind = roleplayMetaOf(message)?.kind;
    return kind === "opening" || kind === "turn" ? kind : null;
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
