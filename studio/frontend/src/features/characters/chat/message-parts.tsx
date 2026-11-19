// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLocale, useT } from "@/i18n";
import { formatMessageDate } from "@/lib/format-message-date";
import { messageTimestamp, replyFinishedTimestamp } from "@/lib/message-timestamp";
import { useTimeFormat, withTimeFormat } from "@/lib/time-format";
import { cn } from "@/lib/utils";
import { useUserProfileStore } from "@/features/profile";
import { useAui, useAuiState } from "@assistant-ui/react";
import { createContext, useContext, type ReactNode } from "react";
import { MessageResponseModelBadge } from "@/components/assistant-ui/message-response-details-sheet";
import { useMessagePartTextSafe } from "@/components/assistant-ui/stale-safe-scopes";
import { Portrait } from "../components/portrait";
import { RoleplayText } from "../components/roleplay-text";
import { toggleOnStage } from "../stage-focus";
import { hiddenRoleplayTurnKind, roleplayMetaOf } from "../meta";
import { useCharactersStore } from "../store";
import { useCharacterChat } from "./character-chat-context";
import { useThreadScene } from "./use-thread-scene";

/** The speaking character's name, so a reply's leading "Name:" can be hidden. */
const RoleplaySpeakerNameContext = createContext<string | null>(null);

export function useRoleplaySpeakerName(): string | null {
  return useContext(RoleplaySpeakerNameContext);
}

/** When this message was written (a reply: when it finished), with the full date on hover. */
function MessageTime({ className }: { className?: string }) {
  const t = useT();
  const locale = useLocale();
  useTimeFormat();
  const at = useAuiState(({ message }) =>
    message.role === "assistant" ? replyFinishedTimestamp(message) : messageTimestamp(message),
  );
  if (at === undefined) return null;
  const date = new Date(at);
  const full = date.toLocaleString(locale, withTimeFormat({ dateStyle: "full", timeStyle: "short" }, locale));
  return (
    <time
      dateTime={date.toISOString()}
      title={full}
      className={cn("shrink-0 text-ui-12 font-normal tabular-nums text-muted-foreground", className)}
      data-testid="roleplay-message-time"
    >
      {formatMessageDate(at, Date.now(), locale, {
        today: (time) => time,
        yesterday: (time) => t("common.yesterdayAt", { time }),
      })}
    </time>
  );
}

/** Who said this reply: its stamped speaker, else the chat's only character. */
function useReplySpeaker() {
  const thread = useCharacterChat();
  const group = Boolean(thread?.group);
  // Strings, not the record: a selector must return a stable value or the store re-renders forever.
  const speakerId = useAuiState(({ message, thread: state }) => {
    const meta = roleplayMetaOf(message);
    if (meta?.kind === "reply") return meta.speakerId;
    if (!group || message.index < 1) return null;
    // A reply saved without its speaker (an older run) answers the turn marker before it.
    const previous = roleplayMetaOf(state.messages[message.index - 1]);
    return previous?.kind === "turn" ? previous.speakerId : null;
  });
  const speakerName = useAuiState(({ message }) => {
    const meta = roleplayMetaOf(message);
    return meta?.kind === "reply" ? meta.speakerName : "";
  });
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
  const thread = useCharacterChat();
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
            label={character && thread ? t("characters.chat.showOnStage", { name: speaker.name }) : speaker.name}
            className={FRAME_PORTRAIT}
            onClick={
              character && thread
                ? () =>
                    toggleOnStage(
                      thread.threadId,
                      character.id,
                      aui.thread().getState().messages,
                      thread.cast.map((member) => member.id),
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
  const thread = useCharacterChat();
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

/** For UI outside CharacterStage (the settings sheet): whether this thread is a character chat. */
export function useIsRoleplayThread(threadId: string | null | undefined): boolean {
  return useThreadScene(threadId) !== null;
}

/** The hidden bookkeeping turn this message is (the opening or a group turn marker), else null. */
export function useHiddenRoleplayTurn(): "opening" | "turn" | null {
  return useAuiState(({ message }) => hiddenRoleplayTurnKind(message));
}

function RoleplayUserText() {
  const { text } = useMessagePartTextSafe();
  return <RoleplayText text={text} className="whitespace-pre-wrap" />;
}

const ROLEPLAY_USER_PARTS = { Text: RoleplayUserText };

/** In a character chat, *actions* in the user's own messages read as italics too. */
export function useRoleplayUserParts(): typeof ROLEPLAY_USER_PARTS | undefined {
  return useCharacterChat() ? ROLEPLAY_USER_PARTS : undefined;
}
