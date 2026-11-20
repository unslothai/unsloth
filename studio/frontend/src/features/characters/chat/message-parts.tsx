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
import { ActionText } from "../components/action-text";
import { toggleOnStage } from "../stage-focus";
import { hiddenRoleplayTurnKind, roleplayMetaOf } from "../meta";
import { stripSpeakerPrefix } from "../prompt/build";
import { tidyCharacterDisplay } from "../prompt/display";
import { useCharactersStore } from "../store";
import { useCharacterChat } from "./character-chat-context";

/** The speaking character's name, so a reply's leading "Name:" can be hidden. */
const CharacterSpeakerNameContext = createContext<string | null>(null);

function isTextPart(aui: ReturnType<typeof useAui>): boolean {
  try {
    return aui.part().getState().type === "text";
  } catch {
    // A part that went stale mid-render: show it unchanged.
    return false;
  }
}

/** Inside a framed character reply, which shows the response model in its name line. */
export function useInCharacterReply(): boolean {
  return useContext(CharacterSpeakerNameContext) !== null;
}

/** A character reply's text part without the leading "Name:" the chat already shows; reasoning keeps its own lines. */
export function useCharacterReplyText(text: string): string {
  const aui = useAui();
  const speakerName = useContext(CharacterSpeakerNameContext);
  return speakerName && isTextPart(aui) ? tidyCharacterDisplay(stripSpeakerPrefix(text, speakerName)) : text;
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
  const isGroup = Boolean(thread?.isGroup);
  // Strings, not the record: a selector must return a stable value or the store re-renders forever.
  const speakerId = useAuiState(({ message, thread: state }) => {
    const meta = roleplayMetaOf(message);
    if (meta?.kind === "reply") return meta.speakerId;
    if (!isGroup || message.index < 1) return null;
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
    return { id: speakerId, name: character?.name ?? speakerName, character, isGroup: thread.isGroup };
  }
  if (thread.cast.length === 1) {
    const only = thread.cast[0];
    return { id: only.id, name: only.name, character: only, isGroup: false };
  }
  return null;
}

/** Shared by replies and the user's own messages. */
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
export function CharacterReplyFrame({ children }: { children: ReactNode }) {
  const t = useT();
  const speaker = useReplySpeaker();
  const thread = useCharacterChat();
  const aui = useAui();
  if (!speaker) return <>{children}</>;
  const character = speaker.character;
  return (
    <CharacterSpeakerNameContext.Provider value={speaker.name}>
      <SpeakerFrame
        className="character-reply"
        speakerId={speaker.id}
        portrait={
          <Portrait
            subject={{
              id: character?.id ?? speaker.id,
              name: speaker.name,
              hasAvatar: Boolean(character?.hasAvatar),
              avatarRev: character?.avatarRev ?? 0,
            }}
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
            <MessageResponseModelBadge className="min-w-0" />
          </>
        }
      >
        {children}
      </SpeakerFrame>
    </CharacterSpeakerNameContext.Provider>
  );
}

export function PersonaMessageFrame({ children }: { children: ReactNode }) {
  const thread = useCharacterChat();
  // Speaking as the account itself shows the account's own avatar.
  const profileAvatar = useUserProfileStore((s) => s.avatarDataUrl);
  if (!thread) return <>{children}</>;
  const { persona } = thread;
  return (
    <SpeakerFrame
      className="character-user"
      portrait={
        <Portrait
          kind="persona"
          subject={persona}
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

/** The hidden bookkeeping turn this message is (the opening or a group turn marker), else null. */
export function useHiddenRoleplayTurn(): "opening" | "turn" | null {
  return useAuiState(({ message }) => hiddenRoleplayTurnKind(message));
}

function CharacterUserText() {
  const { text } = useMessagePartTextSafe();
  return <ActionText text={text} className="whitespace-pre-wrap" />;
}

const CHARACTER_USER_PARTS = { Text: CharacterUserText };

/** In a character chat, *actions* in the user's own messages read as italics too. */
export function useCharacterUserParts(): typeof CHARACTER_USER_PARTS | undefined {
  return useCharacterChat() ? CHARACTER_USER_PARTS : undefined;
}
