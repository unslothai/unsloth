// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { errorMessage } from "@/lib/error-message";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  BubbleChatIcon,
  Cancel01Icon,
  PlusSignIcon,
  ViewIcon,
  VolumeHighIcon,
  VolumeMute01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { useOpenCharacterChat } from "../actions";
import { updateCharacter, type ScenePatch } from "../api";
import { ClampedNumberInput } from "../components/clamped-number-input";
import { Portrait } from "../components/portrait";
import { useAccountName } from "../persona";
import { DEFAULT_NOTE_DEPTH } from "../prompt/defaults";
import { useCharactersStore } from "../store";
import type { Character, ReplyOrder } from "../types";
import { CharacterPickerDialog } from "./character-picker";
import { PromptInspectorDialog } from "./prompt-inspector";
import { useBridgedAsk } from "./ask-bridge";
import { useThreadScene, type ThreadScene } from "./use-thread-scene";

const PROFILE_PERSONA = "__profile__";

export function CharacterChatPanel({ threadId }: { threadId: string }) {
  const t = useT();
  const navigate = useNavigate();
  const thread = useThreadScene(threadId);
  const personas = useCharactersStore((s) => s.personas);
  const [adding, setAdding] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const { open: openChat, starting } = useOpenCharacterChat();
  const accountName = useAccountName();
  const ask = useBridgedAsk(threadId);
  const running = useChatRuntimeStore((s) => Boolean(s.runningByThreadId[threadId]));
  if (!thread) return null;
  const { scene, cast, isGroup } = thread;
  const settings = scene.settings;

  async function patch(update: ScenePatch) {
    try {
      await useCharactersStore.getState().patchScene(threadId, update);
    } catch (error) {
      toast.error(t("characters.panel.updateFailed"), { description: errorMessage(error) });
    }
  }
  const members = [...scene.cast].sort((a, b) => a.position - b.position);
  // The server refuses a group with every member muted; these only disable the buttons that would.
  const leavesNoSpeaker = (next: { muted: boolean }[]) => next.length > 1 && next.every((m) => m.muted);
  const setSettings = (next: NonNullable<ScenePatch["settings"]>) => void patch({ settings: next });

  return (
    <div className="flex flex-col gap-5 pb-2" data-testid="character-chat-panel">
      <div className="flex flex-col gap-1.5">
        <span className="text-ui-12 font-medium text-nav-fg-muted">{t("characters.panel.cast")}</span>
        <ul className="flex flex-col gap-0.5">
          {cast.map((character) => {
            const member = members.find((m) => m.characterId === character.id);
            const muted = Boolean(member?.muted);
            return (
              <li key={character.id} className="group/member flex items-center gap-2.5 rounded-xl py-1 pl-1 pr-1 hover:bg-nav-surface-hover">
                <Portrait
                  subject={character}
                  dimmed={muted}
                  className="size-8 rounded-[10px] text-[0.7rem]"
                />
                <button
                  type="button"
                  className="min-w-0 flex-1 cursor-pointer truncate text-left text-ui-14 font-medium text-nav-fg hover:underline"
                  onClick={() => void navigate({ to: "/characters/$characterId", params: { characterId: character.id } })}
                >
                  {character.name}
                  {muted ? <span className="ml-1.5 text-ui-12 font-normal text-nav-fg-muted">{t("characters.panel.muted")}</span> : null}
                </button>
                {isGroup ? (
                  // Hidden until hovered so names fit; always shown on touch, where hidden buttons still take taps.
                  <span className="hidden shrink-0 items-center group-focus-within/member:flex group-hover/member:flex pointer-coarse:flex">
                    <IconAction
                      label={t("characters.chat.replyNow", { name: character.name })}
                      icon={BubbleChatIcon}
                      disabled={!ask || running}
                      onClick={() => ask?.(character.id, character.name)}
                    />
                    <IconAction
                      label={muted ? t("characters.panel.unmute", { name: character.name }) : t("characters.panel.mute", { name: character.name })}
                      icon={muted ? VolumeMute01Icon : VolumeHighIcon}
                      disabled={leavesNoSpeaker(members.map((m) => (m.characterId === character.id ? { muted: !muted } : m)))}
                      onClick={() => void patch({ mute: { [character.id]: !muted } })}
                    />
                    <IconAction
                      label={t("characters.panel.remove", { name: character.name })}
                      icon={Cancel01Icon}
                      disabled={leavesNoSpeaker(members.filter((m) => m.characterId !== character.id))}
                      onClick={() => void patch({ remove: [character.id] })}
                    />
                  </span>
                ) : muted ? (
                  // A group cut down to one muted member: it must still be possible to unmute it.
                  <IconAction
                    label={t("characters.panel.unmute", { name: character.name })}
                    icon={VolumeMute01Icon}
                    onClick={() => void patch({ mute: { [character.id]: false } })}
                  />
                ) : null}
              </li>
            );
          })}
        </ul>
        <div className="flex flex-wrap gap-1.5">
          <Button variant="ghost" size="sm" className="justify-start" onClick={() => setAdding(true)}>
            <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
            {t("characters.panel.addCharacter")}
          </Button>
          <Button variant="ghost" size="sm" className="justify-start" onClick={() => setInspecting(true)}>
            <HugeiconsIcon icon={ViewIcon} strokeWidth={1.75} className="size-icon" />
            {t("characters.panel.viewPrompt")}
          </Button>
        </div>
      </div>

      {isGroup ? (
        <div className="flex flex-col gap-2">
          <span className="text-ui-12 font-medium text-nav-fg-muted">{t("characters.panel.replyOrder")}</span>
          <ReplyOrderPicker value={settings.replyOrder ?? "natural"} onChange={(replyOrder) => setSettings({ replyOrder })} />
          <label className="flex items-center justify-between gap-3 text-ui-13 text-nav-fg">
            {t("characters.panel.selfReplies")}
            <Switch checked={Boolean(settings.allowSelfReplies)} onCheckedChange={(allowSelfReplies) => setSettings({ allowSelfReplies })} />
          </label>
        </div>
      ) : null}

      <div className="flex flex-col gap-1.5">
        <span className="text-ui-12 font-medium text-nav-fg-muted">{t("characters.panel.persona")}</span>
        <Select
          value={thread.persona.id ?? PROFILE_PERSONA}
          onValueChange={(value) => void patch({ personaId: value === PROFILE_PERSONA ? null : value })}
        >
          <SelectTrigger size="sm" className="w-full" aria-label={t("characters.panel.persona")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={PROFILE_PERSONA}>{t("characters.panel.personaProfile", { name: accountName })}</SelectItem>
            {personas.map((persona) => (
              <SelectItem key={persona.id} value={persona.id}>
                {persona.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <button
          type="button"
          className="self-start text-ui-12 text-nav-fg-muted underline-offset-4 hover:text-nav-fg hover:underline"
          onClick={() => void navigate({ to: "/characters", search: { view: "personas" } })}
        >
          {t("characters.panel.managePersonas")}
        </button>
      </div>

      <SavedTextarea
        label={t("characters.panel.scene")}
        hint={t("characters.panel.sceneHint")}
        placeholder={t("characters.panel.scenePlaceholder")}
        value={settings.scenario ?? ""}
        onSave={(scenario) => setSettings({ scenario: scenario || null })}
      />

      <div className="flex flex-col gap-1.5">
        <SavedTextarea
          label={t("characters.panel.authorsNote")}
          hint={t("characters.panel.authorsNoteHint")}
          placeholder={t("characters.panel.authorsNotePlaceholder")}
          value={settings.authorsNote?.text ?? ""}
          onSave={(text) => setSettings({ authorsNote: text ? { text } : null })}
        />
        <label className="flex items-center gap-2 self-end text-ui-12 text-nav-fg-muted">
          {t("characters.panel.noteDepth")}
          <ClampedNumberInput
            min={0}
            max={100}
            commitOnBlur={true}
            className="h-7 w-16 text-ui-13"
            value={settings.authorsNote?.depth ?? DEFAULT_NOTE_DEPTH}
            onChange={(depth) => setSettings({ authorsNote: { depth } })}
          />
        </label>
      </div>

      <QuickEdit thread={thread} />

      <Button
        variant="outline"
        size="sm"
        disabled={starting}
        onClick={() => void openChat(cast.map((c) => c.id))}
      >
        {t("characters.chat.newChatSameCast")}
      </Button>

      <CharacterPickerDialog
        open={adding}
        mode="add"
        exclude={cast.map((c) => c.id)}
        onClose={() => setAdding(false)}
        onPick={(ids) => {
          setAdding(false);
          void patch({ add: ids });
        }}
      />
      <PromptInspectorDialog open={inspecting} threadId={threadId} onClose={() => setInspecting(false)} />
    </div>
  );
}

function IconAction({
  label,
  icon,
  onClick,
  disabled = false,
}: {
  label: string;
  icon: IconSvgElement;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
          className="flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-lg text-nav-fg-muted hover:bg-background hover:text-nav-fg disabled:cursor-default disabled:text-nav-fg-muted/40 disabled:hover:bg-transparent"
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function ReplyOrderPicker({ value, onChange }: { value: ReplyOrder; onChange: (value: ReplyOrder) => void }) {
  const t = useT();
  const options: { value: ReplyOrder; label: string; hint: string }[] = [
    { value: "natural", label: t("characters.panel.orderNatural"), hint: t("characters.panel.orderNaturalHint") },
    { value: "list", label: t("characters.panel.orderList"), hint: t("characters.panel.orderListHint") },
    { value: "manual", label: t("characters.panel.orderManual"), hint: t("characters.panel.orderManualHint") },
  ];
  const current = options.find((option) => option.value === value) ?? options[0];
  return (
    <div className="flex flex-col gap-1.5">
      <Select value={current.value} onValueChange={(next) => onChange(next as ReplyOrder)}>
        <SelectTrigger size="sm" className="w-full" aria-label={t("characters.panel.replyOrder")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-ui-12 leading-snug text-nav-fg-muted">{current.hint}</p>
    </div>
  );
}

/** A textarea that saves on blur, so typing never fights a round trip. */
function SavedTextarea({
  label,
  hint,
  placeholder,
  value,
  onSave,
  rows = 2,
}: {
  label: string;
  hint?: string;
  placeholder?: string;
  value: string;
  onSave: (value: string) => void;
  rows?: number;
}) {
  const [draft, setDraft] = useState(value);
  // Typed text wins over a stored value arriving meanwhile; untyped, leaving the field saves nothing.
  const typed = useRef(false);
  useEffect(() => {
    if (!typed.current) setDraft(value);
  }, [value]);
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-ui-12 font-medium text-nav-fg-muted">{label}</span>
      <Textarea
        value={draft}
        rows={rows}
        placeholder={placeholder}
        className="max-h-48 min-h-14 text-ui-13"
        onChange={(event) => {
          typed.current = true;
          setDraft(event.target.value);
        }}
        onBlur={() => {
          if (!typed.current) return;
          typed.current = false;
          if (draft.trim() !== value.trim()) onSave(draft.trim());
        }}
      />
      {hint ? <span className="text-ui-12 leading-snug text-nav-fg-muted">{hint}</span> : null}
    </label>
  );
}

/** Edit-while-testing: the speaking character's core fields, saved to the library. */
function QuickEdit({ thread }: { thread: ThreadScene }) {
  const t = useT();
  const [targetId, setTargetId] = useState(thread.cast[0].id);
  const target: Character = thread.cast.find((c) => c.id === targetId) ?? thread.cast[0];
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    if (!saved) return;
    const timer = window.setTimeout(() => setSaved(false), 2500);
    return () => window.clearTimeout(timer);
  }, [saved]);

  async function save(field: "description" | "personality", value: string) {
    try {
      const updated = await updateCharacter(target.id, { [field]: value });
      useCharactersStore.getState().upsert(updated);
      setSaved(true);
    } catch (error) {
      toast.error(t("characters.editor.saveFailed"), { description: errorMessage(error) });
    }
  }

  return (
    <details className="group/quick rounded-xl bg-muted/50 px-3 py-2 dark:bg-card">
      <summary className="flex cursor-pointer list-none items-center justify-between text-ui-13 font-medium text-nav-fg">
        {t("characters.panel.quickEdit")}
        {saved ? <span className="text-ui-12 text-primary">{t("characters.panel.saved")}</span> : null}
      </summary>
      <div className="mt-2 flex flex-col gap-3">
        <p className="text-ui-12 leading-snug text-nav-fg-muted">{t("characters.panel.quickEditHint")}</p>
        {thread.isGroup ? (
          <Select value={target.id} onValueChange={setTargetId}>
            <SelectTrigger size="sm" className="w-full" aria-label={t("characters.panel.quickEdit")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {thread.cast.map((character) => (
                <SelectItem key={character.id} value={character.id}>
                  {character.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : null}
        <SavedTextarea
          key={`${target.id}-description`}
          label={t("characters.editor.description")}
          value={target.description}
          rows={5}
          onSave={(value) => void save("description", value)}
        />
        <SavedTextarea
          key={`${target.id}-personality`}
          label={t("characters.editor.personality")}
          value={target.personality}
          onSave={(value) => void save("personality", value)}
        />
      </div>
    </details>
  );
}
