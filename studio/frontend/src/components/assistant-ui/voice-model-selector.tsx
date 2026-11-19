// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { audioModelsForTask } from "@/features/audio/catalog";
import { VoicePicker } from "@/features/audio/components/voice-picker";
import { pageModels } from "@/features/audio/page-models";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { useAudioVoicesStore } from "@/features/audio/stores/audio-voices-store";
import { useChatRuntimeStore } from "@/features/chat";
import {
  type VoiceModelSpec,
  useVoiceModelSpec,
  voiceSpeakerLabel,
} from "@/features/chat/voice/use-voice-model-spec";
import { voiceModeOptionGroups } from "@/features/chat/voice/voice-tts-options";
import {
  VOICE_CHAT_MODEL_VOICE,
  VOICE_TTS_MODELS,
  chatModelSpeaks,
  isChatModelVoice,
  speakerCarriesOver,
  voiceTtsCanClone,
  voiceVariesPerSentence,
} from "@/features/chat/voice/voice-models";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { VoiceSoundControls } from "./voice-sound-controls";
import { type FC, useEffect, useMemo, useRef } from "react";

interface VoiceModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  loaded?: boolean;
  onClose?: () => void;
}

const VOICE_TTS_RANK = new Map(
  VOICE_TTS_MODELS.map((id, i) => [id.toLowerCase(), i]),
);
const TTS_TASKS = ["text-to-speech", "text-to-audio"] as const;
const speakRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "speak");

export const VoiceModelSelector: FC<VoiceModelSelectorProps> = ({
  value,
  onValueChange,
  loaded = false,
  onClose,
}) => {
  const t = useT();
  const isMac = usePlatformStore((s) => s.deviceType) === "mac";
  const models = useMemo(
    () =>
      pageModels("speak", audioModelsForTask("tts"), { isMac })
        .filter((m) => VOICE_TTS_RANK.has(m.id.toLowerCase()))
        .sort(
          (a, b) =>
            (VOICE_TTS_RANK.get(a.id.toLowerCase()) ?? 0) -
            (VOICE_TTS_RANK.get(b.id.toLowerCase()) ?? 0),
        ),
    [isMac],
  );
  const speaks = useChatRuntimeStore(chatModelSpeaks);
  const pick = (id: string) => {
    if (!speakerCarriesOver(value, id)) {
      useChatRuntimeStore.getState().setSelectedVoiceSpeaker("");
    }
    onValueChange(id);
  };
  const picker = (
    <ModelSelector
      models={models}
      rowFilter={speakRow}
      value={value}
      loaded={loaded}
      loadedModelIdOverride={loaded ? value : undefined}
      onValueChange={(id) => {
        if (!VOICE_TTS_RANK.has(id.toLowerCase())) {
          toast.info(t("voicePanel.ggufVoicesOnly"));
          return;
        }
        pick(id);
      }}
      variant="ghost"
      className="!h-8.5"
      task={[...TTS_TASKS]}
      catalog={AUDIO_CATALOG}
      embedded={true}
      open={true}
      onOpenChange={(next) => {
        if (!next) onClose?.();
      }}
    />
  );
  if (!speaks && !isChatModelVoice(value)) return picker;
  return (
    <div className="flex min-w-0 flex-col">
      <button
        type="button"
        disabled={!speaks}
        aria-pressed={isChatModelVoice(value)}
        onClick={() => {
          pick(VOICE_CHAT_MODEL_VOICE);
          onClose?.();
        }}
        className={cn(
          "mx-3 mt-2 flex flex-col items-start rounded-xl px-2.5 py-1.5 text-left transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50",
          isChatModelVoice(value) && "bg-muted",
        )}
      >
        <span className="text-sm">{t("voicePanel.chatModelVoice")}</span>
        <span className="text-ui-11 text-muted-foreground">
          {speaks ? t("voicePanel.chatModelVoiceHint") : t("voicePanel.chatModelCannotSpeak")}
        </span>
      </button>
      {picker}
    </div>
  );
};

interface BuiltinSpeakers {
  family: string | null;
  values: string[];
  fallback: string | null;
}

function effectiveSpeaker(speakers: BuiltinSpeakers, picked: string): string | null {
  if (picked && speakers.values.includes(picked)) return picked;
  return speakers.fallback && speakers.values.includes(speakers.fallback) ? speakers.fallback : null;
}


export interface VoicePreset {
  modelId: string;
  spec: VoiceModelSpec;
  label: string | null;
  newVoiceEachSession: boolean;
  speakers: BuiltinSpeakers | null;
  canClone: boolean;
  choosable: boolean;
}

export function useVoicePreset(modelId: string, loaded: boolean): VoicePreset {
  const t = useT();
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const savedVoices = useAudioVoicesStore((s) => s.voices);
  const speaker = useChatRuntimeStore((s) => s.selectedVoiceSpeaker);
  // Read before the model loads, so voice options can be set before Start.
  const spec = useVoiceModelSpec(modelId, loaded);
  const voiceDefault = spec.specs.find((s) => s.name === "voice")?.default;
  const speakers: BuiltinSpeakers | null = spec.speakers.length
    ? {
        family: spec.family,
        values: spec.speakers,
        fallback: typeof voiceDefault === "string" ? voiceDefault : null,
      }
    : null;
  const canClone = voiceTtsCanClone(modelId);
  const savedName = savedVoices.find((v) => v.id === savedVoiceId)?.name;
  const shownSpeaker = speakers ? effectiveSpeaker(speakers, speaker) : null;
  const builtin = speakers && shownSpeaker ? voiceSpeakerLabel(shownSpeaker, speakers.family, t) : null;
  const { primary, more } = voiceModeOptionGroups(spec.specs);
  return {
    modelId,
    spec,
    label: savedName ?? builtin,
    newVoiceEachSession: !savedVoiceId && (spec.family === "omnivoice" || voiceVariesPerSentence(modelId)),
    speakers,
    canClone,
    choosable:
      canClone ||
      (speakers?.values.length ?? 0) > 1 ||
      primary.length + more.length > 0 ||
      (spec.state !== "ready" && !isChatModelVoice(modelId)),
  };
}

export const VoicePresetPanel: FC<{
  preset: VoicePreset;
  labels: { modelsOwn: string; builtin: string; saved: string };
  onClose: () => void;
}> = ({ preset, labels, onClose }) => {
  const t = useT();
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const setSavedVoiceId = useChatRuntimeStore((s) => s.setSelectedVoiceId);
  const speaker = useChatRuntimeStore((s) => s.selectedVoiceSpeaker);
  const setSpeaker = useChatRuntimeStore((s) => s.setSelectedVoiceSpeaker);
  const speakers = preset.speakers;
  const heard = !savedVoiceId && speakers ? effectiveSpeaker(speakers, speaker) : null;
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (preset.canClone) void useAudioVoicesStore.getState().refresh();
  }, [preset.canClone]);

  const listed = speakers?.values.length ?? 0;
  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>('[aria-pressed="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [listed]);

  const row =
    "mb-0.5 flex w-full items-center rounded-full px-2 py-1.5 text-left text-ui-13 transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";
  const heading =
    "px-2 pb-1 pt-0.5 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground";
  const choices = (speakers && speakers.values.length > 1) || preset.canClone;
  return (
    <div className="flex min-w-0 max-h-[calc(var(--radix-popover-content-available-height,70vh)-4rem)] flex-col overflow-y-auto overscroll-contain p-1.5">
      <div
        ref={listRef}
        className={cn("flex min-w-0 flex-col", choices && "max-h-[min(15rem,40vh)] overflow-y-auto overscroll-contain")}
      >
      {speakers && speakers.values.length > 1 && (
        <>
          <div className={heading}>{labels.builtin}</div>
          {speakers.values.map((choice) => (
            <button
              key={choice}
              type="button"
              aria-pressed={heard === choice}
              onClick={() => {
                setSpeaker(choice);
                setSavedVoiceId("");
                onClose();
              }}
              className={cn(row, "justify-between gap-2", heard === choice && "bg-muted")}
            >
              <span className="truncate">{voiceSpeakerLabel(choice, speakers.family, t)}</span>
              {heard === choice && (
                <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3.5 shrink-0" aria-hidden="true" />
              )}
            </button>
          ))}
        </>
      )}
      {preset.canClone && (
        <>
          <div className={heading}>{labels.saved}</div>
          <button
            type="button"
            aria-pressed={!savedVoiceId}
            onClick={() => {
              setSavedVoiceId("");
              onClose();
            }}
            className={cn(row, !savedVoiceId && "bg-muted")}
          >
            {labels.modelsOwn}
          </button>
          <VoicePicker
            selectedId={savedVoiceId || null}
            onSelect={(voice) => {
              setSavedVoiceId(voice.id);
              onClose();
            }}
            onDeselect={() => setSavedVoiceId("")}
          />
        </>
      )}
      </div>
      <VoiceSoundControls
        modelId={preset.modelId}
        spec={preset.spec}
        className={cn(choices && "mt-1.5 border-t border-border/60 pt-3")}
      />
    </div>
  );
};
