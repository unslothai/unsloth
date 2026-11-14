// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { audioVoiceLabel } from "@/features/audio/audio-options";
import { audioModelsForTask } from "@/features/audio/catalog";
import { VoicePicker } from "@/features/audio/components/voice-picker";
import { pageModels } from "@/features/audio/page-models";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { useAudioVoicesStore } from "@/features/audio/stores/audio-voices-store";
import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat";
import {
  CHAT_MODEL_VOICE_SPEAKERS,
  VOICE_CHAT_MODEL_VOICE,
  VOICE_TTS_MODELS,
  chatModelSpeaks,
  isChatModelVoice,
  speakerCarriesOver,
  voiceTtsCanClone,
} from "@/features/chat/voice/voice-models";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { type FC, useEffect, useMemo, useState } from "react";

interface VoiceModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  loaded?: boolean;
  onClose?: () => void;
}

// The family the chat model's own voice reports for its two speakers.
const CHAT_MODEL_FAMILY = "chat_model";

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
  // Opt-in only, and only offered while the chat model can speak.
  if (!speaks && !isChatModelVoice(value)) return picker;
  return (
    <div className="flex min-w-0 flex-col">
      <button
        type="button"
        disabled={!speaks}
        onClick={() => {
          pick(VOICE_CHAT_MODEL_VOICE);
          onClose?.();
        }}
        className={cn(
          "mx-3 mt-2 flex flex-col items-start rounded-xl px-2.5 py-1.5 text-left transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none disabled:opacity-50",
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
}

function useBuiltinSpeakers(loadedId: string): BuiltinSpeakers | null {
  const [speakers, setSpeakers] = useState<BuiltinSpeakers | null>(null);
  useEffect(() => {
    setSpeakers(null);
    if (!loadedId) return;
    if (isChatModelVoice(loadedId)) {
      setSpeakers({ family: CHAT_MODEL_FAMILY, values: [...CHAT_MODEL_VOICE_SPEAKERS] });
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await authFetch(
          `/api/inference/status?model=${encodeURIComponent(loadedId)}`,
        );
        if (!res.ok) return;
        const status = (await res.json()) as {
          audio_family?: string | null;
          audio_options?: { name?: string; values?: unknown }[] | null;
        };
        const voice = (status.audio_options ?? []).find((o) => o?.name === "voice");
        const values = Array.isArray(voice?.values)
          ? voice.values.filter((v): v is string => typeof v === "string")
          : [];
        if (!cancelled) setSpeakers({ family: status.audio_family ?? null, values });
      } catch {
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadedId]);
  return speakers;
}

function speakerLabel(
  speaker: string,
  family: string | null,
  t: ReturnType<typeof useT>,
): string {
  if (family !== CHAT_MODEL_FAMILY) return audioVoiceLabel(speaker, family);
  return t(speaker === "male" ? "voicePanel.speakerMale" : "voicePanel.speakerFemale");
}

export interface VoicePreset {
  label: string | null;
  randomEachSentence: boolean;
  speakers: BuiltinSpeakers | null;
  canClone: boolean;
  choosable: boolean;
}

export function useVoicePreset(modelId: string, loaded: boolean): VoicePreset {
  const t = useT();
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const savedVoices = useAudioVoicesStore((s) => s.voices);
  const speaker = useChatRuntimeStore((s) => s.selectedVoiceSpeaker);
  const speakers = useBuiltinSpeakers(loaded ? modelId : "");
  const canClone = voiceTtsCanClone(modelId);
  const savedName = savedVoices.find((v) => v.id === savedVoiceId)?.name;
  const builtin =
    speakers && speakers.values.includes(speaker)
      ? speakerLabel(speaker, speakers.family, t)
      : null;
  return {
    label: savedName ?? builtin,
    randomEachSentence: speakers?.family === "omnivoice" && !savedVoiceId,
    speakers,
    canClone,
    choosable: canClone || (speakers?.values.length ?? 0) > 1,
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

  useEffect(() => {
    if (preset.canClone) void useAudioVoicesStore.getState().refresh();
  }, [preset.canClone]);

  const row =
    "mb-0.5 flex w-full items-center rounded-full px-2 py-1.5 text-left text-ui-13 transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none";
  const heading =
    "px-2 pb-1 pt-0.5 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground";
  return (
    <div className="flex min-w-0 flex-col p-1.5">
      {speakers && speakers.values.length > 1 && (
        <>
          <div className={heading}>{labels.builtin}</div>
          {speakers.values.map((choice) => (
            <button
              key={choice}
              type="button"
              onClick={() => {
                setSpeaker(choice);
                setSavedVoiceId("");
                onClose();
              }}
              className={cn(row, !savedVoiceId && speaker === choice && "bg-muted")}
            >
              {speakerLabel(choice, speakers.family, t)}
            </button>
          ))}
        </>
      )}
      {preset.canClone && (
        <>
          <div className={heading}>{labels.saved}</div>
          <button
            type="button"
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
  );
};
