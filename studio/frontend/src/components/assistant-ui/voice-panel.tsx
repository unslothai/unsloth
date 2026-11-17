// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  setVoiceDecision,
  useVoiceDecision,
  type VoiceLoadState,
} from "@/features/chat/voice/decision-checkpoint";
import { useChatModelStatus } from "@/features/chat/voice/chat-model-ready";
import { useVadDetectors } from "@/features/chat/voice/use-vad-detectors";
import { vadDisplayName } from "@/features/chat/voice/vad-client";
import { isChatModelVoice } from "@/features/chat/voice/voice-models";
import { clearVoiceModelRest } from "@/features/chat/voice/voice-models-client";
import { voicePillIconClass } from "@/features/chat/voice/voice-pill-tone";
import { ORB_IDLE_GRADIENT } from "@/features/chat/voice/orb-config";
import { DotTag } from "@/features/hub";
import { useVoiceSettingsStore } from "@/features/settings";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
// Deep import: through the eager model-picker index it would land in the startup bundle.
// eslint-disable-next-line no-restricted-imports
import { DecisionPicker } from "@/features/model-picker/components/model-selector/decision-picker";
import { type TranslationKey, useT } from "@/i18n";
import {
  ChevronDownStandardIcon,
  ChevronLeftStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { SttModelSelector, sttRepoForKey } from "./stt-model-selector";
import { VadModelSelector } from "./vad-model-selector";
import {
  useVoicePreset,
  VoiceModelSelector,
  VoicePresetPanel,
} from "./voice-model-selector";

type Role = "listen" | "transcribe" | "decide" | "speak" | "voice";
const ROLES: Role[] = ["listen", "transcribe", "decide", "speak", "voice"];
const PICKER_ROLES = new Set<Role>(["listen", "transcribe", "decide", "speak"]);

interface VoicePanelProps {
  vadModelId: string;
  onVadModelChange: (id: string) => void;
  vadLoaded: boolean;
  sttModelId: string;
  onSttModelChange: (id: string) => void;
  sttLoaded: boolean;
  /** The picked transcriber is not downloaded: voice mode cannot hear until it is. */
  sttDownload: "needed" | "downloading" | null;
  onSttDownload: () => void;
  ttsModelId: string;
  onTtsModelChange: (id: string) => void;
  ttsLoading: boolean;
  ttsLoaded: boolean;
  /** No chat model is loaded: the models can still be chosen, but voice mode cannot start. */
  disabled?: boolean;
  anchor?: "header" | "composer";
}

/** The panel's models and their load state, from the chat store. */
function useVoicePanelProps(): Omit<VoicePanelProps, "anchor"> {
  const s = useChatRuntimeStore(
    useShallow((s) => ({
      vadModelId: s.selectedVadModelId,
      onVadModelChange: s.setSelectedVadModelId,
      vadLoaded: s.vadSlotLoaded,
      sttModelId: s.selectedSttModelId,
      onSttModelChange: s.setSelectedSttModelId,
      sttLoaded: s.sttSlotLoaded,
      sttDownload: s.sttSlotDownload,
      ttsModelId: s.selectedVoiceModelId,
      onTtsModelChange: s.setSelectedVoiceModelId,
      ttsLoading: s.voiceSlotLoading,
      ttsLoaded: s.voiceSlotLoaded,
      disabled: !s.params.checkpoint,
    })),
  );
  return { ...s, onSttDownload: () => requestSttDownload(s.sttModelId) };
}

/** Voice mode's models, one row per role. "composer": opens from the voice button beside the mic, to
 *  set up and start. "header": the status pill while voice mode runs. */
export const VoicePanel: FC<{ anchor?: "header" | "composer" }> = ({ anchor = "header" }) => (
  <VoicePanelView {...useVoicePanelProps()} anchor={anchor} />
);

const STATE_KEY: Record<VoiceLoadState, TranslationKey> = {
  loaded: "voicePanel.state.loaded",
  loading: "voicePanel.state.loading",
  not_loaded: "voicePanel.state.notLoaded",
  error: "voicePanel.state.error",
};

function shortName(id: string): string {
  const leaf = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  return leaf.replace(/-GGUF$/i, "");
}

const ROW =
  "group/row flex w-full items-center gap-2 rounded-full py-1.5 pl-2.5 pr-2 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50";

// idle: voice mode has not started, so an unloaded slot is waiting, not failing.
const StateTag: FC<{ state: VoiceLoadState; label: string; idle?: boolean }> = ({
  state,
  label,
  idle = false,
}) => (
  <DotTag
    tone={
      idle
        ? "neutral"
        : state === "loaded"
          ? "success"
          : state === "loading"
            ? "downloading"
            : state === "error"
              ? "danger"
              : "warning"
    }
    label={label}
    className="h-[calc(18px*var(--ui-space-scale,1))] gap-1 rounded-md px-1.5"
    dotClassName="size-[calc(5px*var(--ui-space-scale,1))]"
  />
);

const VoicePanelView: FC<VoicePanelProps> = ({
  vadModelId,
  onVadModelChange,
  vadLoaded,
  sttModelId,
  onSttModelChange,
  sttLoaded,
  sttDownload,
  onSttDownload,
  ttsModelId,
  onTtsModelChange,
  ttsLoading,
  ttsLoaded,
  disabled = false,
  anchor = "header",
}) => {
  const t = useT();
  const composer = anchor === "composer";
  // Turning voice on from the composer opens its setup there, with Start in reach.
  const [open, setOpen] = useState(composer);
  const [role, setRole] = useState<Role | null>(null);
  const rowRefs = useRef<Partial<Record<Role, HTMLButtonElement | null>>>({});
  const decision = useVoiceDecision(open);
  const preset = useVoicePreset(ttsModelId, ttsLoaded);
  const vadDetectors = useVadDetectors();
  // "configuring": the pill is up for choosing models, and nothing loads until Start.
  const started = useChatRuntimeStore((s) => s.voiceMode === "active");
  const chatModel = useChatModelStatus();
  const canThink = useChatRuntimeStore((s) => s.supportsReasoning && !s.reasoningAlwaysOn);
  // Voice can only think when the chat model does: with Thinking off in the model menu the switch would do nothing.
  const chatThinks = useChatRuntimeStore((s) => s.reasoningEnabled);
  const voiceThinking = useVoiceSettingsStore((s) => s.voiceThinking);
  const setVoiceThinking = useVoiceSettingsStore((s) => s.setVoiceThinking);
  const start = () => {
    const store = useChatRuntimeStore.getState();
    store.setVoiceOrbCollapsed(false);
    store.setVoiceMode("active");
    setOpen(false);
    setRole(null);
  };

  const ttsState: VoiceLoadState = ttsLoading
    ? "loading"
    : ttsLoaded
      ? "loaded"
      : "not_loaded";
  const decideModel = decision.status?.models.find(
    (m) => m.name === decision.status?.current,
  );
  const decideName = decision.status
    ? decision.status.current
      ? (decideModel?.label ?? shortName(decision.status.current))
      : t("voicePanel.builtinClassifiers")
    : "";
  // Built-in classifiers with no source on the server never load: a setup gap, not a failure.
  const decideUnset =
    decision.status?.current === null &&
    decision.status.builtin_configured === false;
  const rows: Record<
    Role,
    {
      name: string;
      state: VoiceLoadState | null;
      enabled: boolean;
      title?: string;
      unset?: boolean;
      /** Said in place of the state's own name. */
      label?: TranslationKey;
    }
  > = {
    listen: {
      name: vadDisplayName(vadModelId, vadDetectors),
      state: vadLoaded ? "loaded" : "not_loaded",
      enabled: true,
    },
    transcribe: {
      name: shortName(sttRepoForKey(sttModelId)),
      state: sttLoaded ? "loaded" : sttDownload === "downloading" ? "loading" : "not_loaded",
      enabled: true,
      label: sttLoaded
        ? undefined
        : sttDownload === "downloading"
          ? "voicePanel.state.downloading"
          : sttDownload === "needed"
            ? "settings.voice.dictation.sttNotDownloaded"
            : undefined,
    },
    decide: {
      name: decision.failed ? t("voicePanel.decideOwnerOnly") : decideName,
      state: decision.failed ? null : decision.state,
      enabled: !decision.failed && decision.status !== null,
      title: decideUnset
        ? t("voicePanel.decideNotSetUp")
        : decision.state === "error" && decision.status?.error
          ? `${decideName}: ${decision.status.error}`
          : undefined,
      unset: decideUnset,
    },
    speak: {
      name: isChatModelVoice(ttsModelId) ? t("voicePanel.chatModelVoice") : shortName(ttsModelId),
      state: ttsState,
      enabled: true,
    },
    voice: {
      name:
        preset.label ??
        t(
          preset.randomEachSentence
            ? "voicePanel.randomEachSentence"
            : "voicePanel.modelsOwnVoice",
        ),
      state: ttsState,
      enabled: preset.choosable,
    },
  };
  const states = ROLES.map((r) => (rows[r].unset ? null : rows[r].state));
  // Shown only by the header pill, which is up only while voice mode runs.
  const summary = chatModel !== "ready"
    ? t(chatModel === "switching" ? "voicePanel.chatModelLoading" : "voicePanel.chatModelWaiting")
    : sttDownload === "downloading"
    ? t("voicePanel.downloadingTranscriber")
    : states.includes("loading")
    ? t("voicePanel.loading")
    : states.includes("error")
      ? t("voicePanel.error")
      : sttDownload === "needed" && !sttLoaded
          ? t("voicePanel.downloadNeeded")
          : t("voicePanel.ready", {
            n: states.filter((s) => s === "loaded").length,
          });

  const back = () => {
    const from = role;
    setRole(null);
    if (from) requestAnimationFrame(() => rowRefs.current[from]?.focus());
  };

  const pickDecision = async (name: string | null) => {
    try {
      decision.switched(await setVoiceDecision(name));
      clearVoiceModelRest();
      back();
    } catch (e) {
      toast.error(t("voicePanel.decideSwitchFailed"), {
        description: e instanceof Error ? e.message : undefined,
      });
    }
  };

  const drillIn = (() => {
    switch (role) {
      case "listen":
        return (
          <VadModelSelector
            value={vadModelId}
            onValueChange={onVadModelChange}
            loaded={vadLoaded}
            onClose={back}
          />
        );
      case "transcribe":
        return (
          <SttModelSelector
            value={sttModelId}
            onValueChange={onSttModelChange}
            ready={sttLoaded}
            onClose={back}
          />
        );
      case "speak":
        return (
          <VoiceModelSelector
            value={ttsModelId}
            onValueChange={onTtsModelChange}
            loaded={ttsLoaded}
            onClose={back}
          />
        );
      case "voice":
        return (
          <VoicePresetPanel
            preset={preset}
            labels={{
              modelsOwn: t("voicePanel.modelsOwnVoice"),
              builtin: t("voicePanel.builtinSpeakers"),
              saved: t("voicePanel.savedVoices"),
            }}
            onClose={back}
          />
        );
      case "decide":
        return decision.status ? (
          <div className="w-[min(var(--picker-panel-w),calc(100vw-1rem))] max-w-[calc(100vw-1rem)] pt-2 pl-4 pr-2">
            {decideUnset && (
              <p className="pb-2 pl-1 pr-2 text-xs leading-snug text-muted-foreground">
                {t("voicePanel.decideNotSetUp")}
              </p>
            )}
            <DecisionPicker
              models={decision.status.models}
              current={decision.status.current}
              state={decision.state}
              error={decision.status.error}
              launchDefault={decision.status.launch_default}
              fromSetting={decision.status.source === "setting"}
              onPick={(name) => void pickDecision(name)}
              onFoldersChange={decision.reload}
            />
          </div>
        ) : null;
      default:
        return null;
    }
  })();

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setRole(null);
      }}
    >
      <div className="flex min-w-0 items-center">
        {/* The pill's status changes while voice runs (a model swap, a load): said, not only shown. */}
        {!composer && (
          <output className="sr-only" aria-live="polite" aria-atomic="true">
            {summary}
          </output>
        )}
        <PopoverTrigger asChild={true}>
          {composer ? (
            <button
              type="button"
              aria-label={t("voiceControls.setUp")}
              title={t("voiceControls.setUp")}
              className="flex size-8 items-center justify-center rounded-full transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring data-[state=open]:bg-nav-surface-hover"
            >
              <span
                aria-hidden
                className="size-4 rounded-full ring-2 ring-primary/40"
                style={{ background: ORB_IDLE_GRADIENT }}
              />
            </button>
          ) : (
          <button
            type="button"
            aria-label={t("voicePanel.pillAria")}
            className="flex !h-8.5 min-w-0 items-center gap-2 rounded-[10px] pl-3 pr-1.5 text-sm transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {states.includes("loading") || chatModel === "switching" ? (
              <Spinner className="size-3.5 shrink-0 text-amber-500" />
            ) : (
              <HugeiconsIcon
                icon={AiSpeechIcon}
                strokeWidth={1.75}
                className={cn("size-3.5 shrink-0", voicePillIconClass(states))}
              />
            )}
            <span className="whitespace-nowrap font-heading text-ui-16 font-medium leading-tight text-black dark:text-foreground">
              {t("voicePanel.pill")}
            </span>
            <span className="truncate text-xs leading-tight text-muted-foreground">
              {summary}
            </span>
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
          </button>
          )}
        </PopoverTrigger>
      </div>
      <PopoverContent
        {...(composer
          ? { side: "top", align: "end", sideOffset: 10 }
          : { side: "bottom", align: "start", alignOffset: 10 })}
        dropdownGlow="picker"
        onEscapeKeyDown={(event) => {
          if (role) {
            event.preventDefault();
            back();
          }
        }}
        className={cn(
          "unsloth-model-selector-menu menu-soft-surface ring-0 max-w-[calc(100vw-1rem)] min-w-0 gap-0",
          role && PICKER_ROLES.has(role)
            ? "w-auto p-0"
            : "w-[min(calc(var(--picker-panel-w)*0.85),calc(100vw-1rem))] p-1.5",
        )}
      >
        {role ? (
          <>
            <button
              type="button"
              onClick={back}
              aria-label={t("voicePanel.back")}
              className={cn(
                "flex items-center gap-1 self-start rounded-full py-1 pl-1.5 pr-2.5 text-ui-13 font-medium text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground focus-visible:bg-sidebar-accent focus-visible:outline-none",
                PICKER_ROLES.has(role) && "ml-2.5 mt-2.5",
              )}
            >
              <HugeiconsIcon
                icon={ChevronLeftStandardIcon}
                strokeWidth={1.75}
                className="size-3.5"
              />
              {t(`voicePanel.roles.${role}`)}
            </button>
            {drillIn}
          </>
        ) : (
          <div className="flex flex-col gap-0.5">
            {/* Only the role rows are the list: the switch and the download notice below are not items. */}
            <div role="list" className="flex flex-col gap-0.5">
            {ROLES.map((r) => {
              const row = rows[r];
              return (
                <div role="listitem" key={r}>
                  <button
                    ref={(el) => {
                      rowRefs.current[r] = el;
                    }}
                    type="button"
                    disabled={!row.enabled}
                    onClick={() => setRole(r)}
                    className={ROW}
                  >
                    <span className="w-[calc(4.75rem*var(--ui-font-scale,1))] shrink-0 text-ui-13 text-muted-foreground">
                      {t(`voicePanel.roles.${r}`)}
                    </span>
                    <span
                      // Two lines, not an ellipsis: at 320 px one line held about 10 characters ("Earshot V…").
                      className="line-clamp-2 min-w-0 flex-1 leading-tight [overflow-wrap:anywhere]"
                      title={row.title ?? row.name}
                    >
                      {row.name}
                    </span>
                    {row.state && (
                      <StateTag
                        state={row.state}
                        idle={row.unset || (!started && row.state === "not_loaded")}
                        label={t(
                          row.unset
                            ? "voicePanel.state.notSetUp"
                            : !started && row.state === "not_loaded"
                              ? "voicePanel.state.loadsOnStart"
                              : (row.label ?? STATE_KEY[row.state]),
                        )}
                      />
                    )}
                    <HugeiconsIcon
                      icon={ChevronRightStandardIcon}
                      strokeWidth={1.75}
                      className="size-3.5 shrink-0 text-muted-foreground"
                    />
                  </button>
                </div>
              );
            })}
            </div>
            {started && sttDownload === "needed" && !sttLoaded && (
              <div className="mt-1 flex items-center gap-3 border-t border-border/60 pb-1 pl-2.5 pr-1 pt-2">
                <p className="min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
                  {t("voicePanel.transcriberMissing", { model: rows.transcribe.name })}
                </p>
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  onClick={() => {
                    setOpen(false);
                    onSttDownload();
                  }}
                >
                  {t("settings.voice.dictation.sttDownload")}
                </Button>
              </div>
            )}
            {canThink && (
              <label
                className={cn(
                  "mt-1 flex items-center gap-3 border-t border-border/60 py-2 pl-2.5 pr-2",
                  chatThinks ? "cursor-pointer" : "cursor-default",
                )}
              >
                <span className="min-w-0 flex-1">
                  <span className={cn("block text-sm", !chatThinks && "text-muted-foreground")}>
                    {t("voicePanel.thinkFirst")}
                  </span>
                  <span className="block text-xs leading-snug text-muted-foreground">
                    {t(chatThinks ? "voicePanel.thinkFirstHint" : "voicePanel.thinkNeedsChatThinking")}
                  </span>
                </span>
                <Switch
                  checked={chatThinks && voiceThinking}
                  disabled={!chatThinks}
                  onCheckedChange={setVoiceThinking}
                />
              </label>
            )}
            {!started && (
              <div className="mt-1 flex items-center gap-3 border-t border-border/60 pb-1 pl-2.5 pr-1 pt-2">
                <p className="min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
                  {t(disabled ? "voicePanel.selectChatModelFirst" : "voicePanel.startHint")}
                </p>
                <Button type="button" variant="primary" size="sm" disabled={disabled} onClick={start}>
                  {t("voiceControls.start")}
                </Button>
              </div>
            )}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
};
