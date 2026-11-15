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
import { useVadDetectors } from "@/features/chat/voice/use-vad-detectors";
import { vadDisplayName } from "@/features/chat/voice/vad-client";
import { isChatModelVoice } from "@/features/chat/voice/voice-models";
import { voicePillIconClass } from "@/features/chat/voice/voice-pill-tone";
import { DotTag } from "@/features/hub";
import { useVoiceSettingsStore } from "@/features/settings";
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

export interface VoicePanelProps {
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
  disabled?: boolean;
}

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

export const VoicePanel: FC<VoicePanelProps> = ({
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
}) => {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [role, setRole] = useState<Role | null>(null);
  const rowRefs = useRef<Partial<Record<Role, HTMLButtonElement | null>>>({});
  const decision = useVoiceDecision(open);
  const preset = useVoicePreset(ttsModelId, ttsLoaded);
  const vadDetectors = useVadDetectors();
  // "configuring": the pill is up for choosing models, and nothing loads until Start.
  const started = useChatRuntimeStore((s) => s.voiceMode === "active");
  const canThink = useChatRuntimeStore((s) => s.supportsReasoning && !s.reasoningAlwaysOn);
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
  const summary = states.includes("loading")
    ? t("voicePanel.loading")
    : states.includes("error")
      ? t("voicePanel.error")
      : !started
        ? t("voicePanel.notStarted")
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
      open={!disabled && open}
      onOpenChange={(next) => {
        setOpen(!disabled && next);
        if (!next) setRole(null);
      }}
    >
      <div
        className={cn(
          "flex min-w-0 items-center",
          disabled && "pointer-events-none opacity-50",
        )}
        aria-disabled={disabled || undefined}
        title={disabled ? t("voicePanel.selectChatModelFirst") : undefined}
      >
        <PopoverTrigger asChild={true}>
          <button
            type="button"
            aria-label={t("voicePanel.pillAria")}
            className="flex !h-8.5 min-w-0 items-center gap-2 rounded-[10px] pl-3 pr-1.5 text-sm transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {states.includes("loading") ? (
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
        </PopoverTrigger>
      </div>
      <PopoverContent
        align="start"
        alignOffset={10}
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
                      className="min-w-0 flex-1 truncate whitespace-nowrap"
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
              <label className="mt-1 flex cursor-pointer items-center gap-3 border-t border-border/60 py-2 pl-2.5 pr-2">
                <span className="min-w-0 flex-1">
                  <span className="block text-sm">{t("voicePanel.thinkFirst")}</span>
                  <span className="block text-xs leading-snug text-muted-foreground">
                    {t("voicePanel.thinkFirstHint")}
                  </span>
                </span>
                <Switch checked={voiceThinking} onCheckedChange={setVoiceThinking} />
              </label>
            )}
            {!started && (
              <div className="mt-1 flex items-center gap-3 border-t border-border/60 pb-1 pl-2.5 pr-1 pt-2">
                <p className="min-w-0 flex-1 text-xs leading-snug text-muted-foreground">
                  {t("voicePanel.startHint")}
                </p>
                <Button type="button" variant="primary" size="sm" onClick={start}>
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
