// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { loadOpenAIAutoSwitchSettings } from "../api/openai-auto-switch";
import {
  type AudioApiModel,
  listOpenAIAudioModels,
} from "../api/openai-models";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import {
  AUDIO_API_PLACEHOLDER_MODELS,
  AUDIO_API_TABS,
  AUDIO_RUN_WORKFLOWS,
  type AudioApiExample,
  type AudioApiLang,
  type AudioApiOs,
  type AudioApiTab,
  type AudioRunWorkflow,
  audioApiExampleFor,
  audioApiKey,
  audioApiModelFits,
  audioLangToStore,
  buildAudioApiSnippet,
  langFromStored,
  pickAudioApiModel,
} from "./audio-api-snippets";
import { HighlightedCode } from "./usage-examples";

const LANGS: { id: AudioApiLang; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
];

const TAB_LABEL: Record<AudioApiTab, TranslationKey> = {
  speak: "settings.apiKeys.audioApi.speak",
  clone: "settings.apiKeys.audioApi.clone",
  transcribe: "settings.apiKeys.audioApi.transcribe",
  workflows: "settings.apiKeys.audioApi.workflows",
};

const RUN_LABEL: Record<AudioRunWorkflow, TranslationKey> = {
  separate: "settings.apiKeys.audioApi.separate",
  convert: "settings.apiKeys.audioApi.convert",
  music: "settings.apiKeys.audioApi.music",
  edit: "settings.apiKeys.audioApi.edit",
};

function Pill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "rounded-full px-2.5 py-1 text-ui-11 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        active
          ? "hub-tab-toggle-pill text-foreground"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

export function AudioApiExamples({
  apiKey,
  useTunnel,
  keylessScope = "off",
  keylessExposure = null,
}: {
  apiKey?: string | null;
  useTunnel: boolean;
  keylessScope?: KeylessApiAccessScope;
  keylessExposure?: KeylessApiAccessExposure | null;
}) {
  const t = useT();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const setStoredTab = useSettingsPanelPrefsStore((s) => s.setApiAudioExample);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setApiExampleOs);
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [tab, setTab] = useState<AudioApiTab>(
    AUDIO_API_TABS.find((id) => id === storedPrefs.apiAudioExample) ?? "speak",
  );
  const [run, setRun] = useState<AudioRunWorkflow>("separate");
  const [lang, setLang] = useState<AudioApiLang>(
    langFromStored(storedPrefs.apiExampleLang),
  );
  const [os, setOs] = useState<AudioApiOs>(
    storedPrefs.apiExampleOs ?? (deviceType === "windows" ? "windows" : "unix"),
  );
  // null until /v1/models answers, so a slow or failed listing never claims nothing is downloaded.
  const [models, setModels] = useState<AudioApiModel[] | null>(null);
  const [autoSwitch, setAutoSwitch] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [pageModel, setPageModel] = useState<{
    example: AudioApiExample;
    model: string;
  } | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);
  const audioApiRequested = useSettingsDialogStore((s) => s.audioApiRequested);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);

  useEffect(() => {
    if (!audioApiRequested) return;
    const next = audioApiExampleFor(audioApiRequested.workflow);
    setTab(next.tab);
    if (next.run) setRun(next.run);
    setPageModel(
      audioApiRequested.model
        ? {
            example:
              next.tab === "workflows" ? (next.run ?? "separate") : next.tab,
            model: audioApiRequested.model,
          }
        : null,
    );
    useSettingsDialogStore.getState().consumeAudioApiRequest();
  }, [audioApiRequested]);

  useEffect(() => {
    if (scrollTarget !== "api-keys-audio-api") return;
    const section = sectionRef.current;
    if (!section) return;
    // Consumed only after the user scrolls: clearing it re-runs this effect, whose cleanup stops the pinning.
    const pin = () => section.scrollIntoView({ block: "start" });
    const observer = new ResizeObserver(pin);
    observer.observe(section.parentElement ?? section);
    const release = () => {
      observer.disconnect();
      useSettingsDialogStore
        .getState()
        .consumeScrollTarget("api-keys-audio-api");
    };
    const timer = window.setTimeout(release, 1500);
    const events = ["wheel", "touchmove", "keydown"] as const;
    for (const name of events) {
      window.addEventListener(name, release, { passive: true });
    }
    return () => {
      observer.disconnect();
      window.clearTimeout(timer);
      for (const name of events) window.removeEventListener(name, release);
    };
  }, [scrollTarget]);

  useEffect(() => {
    let cancelled = false;
    listOpenAIAudioModels().then(
      (listed) => !cancelled && setModels(listed),
      () => undefined,
    );
    loadOpenAIAutoSwitchSettings().then(
      (settings) => !cancelled && setAutoSwitch(settings.enabled),
      () => undefined,
    );
    return () => {
      cancelled = true;
    };
  }, []);

  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl ? cloudflareUrl : (serverUrl ?? origin);
  const key = audioApiKey(apiKey, {
    base,
    tunnel: useTunnel && !!cloudflareUrl,
    scope: keylessScope,
    exposure: keylessExposure,
  });
  const example: AudioApiExample = tab === "workflows" ? run : tab;
  const picked =
    pageModel?.example === example &&
    audioApiModelFits(models, pageModel.model, example)
      ? pageModel.model
      : models
        ? pickAudioApiModel(models, example)
        : null;
  const model = picked ?? AUDIO_API_PLACEHOLDER_MODELS[example];
  const snippet = useMemo(
    () =>
      buildAudioApiSnippet(example, {
        base,
        apiKey: key,
        model,
        lang,
        os,
      }),
    [base, example, key, lang, model, os],
  );
  const placeholder = models !== null && !picked;
  const needsLoad =
    !placeholder &&
    autoSwitch === false &&
    models !== null &&
    !models.some((m) => m.id === model && m.loaded);
  const shikiLang =
    lang === "curl" ? (os === "windows" ? "powershell" : "bash") : lang;

  const handleCopy = async () => {
    if (await copyToClipboard(snippet)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <section
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.audioApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col"
    >
      <h2 className="settings-heading mb-1 text-sm font-semibold">
        {t("settings.apiKeys.audioApi.title")}
      </h2>
      <p className="mb-2 text-xs leading-relaxed text-muted-foreground">
        {t("settings.apiKeys.audioApi.description")}
      </p>
      <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-muted/20">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-border px-2 py-1.5">
          <div className="flex min-w-0 flex-wrap items-center gap-0.5">
            {AUDIO_API_TABS.map((id) => (
              <Pill
                key={id}
                active={tab === id}
                onClick={() => {
                  setTab(id);
                  setStoredTab(id);
                }}
              >
                {t(TAB_LABEL[id])}
              </Pill>
            ))}
          </div>
          <div className="flex min-w-0 flex-wrap items-center gap-0.5">
            {LANGS.map((item) => (
              <Pill
                key={item.id}
                active={lang === item.id}
                onClick={() => {
                  setLang(item.id);
                  const stored = audioLangToStore(
                    useSettingsPanelPrefsStore.getState().apiExampleLang,
                    item.id,
                  );
                  if (stored) setStoredLang(stored);
                }}
              >
                {item.label}
              </Pill>
            ))}
          </div>
        </div>
        {tab === "workflows" ? (
          <div className="flex min-w-0 flex-wrap items-center gap-0.5 border-b border-border px-2 py-1.5">
            {AUDIO_RUN_WORKFLOWS.map((id) => (
              <Pill key={id} active={run === id} onClick={() => setRun(id)}>
                {t(RUN_LABEL[id])}
              </Pill>
            ))}
          </div>
        ) : null}
        {lang === "curl" ? (
          <div className="flex min-w-0 items-center gap-0.5 border-b border-border px-2 py-1.5">
            <Pill
              active={os === "unix"}
              onClick={() => {
                setOs("unix");
                setStoredOs("unix");
              }}
            >
              {t("settings.apiKeys.osUnix")}
            </Pill>
            <Pill
              active={os === "windows"}
              onClick={() => {
                setOs("windows");
                setStoredOs("windows");
              }}
            >
              {t("settings.apiKeys.osWindows")}
            </Pill>
          </div>
        ) : null}
        <div className="relative min-w-0">
          <button
            type="button"
            onClick={handleCopy}
            className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded border border-border bg-background/80 px-1.5 py-1 text-ui-11 text-muted-foreground backdrop-blur transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-label={t("settings.apiKeys.copySnippet")}
          >
            <HugeiconsIcon
              icon={copied ? Tick02Icon : Copy01Icon}
              className={cn("size-3.5", copied && "text-emerald-600")}
            />
            {copied ? t("settings.apiKeys.copied") : t("settings.apiKeys.copy")}
          </button>
          <HighlightedCode
            key={snippet}
            code={snippet}
            language={shikiLang}
            redactFromReload={Boolean(apiKey)}
          />
        </div>
        {placeholder || needsLoad ? (
          <div className="flex min-w-0 flex-col gap-1 border-t border-border px-3 py-2.5 text-ui-11 leading-snug text-muted-foreground">
            {placeholder ? (
              <span>{t("settings.apiKeys.audioApi.placeholderModel")}</span>
            ) : null}
            {needsLoad ? (
              <span>{t("settings.apiKeys.audioApi.autoSwitchOff")}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
