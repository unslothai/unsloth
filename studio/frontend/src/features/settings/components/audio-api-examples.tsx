// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import {
  loadOpenAIAutoSwitchSettings,
  subscribeOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import { watchAudioModels } from "../api/openai-model-catalog";
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
import {
  ExampleCodeBox,
  ExampleField,
  ExampleToggle,
  useExampleOs,
} from "./api-example-parts";

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

export function AudioApiExamples({
  apiKey,
  useTunnel,
  keylessScope = "off",
  keylessExposure = null,
  showHeading = true,
}: {
  apiKey?: string | null;
  showHeading?: boolean;
  useTunnel: boolean;
  keylessScope?: KeylessApiAccessScope;
  keylessExposure?: KeylessApiAccessExposure | null;
}) {
  const t = useT();
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const setStoredTab = useSettingsPanelPrefsStore((s) => s.setApiAudioExample);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [tab, setTab] = useState<AudioApiTab>(
    AUDIO_API_TABS.find((id) => id === storedPrefs.apiAudioExample) ?? "speak",
  );
  const [run, setRun] = useState<AudioRunWorkflow>("separate");
  const [lang, setLang] = useState<AudioApiLang>(
    langFromStored(storedPrefs.apiExampleLang),
  );
  // null until /v1/models answers, so a slow or failed listing never claims nothing is downloaded.
  const [models, setModels] = useState<AudioApiModel[] | null>(null);
  const [autoSwitch, setAutoSwitch] = useState<boolean | null>(null);
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
    // The API page's copy (no heading) must not answer, or the dialog's never scrolls.
    if (!showHeading || scrollTarget !== "api-keys-audio-api") return;
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
  }, [scrollTarget, showHeading]);

  useEffect(() => watchAudioModels(listOpenAIAudioModels, setModels), []);

  useEffect(() => {
    let cancelled = false;
    loadOpenAIAutoSwitchSettings().then(
      (settings) => !cancelled && setAutoSwitch(settings.enabled),
      () => undefined,
    );
    // The Model auto-switch switch sits in this same tab.
    const unsubscribe = subscribeOpenAIAutoSwitchSettings((settings) =>
      setAutoSwitch(settings.enabled),
    );
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const base =
    useTunnel && cloudflareUrl
      ? cloudflareUrl
      : isTauri
        ? (serverUrl ?? origin)
        : origin;
  const [os, setOs] = useExampleOs(base);
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
    // Transcriptions load their model on request whatever auto-switch says.
    example !== "transcribe" &&
    autoSwitch === false &&
    models !== null &&
    !models.some((m) => m.id === model && m.loaded);
  const shikiLang =
    lang === "curl" ? (os === "windows" ? "powershell" : "bash") : lang;

  const pickExample = (next: string) => {
    const workflow = AUDIO_RUN_WORKFLOWS.find((id) => id === next);
    if (workflow) {
      setRun(workflow);
      setTab("workflows");
      setStoredTab("workflows");
      return;
    }
    const nextTab = AUDIO_API_TABS.find((id) => id === next);
    if (nextTab) {
      setTab(nextTab);
      setStoredTab(nextTab);
    }
  };

  const pickLang = (next: AudioApiLang) => {
    setLang(next);
    const stored = audioLangToStore(
      useSettingsPanelPrefsStore.getState().apiExampleLang,
      next,
    );
    if (stored) setStoredLang(stored);
  };

  return (
    <section
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.audioApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        {showHeading && (
          <h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">
            <HugeiconsIcon icon={AiSpeechIcon} strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground" />
            {t("settings.apiKeys.audioApi.title")}
          </h2>
        )}
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.apiKeys.audioApi.description")}
        </p>
      </div>
      <div className="@container">
        <div className="grid grid-cols-1 items-start gap-3 @[34rem]:grid-cols-3">
          <ExampleField label={t("settings.apiKeys.exampleKind")}>
            {/* One picker for the tabs and the workflows that used to sit in a second row. */}
            <Select value={example} onValueChange={pickExample}>
              <SelectTrigger
                aria-label={t("settings.apiKeys.exampleKind")}
                className="w-full rounded-lg"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="start">
                {/* Grouped too, so these line up with the workflow items. */}
                <SelectGroup>
                  {AUDIO_API_TABS.filter((id) => id !== "workflows").map(
                    (id) => (
                      <SelectItem key={id} value={id}>
                        {t(TAB_LABEL[id])}
                      </SelectItem>
                    ),
                  )}
                </SelectGroup>
                <SelectGroup>
                  <SelectLabel>{t(TAB_LABEL.workflows)}</SelectLabel>
                  {AUDIO_RUN_WORKFLOWS.map((id) => (
                    <SelectItem key={id} value={id}>
                      {t(RUN_LABEL[id])}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </ExampleField>
        </div>
      </div>
      {placeholder || needsLoad ? (
        <div className="flex min-w-0 flex-col gap-1 text-ui-11 leading-relaxed text-muted-foreground">
          {placeholder ? (
            <p>{t("settings.apiKeys.audioApi.placeholderModel")}</p>
          ) : null}
          {needsLoad ? (
            <p>{t("settings.apiKeys.audioApi.autoSwitchOff")}</p>
          ) : null}
        </div>
      ) : null}
      <ExampleCodeBox
        code={snippet}
        language={shikiLang}
        label={t("settings.apiKeys.audioApi.title")}
        redactFromReload={Boolean(apiKey)}
        pills={
          <ExampleToggle
            label={t("settings.apiKeys.exampleLanguage")}
            value={lang}
            options={LANGS}
            onChange={pickLang}
          />
        }
        os={os}
        onOsChange={setOs}
      />
    </section>
  );
}
