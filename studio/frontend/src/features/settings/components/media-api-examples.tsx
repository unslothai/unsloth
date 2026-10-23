// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import {
  loadOpenAIAutoSwitchSettings,
  subscribeOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import {
  ExampleCodeBox,
  ExampleField,
  PlatformToggle,
  useExampleOs,
} from "./api-example-parts";
import {
  type AudioApiLang,
  audioApiKey,
  audioLangToStore,
  langFromStored,
} from "./audio-api-snippets";
import {
  MEDIA_API_KINDS,
  MEDIA_API_PLACEHOLDER_MODELS,
  type MediaApiKind,
  buildMediaApiSnippet,
} from "./media-api-snippets";

const LANGS: { id: AudioApiLang; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
];

export function MediaApiExamples({
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
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [kind, setKind] = useState<MediaApiKind>("images");
  const [lang, setLang] = useState<AudioApiLang>(
    langFromStored(storedPrefs.apiExampleLang),
  );
  const [pageModels, setPageModels] = useState<
    Partial<Record<MediaApiKind, string>>
  >({});
  const [autoSwitch, setAutoSwitch] = useState<boolean | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);
  const mediaApiRequested = useSettingsDialogStore((s) => s.mediaApiRequested);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);

  useEffect(() => {
    if (!mediaApiRequested) return;
    const { kind: requestedKind, model } = mediaApiRequested;
    setKind(requestedKind);
    if (model) {
      setPageModels((current) => ({ ...current, [requestedKind]: model }));
    }
    useSettingsDialogStore.getState().consumeMediaApiRequest();
  }, [mediaApiRequested]);

  useEffect(() => {
    if (scrollTarget !== "api-keys-media-api") return;
    const section = sectionRef.current;
    if (!section) return;
    const pin = () => section.scrollIntoView({ block: "start" });
    const observer = new ResizeObserver(pin);
    observer.observe(section.parentElement ?? section);
    const release = () => {
      observer.disconnect();
      useSettingsDialogStore
        .getState()
        .consumeScrollTarget("api-keys-media-api");
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
    loadOpenAIAutoSwitchSettings().then(
      (settings) => !cancelled && setAutoSwitch(settings.mediaAutoSwitchModel),
      () => undefined,
    );
    const unsubscribe = subscribeOpenAIAutoSwitchSettings((settings) =>
      setAutoSwitch(settings.mediaAutoSwitchModel),
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
  const pageModel = pageModels[kind] ?? null;
  const model = pageModel ?? MEDIA_API_PLACEHOLDER_MODELS[kind];
  const snippet = useMemo(
    () => buildMediaApiSnippet(kind, { base, apiKey: key, model, lang, os }),
    [base, key, kind, lang, model, os],
  );
  const shikiLang =
    lang === "curl" ? (os === "windows" ? "powershell" : "bash") : lang;

  return (
    <section
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.mediaApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        <h2 className="settings-heading text-sm font-semibold">
          {t("settings.apiKeys.mediaApi.title")}
        </h2>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.apiKeys.mediaApi.description")}
        </p>
      </div>
      <div className="@container">
        <div className="grid grid-cols-1 items-start gap-3 @[34rem]:grid-cols-3">
          <ExampleField label={t("settings.apiKeys.exampleKind")}>
            <Select
              value={kind}
              onValueChange={(next) => {
                const picked = MEDIA_API_KINDS.find((id) => id === next);
                if (picked) setKind(picked);
              }}
            >
              <SelectTrigger
                aria-label={t("settings.apiKeys.exampleKind")}
                className="w-full rounded-lg"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="start">
                {MEDIA_API_KINDS.map((id) => (
                  <SelectItem key={id} value={id}>
                    {t(
                      id === "images"
                        ? "settings.apiKeys.mediaApi.images"
                        : "settings.apiKeys.mediaApi.video",
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ExampleField>
          <ExampleField label={t("settings.apiKeys.exampleLanguage")}>
            <Select
              value={lang}
              onValueChange={(next) => {
                const picked = LANGS.find((item) => item.id === next);
                if (!picked) return;
                setLang(picked.id);
                const stored = audioLangToStore(
                  useSettingsPanelPrefsStore.getState().apiExampleLang,
                  picked.id,
                );
                if (stored) setStoredLang(stored);
              }}
            >
              <SelectTrigger
                aria-label={t("settings.apiKeys.exampleLanguage")}
                className="w-full rounded-lg"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="start">
                {LANGS.map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </ExampleField>
          <ExampleField label={t("settings.apiKeys.examplePlatform")}>
            <PlatformToggle value={os} onChange={setOs} />
          </ExampleField>
        </div>
      </div>
      {pageModel == null || autoSwitch === false ? (
        <div className="flex min-w-0 flex-col gap-1 text-ui-11 leading-relaxed text-muted-foreground">
          {pageModel == null ? (
            <p>{t("settings.apiKeys.mediaApi.placeholderModel")}</p>
          ) : null}
          {autoSwitch === false ? (
            <p>{t("settings.apiKeys.mediaApi.autoSwitchOff")}</p>
          ) : null}
        </div>
      ) : null}
      <ExampleCodeBox
        code={snippet}
        language={shikiLang}
        label={t("settings.apiKeys.mediaApi.title")}
        redactFromReload={Boolean(apiKey)}
      />
    </section>
  );
}
