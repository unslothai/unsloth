// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { Image01Icon } from "@hugeicons/core-free-icons";
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
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { MediaAutoSwitchRow } from "./model-auto-switch-section";
import { useSettingsPanelPrefsStore } from "../stores/settings-panel-prefs-store";
import {
  ExampleCodeBox,
  ExampleToggle,
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
    // The API page's copy (no heading) must not answer, or the dialog's never scrolls.
    if (!showHeading || scrollTarget !== "api-keys-media-api") return;
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
  }, [scrollTarget, showHeading]);

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

  const pickLang = (next: AudioApiLang) => {
    setLang(next);
    const stored = audioLangToStore(
      useSettingsPanelPrefsStore.getState().apiExampleLang,
      next,
    );
    if (stored) setStoredLang(stored);
  };

  const isOwner = useIsAccountOwner();
  return (
    <section
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.mediaApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        {showHeading && (
          <h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">
            <HugeiconsIcon icon={Image01Icon} strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground" />
            {t("settings.apiKeys.mediaApi.title")}
          </h2>
        )}
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.apiKeys.mediaApi.description")}
        </p>
      </div>
      {isOwner ? (
        <div className="-my-3">
          <MediaAutoSwitchRow />
        </div>
      ) : null}
      <ExampleToggle
        label={t("settings.apiKeys.exampleKind")}
        variant="toggle"
        value={kind}
        options={MEDIA_API_KINDS.map((id) => ({
          id,
          label: t(
            id === "images"
              ? "settings.apiKeys.mediaApi.images"
              : "settings.apiKeys.mediaApi.video",
          ),
        }))}
        onChange={setKind}
      />
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
