// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
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
import { Pill } from "./audio-api-examples";
import {
  type AudioApiLang,
  type AudioApiOs,
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
import { HighlightedCode } from "./usage-examples";

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
  const pillId = useId();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const setStoredOs = useSettingsPanelPrefsStore((s) => s.setApiExampleOs);
  const [storedPrefs] = useState(() => useSettingsPanelPrefsStore.getState());
  const [kind, setKind] = useState<MediaApiKind>("images");
  const [lang, setLang] = useState<AudioApiLang>(
    langFromStored(storedPrefs.apiExampleLang),
  );
  const [os, setOs] = useState<AudioApiOs>(
    storedPrefs.apiExampleOs ?? (deviceType === "windows" ? "windows" : "unix"),
  );
  const [pageModels, setPageModels] = useState<
    Partial<Record<MediaApiKind, string>>
  >({});
  const [autoSwitch, setAutoSwitch] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
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
    useTunnel && cloudflareUrl ? cloudflareUrl : (serverUrl ?? origin);
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

  const handleCopy = async () => {
    if (await copyToClipboard(snippet)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <section
      ref={sectionRef}
      data-settings-label={t("settings.apiKeys.mediaApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col"
    >
      <h2 className="settings-heading mb-1 text-sm font-semibold">
        {t("settings.apiKeys.mediaApi.title")}
      </h2>
      <p className="mb-2 text-xs leading-relaxed text-muted-foreground">
        {t("settings.apiKeys.mediaApi.description")}
      </p>
      <div className="min-w-0 max-w-full overflow-hidden rounded-lg border border-border bg-muted/20">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-border px-2 py-1.5">
          <div className="isolate flex min-w-0 flex-wrap items-center gap-0.5">
            {MEDIA_API_KINDS.map((id) => (
              <Pill
                key={id}
                active={kind === id}
                pillGroup={`${pillId}-kind`}
                onClick={() => setKind(id)}
              >
                {t(
                  id === "images"
                    ? "settings.apiKeys.mediaApi.images"
                    : "settings.apiKeys.mediaApi.video",
                )}
              </Pill>
            ))}
          </div>
          <div className="isolate flex min-w-0 flex-wrap items-center gap-0.5">
            {LANGS.map((item) => (
              <Pill
                key={item.id}
                active={lang === item.id}
                pillGroup={`${pillId}-lang`}
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
        {lang === "curl" ? (
          <div className="isolate flex min-w-0 items-center gap-0.5 border-b border-border px-2 py-1.5">
            <Pill
              active={os === "unix"}
              pillGroup={`${pillId}-os`}
              onClick={() => {
                setOs("unix");
                setStoredOs("unix");
              }}
            >
              {t("settings.apiKeys.osUnix")}
            </Pill>
            <Pill
              active={os === "windows"}
              pillGroup={`${pillId}-os`}
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
            label={t("settings.apiKeys.mediaApi.title")}
            redactFromReload={Boolean(apiKey)}
          />
        </div>
        {pageModel == null || autoSwitch === false ? (
          <div className="flex min-w-0 flex-col gap-1 border-t border-border px-3 py-2.5 text-ui-11 leading-snug text-muted-foreground">
            {pageModel == null ? (
              <span>{t("settings.apiKeys.mediaApi.placeholderModel")}</span>
            ) : null}
            {autoSwitch === false ? (
              <span>{t("settings.apiKeys.mediaApi.autoSwitchOff")}</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
