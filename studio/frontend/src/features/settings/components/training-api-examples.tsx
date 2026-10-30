// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import {
  type TrainingApiRequest,
  useSettingsDialogStore,
} from "../stores/settings-dialog-store";
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
  EXAMPLE_TRAINING_API_REQUEST,
  buildTrainingApiSnippet,
  buildTrainingCancelSnippet,
} from "./training-api-snippets";

const LANGS: { id: AudioApiLang; label: string }[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
];

export function TrainingApiExamples({
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
  const [lang, setLang] = useState<AudioApiLang>(
    langFromStored(storedPrefs.apiExampleLang),
  );
  const [request, setRequest] = useState<TrainingApiRequest | null>(null);
  const trainingApiRequested = useSettingsDialogStore(
    (s) => s.trainingApiRequested,
  );

  useEffect(() => {
    if (!trainingApiRequested) return;
    setRequest(trainingApiRequested);
    useSettingsDialogStore.getState().consumeTrainingApiRequest();
  }, [trainingApiRequested]);

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
  const shown = request ?? EXAMPLE_TRAINING_API_REQUEST;
  const snippet = useMemo(
    () => buildTrainingApiSnippet(shown, { base, apiKey: key, lang, os }),
    [base, key, lang, os, shown],
  );
  const cancelSnippet = useMemo(
    () => buildTrainingCancelSnippet({ base, apiKey: key, lang, os }),
    [base, key, lang, os],
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

  const notes = [
    request == null || request.model == null
      ? t("settings.apiKeys.trainingApi.placeholder")
      : null,
    request?.datasetFromHub ? t("settings.apiKeys.trainingApi.hubDataset") : null,
    t("settings.apiKeys.trainingApi.batching"),
    t("settings.apiKeys.trainingApi.chatNote"),
  ].filter((note): note is string => note != null);

  return (
    <section
      data-settings-label={t("settings.apiKeys.trainingApi.title")}
      className="flex min-w-0 max-w-full scroll-mt-6 flex-col gap-4"
    >
      <div className="flex flex-col gap-1">
        {showHeading && (
          <h2 className="settings-heading flex items-center gap-2 text-base font-semibold font-heading">
            <HugeiconsIcon icon={TestTubeOutlineIcon} strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground" />
            {t("settings.apiKeys.trainingApi.title")}
          </h2>
        )}
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.apiKeys.trainingApi.description")}
        </p>
      </div>
      <div className="flex min-w-0 flex-col gap-1 text-ui-11 leading-relaxed text-muted-foreground">
        {notes.map((note) => (
          <p key={note}>{note}</p>
        ))}
      </div>
      <ExampleCodeBox
        code={snippet}
        language={shikiLang}
        label={t("settings.apiKeys.trainingApi.title")}
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
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-medium text-foreground">
          {t("settings.apiKeys.trainingApi.cancelTitle")}
        </h3>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("settings.apiKeys.trainingApi.cancelDescription")}
        </p>
      </div>
      <ExampleCodeBox
        code={cancelSnippet}
        language={shikiLang}
        label={t("settings.apiKeys.trainingApi.cancelTitle")}
        redactFromReload={Boolean(apiKey)}
        pills={null}
        os={os}
        onOsChange={setOs}
      />
    </section>
  );
}
