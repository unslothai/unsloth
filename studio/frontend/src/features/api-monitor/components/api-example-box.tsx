// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ExampleCodeBox,
  type ExampleOs,
  ExampleToggle,
  useExampleOs,
} from "@/features/settings/components/api-example-parts";
import { useSettingsPanelPrefsStore } from "@/features/settings/stores/settings-panel-prefs-store";
import {
  API_EXAMPLE_LANGUAGES,
  type ApiExampleLanguage,
  LANGUAGE_PREFIX_RE,
  apiExampleLanguage,
  apiExampleShikiLanguage,
} from "../api-sampling-example";

export interface ApiExampleChoices {
  language: ApiExampleLanguage;
  setLanguage: (language: ApiExampleLanguage) => void;
  os: ExampleOs;
  setOs: (os: ExampleOs) => void;
}

/** Language and platform, shared with the API settings tab's examples. */
export function useApiExampleChoices(
  baseUrl: string,
  fallback: ApiExampleLanguage,
): ApiExampleChoices {
  // Read from the store on every render, not copied at mount: the API page keeps the quick start and the
  // (closed) example dialog mounted together, and a pick in one must show in the other.
  const [detectedOs, setOs] = useExampleOs(baseUrl);
  const storedOs = useSettingsPanelPrefsStore((s) => s.apiExampleOs);
  const os = storedOs ?? detectedOs;
  const storedLang = useSettingsPanelPrefsStore((s) => s.apiExampleLang);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const language = apiExampleLanguage(storedLang, fallback);
  const setLanguage = (next: ApiExampleLanguage) => {
    // Swap only the language, so the settings tab keeps its request kind (e.g. "pythonTools").
    const stored = useSettingsPanelPrefsStore.getState().apiExampleLang;
    setStoredLang(
      stored && LANGUAGE_PREFIX_RE.test(stored)
        ? stored.replace(LANGUAGE_PREFIX_RE, next)
        : next,
    );
  };
  return { language, setLanguage, os, setOs };
}

export function ApiExampleBox({
  choices,
  code,
  fitContent,
}: {
  choices: ApiExampleChoices;
  code: string;
  fitContent?: boolean;
}) {
  return (
    <ExampleCodeBox
      code={code}
      language={apiExampleShikiLanguage(choices.language, choices.os)}
      label="Example request"
      redactFromReload={false}
      pills={
        <ExampleToggle
          label="Language"
          value={choices.language}
          options={API_EXAMPLE_LANGUAGES}
          onChange={choices.setLanguage}
        />
      }
      os={choices.os}
      onOsChange={choices.setOs}
      fitContent={fitContent}
    />
  );
}
