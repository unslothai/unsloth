// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ExampleCodeBox,
  type ExampleOs,
  ExampleToggle,
  useExampleOs,
} from "@/features/settings/components/api-example-parts";
import { useSettingsPanelPrefsStore } from "@/features/settings/stores/settings-panel-prefs-store";
import { useState } from "react";
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
  const [os, setOs] = useExampleOs(baseUrl);
  const setStoredLang = useSettingsPanelPrefsStore((s) => s.setApiExampleLang);
  const [language, setPicked] = useState<ApiExampleLanguage>(() =>
    apiExampleLanguage(
      useSettingsPanelPrefsStore.getState().apiExampleLang,
      fallback,
    ),
  );
  const setLanguage = (next: ApiExampleLanguage) => {
    setPicked(next);
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
