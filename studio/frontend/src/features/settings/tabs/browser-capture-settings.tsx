// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  type CaptureKind,
  type CaptureRule,
  normalizeCaptureOrigin,
  useCaptureSitesStore,
} from "@/features/browser/capture-sites-store";
import { useT } from "@/i18n";
import { Cancel01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

const KINDS: CaptureKind[] = ["camera", "microphone"];

function RuleSelect({ value, label, onChange }: { value: CaptureRule; label: string; onChange: (rule: CaptureRule) => void }) {
  const t = useT();
  return (
    <Select value={value} onValueChange={(rule) => onChange(rule as CaptureRule)}>
      <SelectTrigger className="w-28" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="ask">{t("browser.capture.ask")}</SelectItem>
        <SelectItem value="block">{t("browser.capture.block")}</SelectItem>
      </SelectContent>
    </Select>
  );
}

export function BrowserCaptureSettings() {
  const t = useT();
  const camera = useCaptureSitesStore((state) => state.camera);
  const microphone = useCaptureSitesStore((state) => state.microphone);
  const sites = useCaptureSitesStore((state) => state.sites);
  const { setDefault, setSite } = useCaptureSitesStore.getState();
  const [draft, setDraft] = useState("");
  const [problem, setProblem] = useState<"invalid" | "full" | null>(null);
  const defaults = { camera, microphone };
  const origins = Object.keys(sites).sort();
  const kindLabel = (kind: CaptureKind) => t(kind === "camera" ? "browser.capture.camera" : "browser.capture.microphone");

  const add = () => {
    const origin = normalizeCaptureOrigin(draft);
    if (!origin) {
      setProblem("invalid");
      return;
    }
    if (!setSite(origin, sites[origin] ?? defaults)) {
      setProblem("full");
      return;
    }
    setDraft("");
    setProblem(null);
  };

  return (
    <SettingsSection title={t("browser.capture.title")}>
      {KINDS.map((kind) => (
        <SettingsRow
          key={kind}
          label={kindLabel(kind)}
          description={t(kind === "camera" ? "browser.capture.cameraDescription" : "browser.capture.microphoneDescription")}
        >
          <RuleSelect value={defaults[kind]} label={kindLabel(kind)} onChange={(rule) => setDefault(kind, rule)} />
        </SettingsRow>
      ))}
      <SettingsRow
        label={t("browser.capture.sites")}
        description={t("browser.capture.sitesDescription")}
        below={
          <div className="flex w-full flex-col gap-2">
            {origins.length > 0 ? (
              <ul className="flex w-full flex-col divide-y divide-border/60 rounded-xl border border-border/80">
                {origins.map((origin) => (
                  <li key={origin} className="flex flex-wrap items-center justify-end gap-x-3 gap-y-1.5 py-1.5 pl-3 pr-1.5">
                    <span className="min-w-32 flex-1 truncate text-sm text-foreground" title={origin}>
                      {origin}
                    </span>
                    {KINDS.map((kind) => (
                      <span key={kind} className="flex items-center gap-2 text-xs text-muted-foreground">
                        {kindLabel(kind)}
                        <RuleSelect
                          value={sites[origin][kind]}
                          label={t("browser.capture.siteFor", { kind: kindLabel(kind), host: origin })}
                          onChange={(rule) => setSite(origin, { ...sites[origin], [kind]: rule })}
                        />
                      </span>
                    ))}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8"
                      aria-label={t("browser.capture.siteRemove", { host: origin })}
                      onClick={() => setSite(origin, null)}
                    >
                      <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
                    </Button>
                  </li>
                ))}
              </ul>
            ) : null}
            <form
              className="flex items-center gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                add();
              }}
            >
              <Input
                value={draft}
                aria-invalid={problem !== null}
                aria-label={t("browser.capture.site")}
                placeholder={t("browser.capture.sitePlaceholder")}
                onChange={(event) => {
                  setDraft(event.target.value);
                  setProblem(null);
                }}
                className="h-8 flex-1"
              />
              <Button variant="secondary" type="submit" size="sm" disabled={!draft.trim()}>
                <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
                {t("browser.capture.add")}
              </Button>
            </form>
            {problem ? (
              <span className="text-xs text-destructive">
                {t(problem === "full" ? "browser.capture.sitesFull" : "browser.capture.siteInvalid")}
              </span>
            ) : null}
          </div>
        }
      />
    </SettingsSection>
  );
}
