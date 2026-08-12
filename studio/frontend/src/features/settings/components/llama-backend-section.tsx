// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatBytes } from "@/features/hub";
import { type TranslationKey, useT } from "@/i18n";
import {
  type LlamaBackendOption,
  isLlamaBackend,
  llamaBackendSelectionNeedsApply,
  visibleLlamaBackendOptions,
} from "../api/llama-backend";
import { useLlamaBackendSwitch } from "../hooks/use-llama-backend-switch";
import { backendDisplayName } from "../lib/llama-backend-labels";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const UNSUPPORTED_REASONS: Record<string, TranslationKey> = {
  not_installed: "settings.resources.llamaBackend.unsupported.notInstalled",
  local_link: "settings.resources.llamaBackend.unsupported.localLink",
  source_build: "settings.resources.llamaBackend.unsupported.sourceBuild",
  no_install_dir: "settings.resources.llamaBackend.unsupported.notInstalled",
  unresolved: "settings.resources.llamaBackend.unsupported.unresolved",
};

export function LlamaBackendSection() {
  const t = useT();
  const { status, selected, setSelected, running, apply, loadError } =
    useLlamaBackendSwitch();

  const optionLabel = (option: LlamaBackendOption) => {
    if (option.backend !== "auto") {
      return backendDisplayName(option.backend, t);
    }
    // "Automatic" alone hides the decision; naming what it picks is the
    // difference between an informed choice and a shot in the dark.
    return option.resolvedBackend
      ? t("settings.resources.llamaBackend.autoWith", {
          backend: backendDisplayName(option.resolvedBackend, t),
        })
      : backendDisplayName("auto", t);
  };

  const job = status?.job;
  const envLocked = status?.envBackend != null;
  const current = status?.backendRequest ?? "auto";
  const value = selected ?? current;
  const options = visibleLlamaBackendOptions(status, value);
  const pending = options.find((option) => option.backend === value);
  const dirty = !running && llamaBackendSelectionNeedsApply(status, selected);
  const unsupportedKey =
    status && !status.supported && !running
      ? (UNSUPPORTED_REASONS[status.reason ?? ""] ??
        "settings.resources.llamaBackend.unsupported.unresolved")
      : null;

  return (
    <SettingsSection title={t("settings.resources.llamaBackend.title")}>
      <SettingsRow
        label={t("settings.resources.llamaBackend.label")}
        description={
          status?.backend
            ? t("settings.resources.llamaBackend.runningOn", {
                backend: backendDisplayName(status.backend, t),
              })
            : t("settings.resources.llamaBackend.description")
        }
        hint={t("settings.resources.llamaBackend.hint")}
      >
        <div className="flex items-center gap-2">
          <Select
            value={value}
            disabled={!status?.supported || envLocked || running}
            onValueChange={(next) => {
              if (isLlamaBackend(next)) {
                setSelected(next);
              }
            }}
          >
            <SelectTrigger
              aria-label={t("settings.resources.llamaBackend.label")}
              className="w-52"
              size="sm"
              data-testid="llama-backend-select"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option.backend} value={option.backend}>
                  {optionLabel(option)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            size="sm"
            onClick={apply}
            disabled={!dirty}
            data-testid="llama-backend-apply"
          >
            {running
              ? t("settings.resources.llamaBackend.applying")
              : t("settings.resources.llamaBackend.apply")}
          </Button>
        </div>
      </SettingsRow>

      {running ? (
        <div className="pb-3" data-testid="llama-backend-progress">
          <Progress
            value={Math.round((job?.progress ?? 0) * 100)}
            aria-label={t("settings.resources.llamaBackend.applying")}
            className="h-1.5 w-full rounded-full bg-muted"
          />
          <p className="mt-2 text-xs text-muted-foreground">
            {job?.message || t("settings.resources.llamaBackend.applying")}
          </p>
        </div>
      ) : null}

      {dirty ? (
        // The two things a switch costs, said before it is paid: a download, and
        // the loaded model going away with the server it runs in.
        <p className="pb-3 text-xs text-muted-foreground">
          {pending?.downloadSizeBytes
            ? t("settings.resources.llamaBackend.applyHintWithSize", {
                size: formatBytes(pending.downloadSizeBytes),
              })
            : t("settings.resources.llamaBackend.applyHint")}
        </p>
      ) : null}

      {envLocked ? (
        <p className="pb-3 text-xs text-amber-600 dark:text-amber-400">
          {t("settings.resources.llamaBackend.envLocked", {
            backend: backendDisplayName(status?.envBackend, t),
          })}
        </p>
      ) : null}

      {unsupportedKey ? (
        <p className="pb-3 text-xs text-muted-foreground">
          {t(unsupportedKey)}
        </p>
      ) : null}

      {loadError ? (
        <p className="pb-3 text-xs text-destructive">{loadError}</p>
      ) : null}
    </SettingsSection>
  );
}
