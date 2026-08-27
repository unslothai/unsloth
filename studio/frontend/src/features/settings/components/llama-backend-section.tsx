// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatBytes } from "@/features/hub";
import {
  FolderBrowser,
  invalidateLlamaFlagCatalog,
} from "@/features/model-picker";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import {
  type LlamaBackendOption,
  isLlamaBackend,
  llamaBackendSelectionNeedsApply,
  visibleLlamaBackendOptions,
} from "../api/llama-backend";
import {
  type LlamaCppPathSettings,
  loadLlamaCppPathSettings,
  updateLlamaCppPathSettings,
} from "../api/llama-cpp-path";
import { useLlamaBackendSwitch } from "../hooks/use-llama-backend-switch";
import { backendDisplayName } from "../lib/llama-backend-labels";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const UNSUPPORTED_REASONS: Record<string, TranslationKey> = {
  not_installed: "settings.resources.llamaBackend.unsupported.notInstalled",
  local_link: "settings.resources.llamaBackend.unsupported.localLink",
  source_build: "settings.resources.llamaBackend.unsupported.sourceBuild",
  no_install_dir: "settings.resources.llamaBackend.unsupported.notInstalled",
  custom_path: "settings.resources.llamaBackend.unsupported.customPath",
  unresolved: "settings.resources.llamaBackend.unsupported.unresolved",
};

function LlamaCppPathRow({ onChanged }: { onChanged: () => void }) {
  const t = useT();
  const [settings, setSettings] = useState<LlamaCppPathSettings | null>(null);
  const [draftPath, setDraftPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void loadLlamaCppPathSettings()
      .then((next) => {
        if (!active) return;
        setSettings(next);
        setDraftPath(next.path ?? "");
        setError(null);
      })
      .catch((reason) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, []);

  const save = async (path: string | null) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateLlamaCppPathSettings(path);
      setSettings(next);
      setDraftPath(next.path ?? "");
      setBrowserOpen(false);
      toast.success(t("settings.resources.llamaBackend.customPath.saved"));
      onChanged();
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      toast.error(t("settings.resources.llamaBackend.customPath.saveError"), {
        description: message,
      });
    } finally {
      setSaving(false);
    }
  };

  const pathDirty = Boolean(
    settings?.editable && draftPath.trim() !== (settings.path ?? ""),
  );
  const detail = settings
    ? settings.source === "environment"
      ? t("settings.resources.llamaBackend.customPath.environmentManaged", {
          variable: settings.environmentVariable ?? "UNSLOTH_LLAMA_CPP_PATH",
        })
      : settings.available
        ? settings.reloadRequired
          ? t("settings.resources.llamaBackend.customPath.reloadRequired")
          : settings.source === "studio"
            ? t("settings.resources.llamaBackend.customPath.active")
            : t("settings.resources.llamaBackend.customPath.bundled")
        : t("settings.resources.llamaBackend.customPath.missingBinary")
    : null;

  return (
    <>
      <SettingsRow
        label={t("settings.resources.llamaBackend.customPath.label")}
        description={t(
          "settings.resources.llamaBackend.customPath.description",
        )}
        hint={t("settings.resources.llamaBackend.customPath.hint")}
        className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
      >
        <div className="grid w-[392px] min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 gap-y-1.5 max-[840px]:w-full">
          <Input
            readOnly={!settings?.editable}
            aria-label={t("settings.resources.llamaBackend.customPath.label")}
            value={draftPath}
            placeholder={
              settings
                ? t("settings.resources.llamaBackend.customPath.automatic")
                : t("common.loading")
            }
            onChange={(event) => setDraftPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && pathDirty && !saving) {
                void save(draftPath.trim() || null);
              }
            }}
            title={settings?.resolvedBinary ?? settings?.path ?? undefined}
            className="h-8 min-w-0 font-mono text-xs"
            data-testid="llama-cpp-path-input"
          />
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            disabled={!pathDirty || saving}
            onClick={() => void save(draftPath.trim() || null)}
            data-testid="llama-cpp-path-save"
          >
            {saving ? t("common.saving") : t("common.save")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            disabled={!settings?.editable || saving}
            onClick={() => setBrowserOpen(true)}
            data-testid="llama-cpp-path-change"
          >
            {saving
              ? t("settings.resources.llamaBackend.customPath.saving")
              : t("settings.resources.llamaBackend.customPath.change")}
          </Button>
          {detail || settings?.source === "studio" ? (
            <div className="col-span-3 flex min-w-0 items-center justify-between gap-2 pl-3.5 pr-1 text-xs text-muted-foreground">
              {detail ? (
                <span title={detail} className="min-w-0 truncate">
                  {detail}
                </span>
              ) : null}
              {settings?.source === "studio" ? (
                <Button
                  variant="link"
                  size="xs"
                  className="h-auto shrink-0 px-0 text-xs"
                  disabled={saving}
                  onClick={() => void save(null)}
                  data-testid="llama-cpp-path-reset"
                >
                  {t("settings.resources.llamaBackend.customPath.useBundled")}
                </Button>
              ) : null}
            </div>
          ) : null}
        </div>
      </SettingsRow>

      {error ? (
        <p
          className="pb-3 text-xs text-destructive"
          data-testid="llama-cpp-path-error"
        >
          {error}
        </p>
      ) : null}

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => void save(path)}
        initialPath={settings?.path ?? undefined}
        title={t("settings.resources.llamaBackend.customPath.chooseTitle")}
        description={t(
          "settings.resources.llamaBackend.customPath.description",
        )}
        confirmLabel={t(
          "settings.resources.llamaBackend.customPath.chooseAction",
        )}
        showModelHints={false}
      />
    </>
  );
}

export function LlamaBackendSection() {
  const t = useT();
  const { status, selected, setSelected, running, apply, refresh, loadError } =
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
  // Null means the marker holds a choice a newer Unsloth wrote. Show it as
  // unknown rather than as Automatic, and let it be replaced deliberately.
  const unknownRecorded = status?.backendRequest === null;
  const value = selected ?? status?.backendRequest ?? "unknown";
  const options = visibleLlamaBackendOptions(
    status,
    selected ?? status?.backendRequest ?? null,
  );
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
              {unknownRecorded ? (
                <SelectItem value="unknown">
                  {t("settings.resources.environment.unknown")}
                </SelectItem>
              ) : null}
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
            // The same hard blockers as the Select above, not just dirtiness. An
            // environment pin with an automatic choice that has since drifted leaves
            // selectionApplied false, so the row is dirty while the Select is disabled:
            // Apply would then be the only live control, and the server rightly refuses
            // it with environment_override. Do not offer an action that cannot succeed.
            disabled={!dirty || !status?.supported || envLocked}
            data-testid="llama-backend-apply"
          >
            {running
              ? t("settings.resources.llamaBackend.applying")
              : t("settings.resources.llamaBackend.apply")}
          </Button>
        </div>
      </SettingsRow>

      <LlamaCppPathRow
        onChanged={() => {
          invalidateLlamaFlagCatalog();
          void refresh(true);
        }}
      />

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
