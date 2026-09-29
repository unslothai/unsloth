// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState, type ReactElement } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  fetchPillModelOptions,
  fetchPillSettings,
  syncNativePillConfig,
  updatePillSettings,
  withNativeApplyLock,
  type PillModelOption,
  type PillSettings,
} from "@/features/system-pill";
import { pillStatus } from "@/lib/pill-native";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

const DEFAULT_MODEL_VALUE = "__none__";

export function SystemPillTab(): ReactElement {
  const t = useT();
  const [settings, setSettings] = useState<PillSettings | null>(null);
  const [hotkey, setHotkey] = useState("");
  const [models, setModels] = useState<PillModelOption[]>([]);
  // The slow initial load must not overwrite an already saved edit.
  const editedRef = useRef(false);
  // Sequence drops superseded UI writes; the chain serialises each PUT with its native sync.
  const saveSeqRef = useRef(0);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      pillStatus(),
      fetchPillSettings(),
      fetchPillModelOptions(),
    ])
      .then(([status, loaded, loadedModels]) => {
        if (cancelled) return;
        setHotkey(status.hotkey);
        if (!editedRef.current) setSettings(loaded);
        setModels(loadedModels);
      })
      .catch(() => {
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const applySettings = (update: Partial<PillSettings>): Promise<void> => {
    const seq = ++saveSeqRef.current;
    // Shared lock: startup sync writes the same native config.
    saveChainRef.current = saveChainRef.current.then(() =>
      withNativeApplyLock(async () => {
        let saved: PillSettings;
        try {
          saved = await updatePillSettings(update);
        } catch {
          toast.error(t("systemPill.settings.saveError"));
          // A predecessor may have skipped its apply as superseded; the newest save reads back what stuck.
          if (seq !== saveSeqRef.current) return;
          try {
            const actual = await fetchPillSettings();
            if (seq !== saveSeqRef.current) return;
            editedRef.current = true;
            setSettings(actual);
            await syncNativePillConfig(actual);
          } catch {
          }
          return;
        }

        if (seq !== saveSeqRef.current) return;
        editedRef.current = true;
        setSettings(saved);

        try {
          await syncNativePillConfig(saved);
        } catch {
          toast.error(t("systemPill.settings.saveError"));
          // Native refused: align backend and switch with native status (Rust restores the old hotkey on a failed disable).
          if (seq !== saveSeqRef.current) return;
          try {
            const status = await pillStatus();
            if (!status.supported || status.enabled === saved.enabled) return;
            const corrected = await updatePillSettings({
              enabled: status.enabled,
            });
            if (seq !== saveSeqRef.current) return;
            setSettings(corrected);
          } catch {
          }
        }
      }),
    );
    return saveChainRef.current;
  };

  return (
    <div className="flex flex-col gap-6">
      <SettingsSection
        title={t("systemPill.settings.title")}
        description={t("systemPill.settings.description")}
      >
        <SettingsRow
          label={t("systemPill.settings.enable")}
          description={t("systemPill.settings.enableDescription", {
            hotkey: hotkey || "⌥Space",
          })}
        >
          <Switch
            checked={settings?.enabled ?? false}
            onCheckedChange={(enabled) => void applySettings({ enabled })}
          />
        </SettingsRow>

        <SettingsRow
          label={t("systemPill.settings.defaultModel")}
          description={t("systemPill.settings.defaultModelDescription")}
        >
          <Select
            value={settings?.defaultModel ?? DEFAULT_MODEL_VALUE}
            onValueChange={(value) =>
              void applySettings({
                defaultModel: value === DEFAULT_MODEL_VALUE ? null : value,
              })
            }
          >
            <SelectTrigger className="w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_MODEL_VALUE}>
                {t("systemPill.settings.actionModelDefault")}
              </SelectItem>
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
