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
  // The model scan gates this whole load, so the settings GET it carries can
  // land long after a toggle has already been saved. Let a saved edit win.
  const editedRef = useRef(false);
  // Saves are independent PUTs each followed by its own native sync, so a slow
  // earlier one can answer, or apply its config to Rust, after a later one.
  // The sequence drops superseded UI writes; the chain keeps the PUT and the
  // native sync of one save from interleaving with the next.
  const saveSeqRef = useRef(0);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    let cancelled = false;
    void Promise.all([pillStatus(), fetchPillSettings(), fetchPillModelOptions()])
      .then(([status, loaded, loadedModels]) => {
        if (cancelled) return;
        setHotkey(status.hotkey);
        if (!editedRef.current) setSettings(loaded);
        setModels(loadedModels);
      })
      .catch(() => {
        // Backend down or non-mac; leave defaults.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const applySettings = (update: Partial<PillSettings>): Promise<void> => {
    const seq = ++saveSeqRef.current;
    saveChainRef.current = saveChainRef.current.then(async () => {
      try {
        const next = await updatePillSettings(update);
        // A superseded save must not restore its own result over the newer one.
        if (seq !== saveSeqRef.current) return;
        // Fenced only once a save actually landed, so a failed edit cannot
        // suppress the initial load and leave the tab showing defaults.
        editedRef.current = true;
        setSettings(next);
        await syncNativePillConfig(next);
      } catch {
        toast.error(t("systemPill.settings.saveError"));
      }
    });
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
