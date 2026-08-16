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
  noteInteractiveSave,
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
    // Tells the startup sync that any snapshot it is holding is now stale.
    noteInteractiveSave();
    saveChainRef.current = saveChainRef.current.then(async () => {
      // The write and the native apply fail for different reasons and need
      // different recoveries, so they are handled separately rather than in
      // one catch: a rejected write means we do not know what is stored, while
      // a rejected apply means the backend holds a value the machine refused.
      let saved: PillSettings;
      try {
        saved = await updatePillSettings(update);
      } catch {
        toast.error(t("systemPill.settings.saveError"));
        // A predecessor may have persisted and then skipped its own apply as
        // superseded by this save. With this one failed, nothing else will
        // apply it, so read back what actually stuck. Only the newest save
        // reconciles; an older one still has a successor coming.
        if (seq !== saveSeqRef.current) return;
        try {
          const actual = await fetchPillSettings();
          if (seq !== saveSeqRef.current) return;
          // Armed here too: this is persisted state, so the initial load must
          // not later commit its older snapshot over it.
          editedRef.current = true;
          setSettings(actual);
          await syncNativePillConfig(actual);
        } catch {
          // Backend unreachable too; the next open re-reads it.
        }
        return;
      }

      // A superseded save must not restore its own result over the newer one.
      if (seq !== saveSeqRef.current) return;
      // Fenced only once a save actually landed, so a failed edit cannot
      // suppress the initial load and leave the tab showing defaults.
      editedRef.current = true;
      setSettings(saved);

      try {
        await syncNativePillConfig(saved);
      } catch {
        toast.error(t("systemPill.settings.saveError"));
        // The backend took the value but the native layer refused it, so the
        // two now disagree about whether a shortcut exists. Native status is
        // the only one that knows what is actually registered, so make the
        // backend and the switch agree with IT rather than with what we asked
        // for. This covers both directions: a refused enable (nothing
        // registered, so back to disabled) and a refused disable, where Rust
        // restores the previous hotkey when its save fails and the bar is
        // therefore still live despite the request to turn it off.
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
          // Could not correct it either; the next open reads the backend.
        }
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
