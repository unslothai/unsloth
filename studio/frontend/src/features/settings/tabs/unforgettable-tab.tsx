// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { fetchAdapters } from "@/features/unforgettable/api/memory-api";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  type UnforgettableSettings,
  loadUnforgettableSettings,
  updateUnforgettableSettings,
} from "../api/unforgettable";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

type ConfirmChoice = "default" | "always" | "never";

function confirmFromSettings(value: boolean | null | undefined): ConfirmChoice {
  if (value === true) return "always";
  if (value === false) return "never";
  return "default";
}

function confirmToPatch(value: ConfirmChoice): boolean | null {
  if (value === "always") return true;
  if (value === "never") return false;
  return null;
}

export function UnforgettableTab() {
  const t = useT();
  const navigate = useNavigate();
  const closeDialog = useSettingsDialogStore((s) => s.closeDialog);
  const [settings, setSettings] = useState<UnforgettableSettings | null>(null);
  const [adapters, setAdapters] = useState<{ id: string; status: string }[]>(
    [],
  );
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadUnforgettableSettings()
      .then((loaded) => {
        if (!cancelled) setSettings(loaded);
      })
      .catch((error: unknown) => {
        toast.error(
          error instanceof Error
            ? error.message
            : t("unforgettable.errors.loadSettings"),
        );
      });
    void fetchAdapters()
      .then((data) => {
        if (!cancelled) setAdapters(data.adapters);
      })
      .catch(() => {
        if (!cancelled) setAdapters([]);
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  async function patch(next: Partial<UnforgettableSettings>) {
    setSaving(true);
    try {
      const updated = await updateUnforgettableSettings(next);
      setSettings(updated);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : t("unforgettable.errors.saveSettings"),
      );
    } finally {
      setSaving(false);
    }
  }

  const confirmChoice = confirmFromSettings(settings?.confirm_retry);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1 pr-10">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.unforgettable.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.unforgettable.description")}
        </p>
        <div className="pt-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              closeDialog();
              void navigate({ to: "/unforgettable" });
            }}
          >
            {t("settings.unforgettable.openDashboard")}
          </Button>
        </div>
      </header>

      <SettingsSection
        title={t("settings.unforgettable.episode.title")}
        description={t("settings.unforgettable.episode.description")}
      >
        <SettingsRow
          label={t("settings.unforgettable.episode.planner")}
          description={t("settings.unforgettable.episode.plannerDescription")}
          hint={t("settings.unforgettable.episode.plannerHint")}
        >
          <Switch
            checked={settings?.planner === "on"}
            disabled={!settings || saving}
            onCheckedChange={(on) => void patch({ planner: on ? "on" : "off" })}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.plannerModel")}
          description={t(
            "settings.unforgettable.episode.plannerModelDescription",
          )}
        >
          <Input
            className="w-56"
            value={settings?.planner_model ?? ""}
            disabled={!settings || saving}
            placeholder={t("settings.unforgettable.episode.modelPlaceholder")}
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, planner_model: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({ planner_model: settings?.planner_model || null })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.filter")}
          description={t("settings.unforgettable.episode.filterDescription")}
          hint={t("settings.unforgettable.episode.filterHint")}
        >
          <Switch
            checked={settings?.filter !== "off"}
            disabled={!settings || saving}
            onCheckedChange={(on) =>
              void patch({ filter: on ? "on" : "off" })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.filterModel")}
          description={t(
            "settings.unforgettable.episode.filterModelDescription",
          )}
        >
          <Input
            className="w-56"
            value={settings?.filter_model ?? ""}
            disabled={!settings || saving}
            placeholder={t("settings.unforgettable.episode.modelPlaceholder")}
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, filter_model: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({ filter_model: settings?.filter_model || null })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.judgeModel")}
          description={t(
            "settings.unforgettable.episode.judgeModelDescription",
          )}
          hint={t("settings.unforgettable.episode.judgeHint")}
        >
          <Input
            className="w-56"
            value={settings?.judge_model ?? ""}
            disabled={!settings || saving}
            placeholder={t(
              "settings.unforgettable.episode.judgeModelPlaceholder",
            )}
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, judge_model: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({ judge_model: settings?.judge_model || null })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.highStakes")}
          description={t(
            "settings.unforgettable.episode.highStakesDescription",
          )}
          hint={t("settings.unforgettable.episode.highStakesHint")}
        >
          <Switch
            checked={settings?.stakes === "high"}
            disabled={!settings || saving}
            onCheckedChange={(on) =>
              void patch({ stakes: on ? "high" : null })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.confirmRetry")}
          description={t(
            "settings.unforgettable.episode.confirmRetryDescription",
          )}
        >
          <Select
            value={confirmChoice}
            disabled={!settings || saving}
            onValueChange={(value) =>
              void patch({
                confirm_retry: confirmToPatch(value as ConfirmChoice),
              })
            }
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">
                {t("settings.unforgettable.episode.confirmDefault")}
              </SelectItem>
              <SelectItem value="always">
                {t("settings.unforgettable.episode.confirmAlways")}
              </SelectItem>
              <SelectItem value="never">
                {t("settings.unforgettable.episode.confirmNever")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.skipStanding")}
          description={t(
            "settings.unforgettable.episode.skipStandingDescription",
          )}
        >
          <Switch
            checked={Boolean(settings?.skip_standing)}
            disabled={!settings || saving}
            onCheckedChange={(on) => void patch({ skip_standing: on })}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.adapter")}
          description={t("settings.unforgettable.episode.adapterDescription")}
          hint={t("settings.unforgettable.episode.adapterHint")}
        >
          <Select
            value={settings?.adapter_id ?? "none"}
            disabled={!settings || saving}
            onValueChange={(value) =>
              void patch({ adapter_id: value === "none" ? null : value })
            }
          >
            <SelectTrigger className="w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">
                {t("settings.unforgettable.episode.adapterNone")}
              </SelectItem>
              {adapters.map((adapter) => (
                <SelectItem key={adapter.id} value={adapter.id}>
                  {adapter.id.slice(0, 8)} · {adapter.status}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.testCommand")}
          description={t(
            "settings.unforgettable.episode.testCommandDescription",
          )}
        >
          <Input
            className="w-56"
            value={settings?.test_command ?? ""}
            disabled={!settings || saving}
            placeholder="pytest"
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, test_command: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({ test_command: settings?.test_command || null })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.maxClones")}
          description={t("settings.unforgettable.episode.budgetDescription")}
        >
          <Input
            className="w-24"
            type="number"
            min={1}
            value={settings?.max_clones ?? ""}
            disabled={!settings || saving}
            placeholder="1"
            onChange={(event) =>
              setSettings((prev) =>
                prev
                  ? {
                      ...prev,
                      max_clones: event.target.value
                        ? Number(event.target.value)
                        : null,
                    }
                  : prev,
              )
            }
            onBlur={() => void patch({ max_clones: settings?.max_clones })}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.maxSimTurns")}
          description={t("settings.unforgettable.episode.budgetDescription")}
        >
          <Input
            className="w-24"
            type="number"
            min={1}
            value={settings?.max_sim_turns ?? ""}
            disabled={!settings || saving}
            placeholder="8"
            onChange={(event) =>
              setSettings((prev) =>
                prev
                  ? {
                      ...prev,
                      max_sim_turns: event.target.value
                        ? Number(event.target.value)
                        : null,
                    }
                  : prev,
              )
            }
            onBlur={() =>
              void patch({ max_sim_turns: settings?.max_sim_turns })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.episode.twinPlugin")}
          description={t(
            "settings.unforgettable.episode.twinPluginDescription",
          )}
          hint={t("settings.unforgettable.episode.twinPluginHint")}
        >
          <Select
            value={settings?.twin_plugin ?? "fs.copy"}
            disabled={!settings || saving}
            onValueChange={(value) => void patch({ twin_plugin: value })}
          >
            <SelectTrigger className="w-52">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="fs.copy">
                {t("settings.unforgettable.episode.twinFsCopy")}
              </SelectItem>
              <SelectItem value="none">
                {t("settings.unforgettable.episode.twinNone")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.unforgettable.approver.title")}
        description={t("settings.unforgettable.approver.description")}
      >
        <SettingsRow
          label={t("settings.unforgettable.approver.voter")}
          hint={t("settings.unforgettable.approver.voterHint")}
        >
          <Select
            value={settings?.voter ?? "off"}
            disabled={!settings || saving}
            onValueChange={(value) => void patch({ voter: value })}
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="off">
                {t("settings.unforgettable.approver.voterOff")}
              </SelectItem>
              <SelectItem value="advisory">
                {t("settings.unforgettable.approver.voterAdvisory")}
              </SelectItem>
              <SelectItem value="binding">
                {t("settings.unforgettable.approver.voterBinding")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow label={t("settings.unforgettable.approver.voterModel")}>
          <Input
            className="w-56"
            value={settings?.voter_model ?? ""}
            disabled={!settings || saving}
            placeholder={t("settings.unforgettable.episode.modelPlaceholder")}
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, voter_model: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({ voter_model: settings?.voter_model || null })
            }
          />
        </SettingsRow>
        <SettingsRow label={t("settings.unforgettable.approver.supervisorUrl")}>
          <Input
            className="w-56"
            value={settings?.supervisor_url ?? ""}
            disabled={!settings || saving}
            placeholder="http://127.0.0.1:8080/supervise"
            onChange={(event) =>
              setSettings((prev) =>
                prev ? { ...prev, supervisor_url: event.target.value } : prev,
              )
            }
            onBlur={() =>
              void patch({
                supervisor_url: settings?.supervisor_url || null,
              })
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.unforgettable.approver.supervisorTimeout")}
        >
          <Input
            className="w-24"
            type="number"
            min={1}
            value={settings?.supervisor_timeout ?? 30}
            disabled={!settings || saving}
            onChange={(event) =>
              setSettings((prev) =>
                prev
                  ? {
                      ...prev,
                      supervisor_timeout: Number(event.target.value) || 30,
                    }
                  : prev,
              )
            }
            onBlur={() =>
              void patch({
                supervisor_timeout: settings?.supervisor_timeout ?? 30,
              })
            }
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.unforgettable.store.title")}
        description={t("settings.unforgettable.store.description")}
      >
        <SettingsRow label={t("settings.unforgettable.store.path")}>
          <span
            className={cn(
              "max-w-xs truncate text-xs text-muted-foreground",
              !settings?.db_path && "italic",
            )}
            title={settings?.db_path}
          >
            {settings?.db_path || "…"}
          </span>
        </SettingsRow>
        <SettingsRow label={t("settings.unforgettable.store.namespace")}>
          <span className="text-sm">{settings?.namespace || "default"}</span>
        </SettingsRow>
        <p className="pt-2 text-xs text-muted-foreground">
          {t("settings.unforgettable.store.notRag")}
        </p>
      </SettingsSection>
    </div>
  );
}
