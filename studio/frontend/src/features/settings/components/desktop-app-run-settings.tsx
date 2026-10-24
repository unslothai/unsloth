// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import {
  PER_MODEL_CONFIG_STORAGE_KEY,
  PER_MODEL_CONFIG_UPDATED_EVENT,
  readRememberedModelSettings,
  rememberModelSettingsPatch,
  rememberedSettingsKey,
} from "@/features/model-picker";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DesktopAppModel,
  type DesktopAppModelSettings,
  fetchClaudeDesktopModels,
  fetchDesktopAppModelSettings,
} from "../api/claude-desktop";
import {
  type ModelMemorySettings,
  loadModelMemorySettings,
  subscribeModelMemorySettings,
} from "../api/model-memory";
import {
  type OpenAIAutoSwitchSettings,
  loadOpenAIAutoSwitchSettings,
  subscribeOpenAIAutoSwitchSettings,
  updateOpenAIAutoSwitchSettings,
} from "../api/openai-auto-switch";
import {
  AGENT_CONTEXT_STEP,
  CLAUDE_MIN_CONTEXT,
  CODEX_MIN_CONTEXT,
  contextPatch,
  contextView,
  defaultRunSettingsModel,
  keepAliveChoices,
  keepAliveUpdate,
  keepAliveView,
  mtpPatch,
  mtpView,
  sharedModelFields,
} from "./desktop-app-run-settings-state";

const K = "settings.agents.desktopApps.runSettings";

function tokens(value: number): string {
  return value.toLocaleString();
}

function gib(bytes: number): string {
  return (bytes / 1024 ** 3).toFixed(1);
}

/** Context window, MTP and keep-loaded for the model Claude Desktop and Codex Desktop use.
 *  None of these is stored here: each control reads and writes the setting Studio already has
 *  (see desktop-app-run-settings-state.ts) and re-reads it whenever anything else changes it. */
export function DesktopAppRunSettings({ preferred }: { preferred: string[] }) {
  const t = useT();
  const [models, setModels] = useState<DesktopAppModel[]>([]);
  const [model, setModel] = useState<string | null>(null);
  // Fetched per model; a row for another model (mid-switch) is never shown.
  const [fetched, setFetched] = useState<{
    model: string;
    info: DesktopAppModelSettings | null;
    error: string | null;
  } | null>(null);
  // Bumped by every settings change event, so the remembered record is re-read.
  const [localVersion, setLocalVersion] = useState(0);
  const [autoSwitch, setAutoSwitch] = useState<OpenAIAutoSwitchSettings | null>(null);
  const [memory, setMemory] = useState<ModelMemorySettings | null>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const preferredKey = preferred.join("\n");
  const infoSeq = useRef(0);

  useEffect(() => {
    let cancelled = false;
    void fetchClaudeDesktopModels()
      .then((rows) => {
        if (cancelled) return;
        setModels(rows);
        setModel((current) =>
          current && rows.some((row) => row.id === current)
            ? current
            : defaultRunSettingsModel(rows, preferredKey ? preferredKey.split("\n") : []),
        );
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [preferredKey]);

  // The model's bounds and server-side values. Re-read on every settings change too: the guard
  // estimate depends on the speculative mode, and another browser may have saved new values.
  const refreshInfo = useCallback(() => {
    if (!model) return;
    const seq = ++infoSeq.current;
    void fetchDesktopAppModelSettings(model)
      .then((next) => {
        if (seq === infoSeq.current) setFetched({ model, info: next, error: null });
      })
      .catch((err: unknown) => {
        if (seq === infoSeq.current) {
          setFetched({ model, info: null, error: err instanceof Error ? err.message : String(err) });
        }
      });
  }, [model]);
  const current = model && fetched?.model === model ? fetched : null;
  const info = current?.info ?? null;
  const error = saveError ?? current?.error ?? null;

  // This browser's remembered record: the copy Run settings shows. Live: Run settings, the
  // model picker and this panel all fire PER_MODEL_CONFIG_UPDATED_EVENT on save, another tab
  // fires "storage".
  const key = info?.config_id ? rememberedSettingsKey(info.config_id, info.gguf_variant) : null;
  const keyId = key?.modelId ?? null;
  const keyVariant = key?.ggufVariant ?? null;
  const aliasId = info?.alias_id ?? null;
  const local = useMemo(
    () => (keyId ? readRememberedModelSettings(keyId, keyVariant, aliasId) : null),
    // localVersion: re-read the store after any change event.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [keyId, keyVariant, aliasId, localVersion],
  );

  useEffect(refreshInfo, [refreshInfo]);

  useEffect(() => {
    const onChange = () => {
      setLocalVersion((v) => v + 1);
      refreshInfo();
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key === null || event.key === PER_MODEL_CONFIG_STORAGE_KEY) onChange();
    };
    window.addEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onChange);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(PER_MODEL_CONFIG_UPDATED_EVENT, onChange);
      window.removeEventListener("storage", onStorage);
    };
  }, [refreshInfo]);

  // Keep loaded: the auto-switch settings store already broadcasts every write; Model memory
  // broadcasts its own.
  useEffect(() => {
    let cancelled = false;
    void loadOpenAIAutoSwitchSettings()
      .then((loaded) => !cancelled && setAutoSwitch(loaded))
      .catch(() => undefined);
    void loadModelMemorySettings()
      .then((loaded) => !cancelled && setMemory(loaded))
      .catch(() => undefined);
    const offSwitch = subscribeOpenAIAutoSwitchSettings(setAutoSwitch);
    const offMemory = subscribeModelMemorySettings(setMemory);
    return () => {
      cancelled = true;
      offSwitch();
      offMemory();
    };
  }, []);

  const fields = sharedModelFields(local, info);
  const ctx = contextView(fields, info);
  const mtp = mtpView(fields, info);
  const keep = keepAliveView(autoSwitch, memory?.keepResident ?? false);

  const write = (patch: Parameters<typeof rememberModelSettingsPatch>[2]) => {
    if (!keyId) return;
    if (rememberModelSettingsPatch(keyId, keyVariant, patch, undefined, local)) {
      setSaveError(null);
    } else {
      setSaveError(t(`${K}.saveError`));
    }
  };

  const saveKeepAlive = (seconds: number) => {
    void updateOpenAIAutoSwitchSettings(keepAliveUpdate(seconds, autoSwitch)).catch(
      (err: unknown) => setSaveError(err instanceof Error ? err.message : String(err)),
    );
  };

  const shown = drag ?? ctx?.value ?? 0;
  return (
    <div className="flex min-w-0 flex-col gap-3 rounded-lg border p-3">
      <div className="flex flex-col gap-0.5">
        <div className="font-medium text-sm">{t(`${K}.title`)}</div>
        <div className="text-muted-foreground text-xs">{t(`${K}.description`)}</div>
      </div>

      <div className="flex items-center justify-between gap-3">
        <span className="text-sm">{t(`${K}.model`)}</span>
        <Select value={model ?? undefined} onValueChange={setModel}>
          <SelectTrigger className="w-72 max-w-full" aria-label={t(`${K}.model`)}>
            <SelectValue placeholder={t(`${K}.pickModel`)} />
          </SelectTrigger>
          <SelectContent>
            {models.map((row) => (
              <SelectItem key={row.id} value={row.id}>
                {row.loaded ? `${row.id} · ${t(`${K}.loaded`)}` : row.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {ctx ? (
        <div className="flex flex-col gap-1.5" data-testid="agents-context">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span>{t(`${K}.context`)}</span>
            <span className="tabular-nums">
              {drag == null && ctx.auto
                ? t(`${K}.contextAuto`, { tokens: tokens(ctx.max) })
                : t(`${K}.contextTokens`, { tokens: tokens(shown) })}
            </span>
          </div>
          <Slider
            min={ctx.min}
            max={ctx.max}
            step={AGENT_CONTEXT_STEP}
            value={[shown]}
            onValueChange={([value]) => setDrag(value ?? null)}
            onValueCommit={([value]) => {
              setDrag(null);
              if (value != null) write(contextPatch(value, ctx.max));
            }}
            aria-label={t(`${K}.context`)}
            thumbValueText={(value) =>
              value >= ctx.max
                ? t(`${K}.contextAuto`, { tokens: tokens(ctx.max) })
                : t(`${K}.contextTokens`, { tokens: tokens(value) })
            }
          />
          <div className="text-muted-foreground text-xs">
            {t(`${K}.contextHint`, {
              max: tokens(ctx.max),
              claude: tokens(CLAUDE_MIN_CONTEXT),
              codex: tokens(CODEX_MIN_CONTEXT),
            })}
          </div>
          {ctx.guardCapped && ctx.guardContext != null ? (
            <div className="text-amber-600 text-xs dark:text-amber-400" data-testid="agents-guard">
              {t(`${K}.guard`, {
                tokens: tokens(ctx.guardContext),
                budget: gib(info?.guard_budget_bytes ?? 0),
              })}
            </div>
          ) : info?.guard_budget_bytes ? (
            <div className="text-muted-foreground text-xs">
              {t(`${K}.guardFits`, { budget: gib(info.guard_budget_bytes) })}
            </div>
          ) : null}
          {ctx.belowClaude ? (
            <div className="text-amber-600 text-xs dark:text-amber-400">
              {ctx.belowCodex
                ? t(`${K}.tooSmallBoth`, { tokens: tokens(ctx.effective) })
                : t(`${K}.tooSmallClaude`, { tokens: tokens(ctx.effective) })}
            </div>
          ) : null}
          {info?.loaded && info.loaded_context ? (
            <div className="text-muted-foreground text-xs">
              {t(`${K}.loadedWith`, { tokens: tokens(info.loaded_context) })}
            </div>
          ) : null}
        </div>
      ) : info && !info.is_gguf ? (
        <div className="text-muted-foreground text-xs">{t(`${K}.notGguf`)}</div>
      ) : null}

      {info?.config_id ? (
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col">
            <span className="text-sm">{t(`${K}.mtp`)}</span>
            <span className="text-muted-foreground text-xs">
              {mtp.available === false
                ? t(`${K}.mtpUnavailable`)
                : mtp.auto
                  ? t(`${K}.mtpAuto`)
                  : t(`${K}.mtpHint`)}
            </span>
          </div>
          <Switch
            checked={mtp.on}
            disabled={mtp.available === false}
            onCheckedChange={(on) => write(mtpPatch(on))}
            aria-label={t(`${K}.mtp`)}
          />
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col">
          <span className="text-sm">{t(`${K}.keepAlive`)}</span>
          <span className="text-muted-foreground text-xs">
            {keep.vetoed ? t(`${K}.keepAliveResident`) : t(`${K}.keepAliveHint`)}
          </span>
        </div>
        <Select
          value={String(keep.seconds)}
          disabled={keep.vetoed || autoSwitch == null}
          onValueChange={(value) => saveKeepAlive(Number(value))}
        >
          <SelectTrigger className="w-44" aria-label={t(`${K}.keepAlive`)}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {keepAliveChoices(keep.seconds).map((seconds) => (
              <SelectItem key={seconds} value={String(seconds)}>
                {seconds === 0
                  ? t(`${K}.keepAliveNever`)
                  : t(`${K}.keepAliveMinutes`, { minutes: Math.round(seconds / 60) })}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {error ? <div className="text-destructive text-xs">{error}</div> : null}
    </div>
  );
}
