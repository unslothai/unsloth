// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useIsAccountOwner } from "@/features/auth";
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { useT } from "@/i18n";
import { useGpuDevices } from "@/hooks/use-gpu-info";
import { useEffect, useRef, useState } from "react";
import {
  type EngineStatus,
  type InferenceEngine,
  changeEngine,
  isEngineReady,
} from "../api/engines";
import { useEngines } from "../hooks/use-engines";

const names = { vllm: "vLLM", sglang: "SGLang" };

function EngineInstall({
  engine,
  management = false,
  onUse,
}: { engine: EngineStatus; management?: boolean; onUse?: () => void }) {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const useAfterInstall = useRef(false);
  useEffect(() => {
    if (engine.job.state === "error" || engine.job.state === "cancelled") {
      useAfterInstall.current = false;
    }
    if (
      useAfterInstall.current &&
      engine.installed &&
      engine.job.state === "success"
    ) {
      useAfterInstall.current = false;
      onUse?.();
    }
  }, [engine.installed, engine.job.state, onUse]);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const running = engine.job.state === "running";
  const action = async (
    operation: "install" | "cancel" | "remove" | "rollback",
  ) => {
    setBusy(true);
    setError("");
    try {
      if (operation === "install") {
        useAfterInstall.current = !!onUse;
      }
      if (operation === "cancel") {
        useAfterInstall.current = false;
      }
      await changeEngine(engine.engine, operation);
      setConfirm(false);
    } catch (err) {
      useAfterInstall.current = false;
      setError(err instanceof Error ? err.message : "Engine operation failed");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-2 text-ui-12">
      {!(isOwner || engine.installed) && (
        <p className="text-muted-foreground">
          {t("managedEngines.ownerRequired")}
        </p>
      )}
      {engine.unsupported_reason && (
        <p className="text-muted-foreground">{engine.unsupported_reason}</p>
      )}
      {running ? (
        <output
          aria-live="polite"
          className="flex items-center justify-between gap-3"
        >
          <span>{t("managedEngines.installing")}</span>
          <Button
            size="sm"
            variant="outline"
            disabled={busy || !isOwner}
            onClick={() => void action("cancel")}
          >
            {t("managedEngines.cancelInstall")}
          </Button>
        </output>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          {engine.installed && (
            <span>
              {t("managedEngines.installed", {
                version: engine.installed_version ?? "",
              })}
            </span>
          )}
          {(!engine.installed || !engine.current || management) && (
            <Button
              size="sm"
              variant="outline"
              disabled={
                busy || engine.in_use || !isOwner || !!engine.unsupported_reason
              }
              onClick={() => setConfirm(true)}
            >
              {engine.installed
                ? t(
                    engine.current
                      ? "managedEngines.repair"
                      : "managedEngines.update",
                  )
                : t("managedEngines.install")}
            </Button>
          )}
          {management && isOwner && engine.installed && (
            <Button
              size="sm"
              variant="ghost"
              disabled={busy || engine.in_use}
              onClick={() => void action("remove")}
            >
              {t("managedEngines.remove")}
            </Button>
          )}
          {management && isOwner && engine.can_rollback && (
            <Button
              size="sm"
              variant="ghost"
              disabled={busy || engine.in_use}
              onClick={() => void action("rollback")}
            >
              {t("managedEngines.rollback")}
            </Button>
          )}
        </div>
      )}
      {management && engine.in_use && (
        <p className="text-muted-foreground">{t("managedEngines.inUse")}</p>
      )}
      {confirm && (
        <section
          aria-label={t("managedEngines.installTitle", {
            engine: names[engine.engine],
          })}
          className="space-y-3 rounded-lg border p-3"
        >
          <p>
            {t("managedEngines.confirm", {
              engine: names[engine.engine],
              version: engine.version,
            })}
          </p>
          <p>{t("managedEngines.background")}</p>
          <div className="flex gap-2">
            <Button
              size="sm"
              disabled={busy}
              onClick={() => void action("install")}
            >
              {onUse
                ? t("managedEngines.installAndLoad")
                : t("managedEngines.install")}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirm(false)}>
              {t("common.cancel")}
            </Button>
          </div>
        </section>
      )}
      {(error || engine.job.state === "error") && (
        <p role="alert" className="whitespace-pre-wrap text-destructive">
          {error || t("managedEngines.failed")}
        </p>
      )}
      {engine.job.state === "error" && (
        <details>
          <summary>{t("managedEngines.details")}</summary>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-ui-11">
            {engine.job.message}
          </pre>
        </details>
      )}
      {engine.job.state === "cancelled" && (
        <output>{t("managedEngines.cancelled")}</output>
      )}
    </div>
  );
}

export function InferenceEnginesSection() {
  const t = useT();
  const { engines, error } = useEngines();
  return (
    <SettingsSection
      title={t("managedEngines.title")}
      description={t("managedEngines.description")}
    >
      {error && <p role="alert">{error}</p>}
      {engines.map((engine) => (
        <SettingsRow
          key={engine.engine}
          label={names[engine.engine]}
          alignTop={true}
        >
          <EngineInstall engine={engine} management={true} />
        </SettingsRow>
      ))}
    </SettingsSection>
  );
}

export function InferenceEnginePicker({
  value,
  onChange,
  onReadyChange,
  onUse,
  gpuIds,
  parallelism = "tensor",
  onParallelismChange,
  precision = "auto",
  onPrecisionChange,
  onGpuChange,
}: {
  value: InferenceEngine;
  onChange: (engine: InferenceEngine) => void;
  onReadyChange: (ready: boolean) => void;
  onUse: () => void;
  gpuIds?: number[] | null;
  parallelism?: "tensor" | "pipeline" | "data";
  onParallelismChange: (mode: "tensor" | "pipeline" | "data") => void;
  precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  onPrecisionChange: (
    precision: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8",
  ) => void;
  onGpuChange: (ids: number[]) => void;
}) {
  const { engines, error } = useEngines();
  const t = useT();
  const devices = useGpuDevices()?.filter(
    (device) => device.indexKind === "physical" && /nvidia/i.test(device.name),
  );
  const selectedGpuIds = gpuIds?.length ? gpuIds : [0];
  const selected = engines.find((engine) => engine.engine === value);
  const ready = value === "auto" || isEngineReady(selected);
  useEffect(() => {
    onReadyChange(ready);
  }, [ready, onReadyChange]);
  if (
    value === "auto" &&
    engines.length > 0 &&
    engines.every((engine) => engine.unsupported_reason)
  ) {
    return null;
  }
  return (
    <div className="space-y-2 rounded-lg border p-3">
      <div className="flex items-center justify-between gap-3 text-ui-13">
        <span>{t("managedEngines.picker")}</span>
        <Select
          value={value}
          onValueChange={(next) => onChange(next as InferenceEngine)}
        >
          <SelectTrigger
            className="w-44"
            aria-label={t("managedEngines.picker")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">{t("managedEngines.default")}</SelectItem>
            {engines.map((engine) => (
              <SelectItem
                key={engine.engine}
                value={engine.engine}
                disabled={!!engine.unsupported_reason}
              >
                {names[engine.engine]}
                {engine.installed
                  ? ""
                  : ` (${t("managedEngines.installRequired")})`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {error && (
        <p role="alert" className="text-ui-12">
          {error}
        </p>
      )}
      {selected && (
        <>
          <p className="text-ui-12 text-muted-foreground">
            {t("managedEngines.scope")}
          </p>
          {devices && devices.length > 0 && (
            <div
              role="group"
              aria-label={t("managedEngines.gpu")}
              className="space-y-2"
            >
              <span className="text-ui-13">{t("managedEngines.gpu")}</span>
              <p className="text-ui-12 text-muted-foreground">
                {t("managedEngines.gpuHelp")}
              </p>
              {devices.map((device) => {
                const checked = selectedGpuIds.includes(device.index);
                return (
                  <label
                    key={device.index}
                    className="flex items-center justify-between gap-3 text-ui-12"
                  >
                    <span className="min-w-0">
                      {device.index}: {device.name}
                    </span>
                    <Switch
                      className="panel-switch shrink-0"
                      aria-label={`GPU ${device.index}: ${device.name}`}
                      checked={checked}
                      disabled={checked && selectedGpuIds.length === 1}
                      onCheckedChange={(next) =>
                        onGpuChange(
                          next
                            ? [...selectedGpuIds, device.index]
                            : selectedGpuIds.filter(
                                (id) => id !== device.index,
                              ),
                        )
                      }
                    />
                  </label>
                );
              })}
              {selectedGpuIds.length > 1 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3 text-ui-13">
                    <span>{t("managedEngines.parallelism")}</span>
                    <Select value={parallelism} onValueChange={value => onParallelismChange(value as typeof parallelism)}>
                      <SelectTrigger className="w-52" aria-label={t("managedEngines.parallelism")}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tensor">{t("managedEngines.tensor")}</SelectItem>
                        <SelectItem value="pipeline">{t("managedEngines.pipeline")}</SelectItem>
                        <SelectItem value="data">{t("managedEngines.data")}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <p className="text-ui-12 text-muted-foreground">
                    {parallelism === "tensor"
                      ? t("managedEngines.tensorParallel", { count: selectedGpuIds.length })
                      : parallelism === "pipeline"
                        ? t("managedEngines.pipelineHelp")
                        : t("managedEngines.dataHelp")}
                  </p>
                </div>
              )}
            </div>
          )}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3 text-ui-13">
              <span>{t("managedEngines.precision")}</span>
              <Select
                value={precision}
                onValueChange={(value) =>
                  onPrecisionChange(value as typeof precision)
                }
              >
                <SelectTrigger
                  className="w-44"
                  aria-label={t("managedEngines.precision")}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">
                    {t("managedEngines.precisionAuto")}
                  </SelectItem>
                  <SelectItem value="bf16">BF16 (16-bit)</SelectItem>
                  <SelectItem value="fp16">FP16 (16-bit)</SelectItem>
                  <SelectItem value="int4">4-bit</SelectItem>
                  <SelectItem value="int8">INT8 (8-bit)</SelectItem>
                  <SelectItem value="fp8">FP8 (8-bit)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <p className="text-ui-12 text-muted-foreground">
              {t("managedEngines.precisionHelp")}
            </p>
          </div>
          <EngineInstall
            key={selected.engine}
            engine={selected}
            onUse={onUse}
          />
        </>
      )}
    </div>
  );
}
