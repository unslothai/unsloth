// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { useId, useMemo, useState } from "react";
import type { ModelPickTarget } from "../components/model-selector/types";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
} from "../model-config/per-model-config";
import { sharedExtraArgsError } from "./extra-args";
import {
  SHARED_CONFIG_FIELDS,
  SHARED_CONFIG_KEYS,
  type SharedConfigKey,
  formatSharedConfigValue,
} from "./fields";
import {
  DESKTOP_RUN_CONFIG_URL_WARNING_LENGTH,
  type SharedRunConfig,
  createRunConfigLink,
  isShareableModelId,
} from "./links";

const SHARING_DEFAULTS: PerModelConfig = {
  ...DEFAULT_PER_MODEL_CONFIG,
  gpuMemoryMode: "auto",
  gpuLayers: -1,
  nCpuMoe: 0,
  selectedGpuIds: null,
  selectedGpuIndexKind: null,
};

const loopbackHostname = /^(?:localhost|127(?:\.\d{1,3}){3}|\[::1\])$/;

function linkPreview(value: SharedRunConfig, destination: string) {
  try {
    return {
      link: createRunConfigLink(
        value,
        destination === "browser" ? window.location.href : undefined,
      ),
      error: "",
    };
  } catch (cause) {
    return {
      link: "",
      error:
        cause instanceof Error ? cause.message : "Could not create this link.",
    };
  }
}

function configDetail(
  key: SharedConfigKey,
  config: PerModelConfig,
  error: string | null,
) {
  if (error !== null) {
    return key === "llamaExtraArgs"
      ? `Excluded: ${error} Edit Extra Arguments in Run settings to share them.`
      : error;
  }
  return formatSharedConfigValue(key, config);
}

export function ShareRunConfigDialog({
  target,
  config: sourceConfig,
  onClose,
}: {
  target: ModelPickTarget;
  config: PerModelConfig;
  onClose: () => void;
}) {
  const id = useId();
  const config = useMemo(
    () => ({
      ...sourceConfig,
      llamaExtraArgs: sourceConfig.llamaExtraArgs ?? null,
    }),
    [sourceConfig],
  );
  const model = target.configId ?? target.id;
  const shareableModel = isShareableModelId(model);
  const fields = useMemo(
    () =>
      SHARED_CONFIG_KEYS.filter((key) => config[key] !== undefined).map(
        (key) => {
          const error =
            key === "llamaExtraArgs" && config[key] !== null
              ? sharedExtraArgsError(config[key])
              : SHARED_CONFIG_FIELDS[key].valid(config[key])
                ? null
                : (SHARED_CONFIG_FIELDS[key].error ??
                  "This value cannot be shared.");
          return {
            key,
            valid: error === null,
            detail: configDetail(key, config, error),
          };
        },
      ),
    [config],
  );
  const [includeModel, setIncludeModel] = useState(shareableModel);
  const [includeVariant, setIncludeVariant] = useState(
    shareableModel && Boolean(target.ggufVariant),
  );
  const [destination, setDestination] = useState(() =>
    !isTauri && loopbackHostname.test(window.location.hostname)
      ? "browser"
      : "desktop",
  );
  const [selected, setSelected] = useState<Set<SharedConfigKey>>(
    () =>
      new Set(
        fields
          .filter(
            ({ key, valid }) =>
              valid &&
              (key === "llamaExtraArgs"
                ? (config.llamaExtraArgs?.length ?? 0) > 0
                : JSON.stringify(config[key]) !==
                  JSON.stringify(SHARING_DEFAULTS[key])),
          )
          .map(({ key }) => key),
      ),
  );
  const [copying, setCopying] = useState(false);
  const { link, error } = useMemo(
    () =>
      linkPreview(
        {
          ...(includeModel ? { model } : {}),
          ...(includeVariant && target.ggufVariant
            ? { ggufVariant: target.ggufVariant }
            : includeModel
              ? { isGguf: target.isGguf }
              : {}),
          config: Object.fromEntries(
            [...selected].map((key) => [key, config[key]]),
          ),
        },
        destination,
      ),
    [
      config,
      selected,
      includeModel,
      model,
      includeVariant,
      target.ggufVariant,
      target.isGguf,
      destination,
    ],
  );
  const choice = (
    key: string,
    label: string,
    checked: boolean,
    change: (checked: boolean) => void,
    detail?: string,
    disabled = false,
  ) => (
    <div key={key} className="py-2.5">
      <div className="flex items-center gap-3">
        <Checkbox
          id={`${id}-${key}`}
          aria-describedby={
            detail !== undefined ? `${id}-${key}-detail` : undefined
          }
          checked={checked}
          disabled={disabled}
          onCheckedChange={(value) => change(value === true)}
        />
        <label
          htmlFor={`${id}-${key}`}
          className={`flex min-w-0 flex-1 items-baseline justify-between gap-4 text-sm ${disabled ? "cursor-not-allowed text-muted-foreground" : "cursor-pointer"}`}
        >
          <span className="shrink-0">{label}</span>
          {detail !== undefined && !disabled && (
            <span
              id={`${id}-${key}-detail`}
              aria-hidden={true}
              title={detail}
              className="min-w-0 truncate text-xs text-muted-foreground tabular-nums"
            >
              {detail}
            </span>
          )}
        </label>
      </div>
      {detail !== undefined && disabled && (
        <p
          id={`${id}-${key}-detail`}
          className="mt-1 pl-7 text-xs text-muted-foreground"
        >
          {detail}
        </p>
      )}
    </div>
  );
  return (
    <Dialog
      open={true}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <DialogContent className="grid-cols-1 grid-rows-[auto_minmax(0,1fr)_auto_auto] gap-5 sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Share run settings</DialogTitle>
          <DialogDescription>
            Choose what to include. Omitted settings use the recipient’s
            existing defaults. Opening a link shows the settings before running.
          </DialogDescription>
        </DialogHeader>
        <div className="flex min-h-0 flex-col gap-3">
          <div className="divide-y divide-border/50 overflow-hidden rounded-2xl border border-border/60 px-4 [scrollbar-gutter:stable_both-edges]">
            {shareableModel ? (
              choice("model", "Model", includeModel, setIncludeModel, model)
            ) : (
              <p className="py-2.5 text-sm text-muted-foreground">
                This model uses a local path. The recipient can choose their own
                model.
              </p>
            )}
            {target.ggufVariant &&
              choice(
                "variant",
                "GGUF variant",
                includeVariant,
                setIncludeVariant,
                target.ggufVariant,
              )}
          </div>
          <div className="hover-scrollbar min-h-0 flex-1 divide-y divide-border/50 overflow-y-auto rounded-2xl border border-border/60 px-4 [scrollbar-gutter:stable_both-edges] sm:max-h-75">
            {fields.map(({ key, valid, detail }) =>
              choice(
                key,
                SHARED_CONFIG_FIELDS[key].label,
                selected.has(key),
                (checked) =>
                  setSelected((current) => {
                    const next = new Set(current);
                    if (checked) {
                      next.add(key);
                    } else {
                      next.delete(key);
                    }
                    return next;
                  }),
                detail,
                !valid,
              ),
            )}
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-4">
            {isTauri ? (
              <>
                <span className="text-sm font-medium">Open in</span>
                <span className="text-sm text-muted-foreground">
                  Unsloth desktop app
                </span>
              </>
            ) : (
              <>
                <label
                  htmlFor={`${id}-destination`}
                  className="text-sm font-medium"
                >
                  Open in
                </label>
                <Select value={destination} onValueChange={setDestination}>
                  <SelectTrigger id={`${id}-destination`} className="w-56">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="desktop">Unsloth desktop app</SelectItem>
                    <SelectItem value="browser">
                      This Studio web address
                    </SelectItem>
                  </SelectContent>
                </Select>
              </>
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            {destination === "browser"
              ? "This link contains your Studio web address. Recipients need access to that address. A localhost address opens Studio on their own computer."
              : "The recipient needs the Unsloth desktop app installed."}
          </p>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <label
              htmlFor={`${id}-link`}
              className="flex h-9 min-w-0 flex-1 cursor-text items-center rounded-full border border-border bg-background px-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus-within:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]"
            >
              <span className="sr-only">Shareable link</span>
              <Textarea
                id={`${id}-link`}
                readOnly={true}
                value={link}
                rows={1}
                fieldSizing="fixed"
                onKeyUp={(event) => {
                  if (event.key === "Tab") {
                    event.currentTarget.select();
                  }
                }}
                className="no-scrollbar! min-h-0 flex-1 overflow-x-auto overflow-y-hidden whitespace-pre rounded-none border-0 bg-transparent p-0 py-1 font-mono text-xs leading-4 md:text-xs dark:bg-transparent dark:focus-visible:bg-transparent"
              />
            </label>
            <Button
              className="shrink-0"
              disabled={!link || copying}
              onClick={async () => {
                setCopying(true);
                try {
                  if (await copyToClipboard(link)) {
                    toast.success("Run settings link copied");
                  } else {
                    toast.error(
                      "Could not copy the link. Select and copy it from the link field.",
                    );
                  }
                } finally {
                  setCopying(false);
                }
              }}
            >
              Copy link
            </Button>
          </div>
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
          {destination === "desktop" &&
            link.length > DESKTOP_RUN_CONFIG_URL_WARNING_LENGTH && (
              <output className="block text-xs text-amber-700 dark:text-amber-300">
                Long desktop links may not open on Windows. Include fewer
                settings{!isTauri && " or use a browser link"}.
              </output>
            )}
          <p className="text-xs text-muted-foreground">
            Anyone with the link can read the included settings. Custom text,
            template code and file, network or tool arguments cannot be shared.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
