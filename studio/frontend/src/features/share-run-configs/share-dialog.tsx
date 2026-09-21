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
import { useId, useState } from "react";
import type { ModelPickTarget } from "../model-picker/components/model-selector/types";
import { formatExtraArgs } from "../model-picker/model-config/llama-extra-args";
import {
  DEFAULT_PER_MODEL_CONFIG,
  type PerModelConfig,
} from "../model-picker/model-config/per-model-config";
import { sharedExtraArgsError } from "./extra-args";
import {
  SHARED_CONFIG_FIELDS,
  SHARED_CONFIG_KEYS,
  type SharedConfigKey,
} from "./fields";
import {
  type SharedRunConfig,
  createRunConfigLink,
  isShareableModelId,
} from "./links";

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

function configDetail(key: SharedConfigKey, config: PerModelConfig) {
  const value = config[key];
  if (!SHARED_CONFIG_FIELDS[key].valid(value)) {
    return key === "llamaExtraArgs"
      ? `Excluded: ${sharedExtraArgsError(value)} Edit Extra Arguments in Run settings to share them.`
      : (SHARED_CONFIG_FIELDS[key].error ?? "This value cannot be shared.");
  }
  if (key === "llamaExtraArgs") {
    return formatExtraArgs(config.llamaExtraArgs) || "No extra arguments";
  }
  if (value === null) {
    return "Default";
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}

export function ShareRunConfigDialog({
  target,
  config,
  onClose,
}: {
  target: ModelPickTarget;
  config: PerModelConfig;
  onClose: () => void;
}) {
  const id = useId();
  const model = target.configId ?? target.id;
  const shareableModel = isShareableModelId(model);
  const [includeModel, setIncludeModel] = useState(shareableModel);
  const [includeVariant, setIncludeVariant] = useState(
    Boolean(target.ggufVariant),
  );
  const [includeFormat, setIncludeFormat] = useState(true);
  const [destination, setDestination] = useState(
    isTauri ? "desktop" : "browser",
  );
  const [selected, setSelected] = useState<Set<SharedConfigKey>>(
    () =>
      new Set(
        SHARED_CONFIG_KEYS.filter(
          (key) =>
            config[key] !== undefined &&
            SHARED_CONFIG_FIELDS[key].valid(config[key]) &&
            JSON.stringify(config[key]) !==
              JSON.stringify(DEFAULT_PER_MODEL_CONFIG[key]),
        ),
      ),
  );
  const [copying, setCopying] = useState(false);
  const patch = Object.fromEntries(
    [...selected].map((key) => [key, config[key]]),
  );
  const { link, error } = linkPreview(
    {
      ...(includeModel ? { model } : {}),
      ...(includeVariant && target.ggufVariant
        ? { ggufVariant: target.ggufVariant }
        : {}),
      ...(includeFormat ? { isGguf: target.isGguf } : {}),
      config: patch,
    },
    destination,
  );
  const choice = (
    key: string,
    label: string,
    checked: boolean,
    change: (checked: boolean) => void,
    detail?: string,
    disabled = false,
  ) => (
    <div
      key={key}
      className={`flex min-w-0 items-start gap-2 py-1.5 ${disabled ? "sm:col-span-2" : ""}`}
    >
      <Checkbox
        id={`${id}-${key}`}
        aria-describedby={
          detail !== undefined ? `${id}-${key}-detail` : undefined
        }
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => change(value === true)}
      />
      <div className="min-w-0">
        <label
          htmlFor={`${id}-${key}`}
          className={`text-sm ${disabled ? "cursor-not-allowed text-muted-foreground" : "cursor-pointer"}`}
        >
          {label}
        </label>
        {detail !== undefined && (
          <p
            id={`${id}-${key}-detail`}
            className={`text-xs text-muted-foreground ${disabled ? "whitespace-normal break-words" : "truncate"}`}
            title={disabled ? undefined : detail}
          >
            {detail}
          </p>
        )}
      </div>
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
      <DialogContent className="gap-4 sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Share run settings</DialogTitle>
          <DialogDescription>
            Choose what to include. Omitted settings use the recipient’s
            existing defaults. Opening a link shows the settings before running.
          </DialogDescription>
        </DialogHeader>
        <div className="grid max-h-[min(40dvh,24rem)] grid-cols-1 gap-x-4 overflow-y-auto rounded-xl border p-3 sm:grid-cols-2">
          {shareableModel ? (
            choice("model", "Model", includeModel, setIncludeModel, model)
          ) : (
            <p className="mb-2 text-sm text-muted-foreground">
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
          {choice(
            "format",
            "Model format",
            includeFormat,
            setIncludeFormat,
            target.isGguf ? "GGUF" : "Native weights",
          )}
          {SHARED_CONFIG_KEYS.filter((key) => config[key] !== undefined).map(
            (key) =>
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
                configDetail(key, config),
                !SHARED_CONFIG_FIELDS[key].valid(config[key]),
              ),
          )}
        </div>
        <div className="space-y-2">
          <label htmlFor={`${id}-destination`} className="text-sm font-medium">
            Open in
          </label>
          <Select value={destination} onValueChange={setDestination}>
            <SelectTrigger id={`${id}-destination`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="desktop">Unsloth desktop app</SelectItem>
              {!isTauri && (
                <SelectItem value="browser">This Studio web address</SelectItem>
              )}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {destination === "browser"
              ? "The recipient needs access to this Studio address. A localhost address opens Studio on their own computer."
              : "The recipient needs the Unsloth desktop app installed."}
          </p>
          <label htmlFor={`${id}-link`} className="block text-sm font-medium">
            Shareable link
          </label>
          <Textarea
            id={`${id}-link`}
            readOnly={true}
            value={link}
            rows={3}
            fieldSizing="fixed"
            onFocus={(event) => event.target.select()}
            className="text-xs md:text-xs"
          />
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Anyone with the link can read the included settings. Check extra
            arguments and text for private values before sharing. Custom
            template code and file, network or tool arguments cannot be shared.
          </p>
        </div>
        <Button
          disabled={!link || copying}
          onClick={async () => {
            setCopying(true);
            try {
              if (await copyToClipboard(link)) {
                toast.success("Run settings link copied");
              } else {
                toast.error(
                  "Could not copy the link. Select and copy it above.",
                );
              }
            } finally {
              setCopying(false);
            }
          }}
        >
          Copy link
        </Button>
      </DialogContent>
    </Dialog>
  );
}
