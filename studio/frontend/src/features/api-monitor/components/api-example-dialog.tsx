// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { authFetch } from "@/features/auth";
import { useFreshApiKeyStore } from "@/features/settings/stores/fresh-api-key-store";
import { type ReactElement, useEffect, useState } from "react";
import {
  type ApiSamplingDefaults,
  buildApiExample,
} from "../api-sampling-example";
import { ApiExampleBox, useApiExampleChoices } from "./api-example-box";

export function ApiExampleDialog({
  open,
  onOpenChange,
  baseUrl,
  model,
  samplingModelId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  baseUrl: string;
  model: string;
  samplingModelId: string;
}): ReactElement {
  const [sampling, setSampling] = useState<ApiSamplingDefaults | null>(null);
  const choices = useApiExampleChoices(baseUrl, "python");
  const freshKey = useFreshApiKeyStore((s) => s.key);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setSampling(null);
    authFetch(
      `/api/inference/sampling-defaults?model=${encodeURIComponent(samplingModelId)}`,
    )
      .then((res) => (res.ok ? res.json() : {}))
      .then((body: ApiSamplingDefaults) => {
        if (!cancelled) setSampling(body);
      })
      // Older backends lack the route: the example still runs, without values.
      .catch(() => {
        if (!cancelled) setSampling({});
      });
    return () => {
      cancelled = true;
    };
  }, [open, samplingModelId]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>API example</DialogTitle>
          <DialogDescription>
            A request to <span className="font-mono">{model}</span> with the
            sampling it gets when a request leaves these out. Send your own
            values to override them.
          </DialogDescription>
        </DialogHeader>
        {sampling ? (
          <ApiExampleBox
            fitContent={true}
            choices={choices}
            hasRealKey={freshKey !== null}
            code={buildApiExample({
              baseUrl,
              model,
              sampling,
              apiKey: freshKey ?? undefined,
              language: choices.language,
              os: choices.os,
            })}
          />
        ) : (
          <span className="font-mono text-ui-12 text-muted-foreground">
            Loading…
          </span>
        )}
      </DialogContent>
    </Dialog>
  );
}
