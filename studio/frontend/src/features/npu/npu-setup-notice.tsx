// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { NpuCatalog } from "./use-npu-catalog";

export function NpuSetupNotice({ catalog }: { catalog: NpuCatalog }) {
  const { status, ready, models, listError } = catalog;
  const problems = status.validation?.problems ?? [];
  const setUp = ready && status.runtime_installed;
  return (
    <div
      data-testid="npu-setup-notice"
      className="flex flex-col gap-2 px-2.5 pb-2 text-xs"
    >
      {setUp ? (
        <p className="text-muted-foreground">
          These models run on the AMD NPU, not on the GPU.
        </p>
      ) : (
        <NpuEnableCard catalog={catalog} />
      )}
      {status.error || problems.length > 0 ? (
        <div className="flex gap-2 rounded-lg bg-destructive/10 p-2.5 text-destructive">
          <HugeiconsIcon
            icon={AlertCircleIcon}
            className="mt-0.5 size-3.5 shrink-0"
          />
          <div className="flex flex-col gap-1">
            <span>{status.error ?? problems.join(" ")}</span>
            {status.help_url ? (
              <a
                href={status.help_url}
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-2"
              >
                NPU driver setup
              </a>
            ) : null}
          </div>
        </div>
      ) : null}
      {listError && listError !== status.error ? (
        <p className="text-destructive">{listError}</p>
      ) : null}
      {setUp && !models && !listError ? (
        <div className="flex justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      ) : null}
    </div>
  );
}

function NpuEnableCard({ catalog }: { catalog: NpuCatalog }) {
  const { status, enabling, enable } = catalog;
  const action =
    status.state === "failed"
      ? "Try again"
      : status.runtime_installed
        ? "Start NPU runtime"
        : "Enable NPU";
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-muted/50 p-3">
      <p className="font-medium text-foreground">
        This PC has an AMD Ryzen AI NPU
      </p>
      <p className="text-muted-foreground">
        The NPU runs its own builds of models such as Qwen and Gemma, alongside
        your GPU models rather than instead of them. Enable it to see them.
      </p>
      <div className="flex items-center justify-between gap-3">
        <span className="text-muted-foreground">
          {status.runtime_installed
            ? "Finishes setup and checks the NPU driver."
            : "One-time setup: downloads about 47 MB and checks the NPU driver."}
        </span>
        <Button size="sm" onClick={() => void enable()} disabled={enabling}>
          {enabling ? <Spinner className="size-3.5" /> : null}
          {action}
        </Button>
      </div>
    </div>
  );
}
