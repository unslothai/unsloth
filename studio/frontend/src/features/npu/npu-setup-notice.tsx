// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { NpuCatalog } from "./use-npu-catalog";

export function NpuSetupNotice({ catalog }: { catalog: NpuCatalog }) {
  const { status, ready, models, listError, enabling, enable } = catalog;
  const problems = status.validation?.problems ?? [];
  return (
    <div className="flex flex-col gap-2 px-2.5 pb-2 text-xs">
      <p className="text-muted-foreground">
        {status.hardware.name ?? "AMD NPU"}: FastFlowLM's own NPU builds, run
        through Lemonade. GGUF models run on the GPU.
      </p>
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
      {!ready || !status.runtime_installed ? (
        <div className="flex items-center justify-between gap-3 rounded-lg bg-muted/50 p-3">
          <span className="text-muted-foreground">
            Enabling downloads Lemonade (about 7 MB) and FastFlowLM (about 40
            MB), then checks the NPU driver.
          </span>
          <Button size="sm" onClick={() => void enable()} disabled={enabling}>
            {enabling ? <Spinner className="size-3.5" /> : null}
            {status.state === "failed"
              ? "Try again"
              : status.runtime_installed
                ? "Start NPU runtime"
                : "Enable NPU"}
          </Button>
        </div>
      ) : null}
      {listError ? <p className="text-destructive">{listError}</p> : null}
      {ready && status.runtime_installed && !models && !listError ? (
        <div className="flex justify-center py-2">
          <Spinner className="size-3.5 text-muted-foreground" />
        </div>
      ) : null}
      <p className="text-ui-11 text-muted-foreground">
        Powered by{" "}
        <a
          href="https://github.com/ROCm/FastFlowLM"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2"
        >
          FastFlowLM
        </a>{" "}
        and{" "}
        <a
          href="https://github.com/lemonade-sdk/lemonade"
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2"
        >
          Lemonade
        </a>
        .
      </p>
    </div>
  );
}
