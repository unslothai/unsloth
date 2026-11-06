// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useSettingsDialogStore } from "@/features/settings";
import { useEffect, useState } from "react";
import {
  type ClaudeAgentStatus,
  getClaudeAgentStatus,
  installClaudeAgentSdk,
} from "./api/providers-api";
import {
  claudeAgentModelLabels,
  isClaudeAgentOffMessage,
} from "./claude-agent";

export function ClaudeAgentStatusCard() {
  const [status, setStatus] = useState<ClaudeAgentStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const [installing, setInstalling] = useState(false);

  async function run(
    request: () => Promise<ClaudeAgentStatus>,
    fallback: string,
  ) {
    setError("");
    try {
      setStatus(await request());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : fallback);
    }
  }

  async function check() {
    setBusy(true);
    await run(getClaudeAgentStatus, "Could not check Claude Agent.");
    setBusy(false);
  }

  async function install() {
    setInstalling(true);
    await run(installClaudeAgentSdk, "Could not install the Claude Agent SDK.");
    setInstalling(false);
  }

  // biome-ignore lint/correctness/useExhaustiveDependencies: checks once on mount
  useEffect(() => {
    void check();
  }, []);

  const off = isClaudeAgentOffMessage(error);
  const sdkMissing = status?.sdk_missing === true && !status.ready;
  const summary = installing
    ? "Installing the Claude Agent SDK. This can take a few minutes..."
    : busy
      ? "Checking Claude Code on this machine..."
      : status?.ready
        ? `Signed in (${status.account}) with Claude Code ${status.version ?? ""}.`
        : sdkMissing
          ? "The Claude Agent SDK is not installed in Studio's Python yet."
          : (status?.message ?? "");
  const models = status?.ready ? claudeAgentModelLabels(status.models) : [];
  return (
    <section className="space-y-3 rounded-xl border border-border/70 bg-background/45 p-4">
      <div>
        <p className="text-sm font-medium">Claude Agent (experimental)</p>
        <p className="text-xs text-muted-foreground">
          Uses the Claude Code login on this machine. Unsloth never reads or
          stores it.
        </p>
      </div>
      {summary && !off ? <p className="text-xs">{summary}</p> : null}
      {models.length ? (
        <p className="text-xs text-muted-foreground">
          Models: {models.join(", ")}
        </p>
      ) : null}
      {off ? (
        <p className="text-xs text-muted-foreground">
          Claude Agent is off. Turn it on in Settings &gt; System &gt; Developer
          mode.
        </p>
      ) : error ? (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      ) : null}
      {sdkMissing && status?.message && !installing ? (
        <p className="text-xs text-muted-foreground">
          Or install it by hand:{" "}
          <code className="break-all">{status.message}</code>
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        {off ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("resources")
            }
          >
            Open Developer mode
          </Button>
        ) : null}
        {sdkMissing ? (
          <Button
            type="button"
            size="sm"
            disabled={busy || installing}
            onClick={() => void install()}
          >
            {installing ? "Installing..." : "Install"}
          </Button>
        ) : null}
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={busy || installing}
          onClick={() => void check()}
        >
          Check again
        </Button>
      </div>
    </section>
  );
}
