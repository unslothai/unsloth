// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  type KeylessApiAccessScope,
  type KeylessApiAccessSettings,
  keylessAudience,
  loadKeylessApiAccess,
  updateKeylessApiAccess,
} from "@/features/settings/api/keyless-api-access";
import { cn } from "@/lib/utils";
import { LaptopIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";

type PendingGrant = Exclude<KeylessApiAccessScope, "off"> | "tools";

const CONFIRM_COPY: Record<
  PendingGrant,
  { title: string; body: (audience: string) => string; action: string }
> = {
  inference: {
    title: "Allow chat without a key?",
    body: (audience) =>
      `${audience} will be able to chat with your loaded model, with no API key. Training, files and settings keep needing one.`,
    action: "Allow chat",
  },
  full: {
    title: "Allow everything without a key?",
    body: (audience) =>
      `${audience} will be able to chat, start training runs, and read the files and settings in Unsloth, with no API key.`,
    action: "Allow everything",
  },
  tools: {
    title: "Let keyless callers run tools?",
    body: (audience) =>
      `${audience} will be able to run Python and terminal commands through the model, with no API key.`,
    action: "Allow tools",
  },
};

export function KeylessApiAccessSection({
  onScopeChange,
}: {
  /** fires on load and after every save, so the usage examples can follow */
  onScopeChange?: (scope: KeylessApiAccessScope) => void;
}) {
  const [settings, setSettings] = useState<KeylessApiAccessSettings | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [pending, setPending] = useState<PendingGrant | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadKeylessApiAccess()
      .then((next) => {
        if (!cancelled) {
          setSettings(next);
          setError(null);
          onScopeChange?.(next.scope);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(
            cause instanceof Error
              ? cause.message
              : "Couldn't load keyless API access.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [onScopeChange]);

  const save = async (next: KeylessApiAccessScope, tools?: boolean) => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateKeylessApiAccess(next, tools);
      setSettings(saved);
      onScopeChange?.(saved.scope);
    } catch (cause: unknown) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Couldn't update keyless API access.",
      );
    } finally {
      setSaving(false);
      setPending(null);
    }
  };

  // re-read who the dialog names, held busy so no second click lands during the round trip
  const confirm = async (confirmAs: PendingGrant) => {
    setSaving(true);
    try {
      setSettings(await loadKeylessApiAccess());
    } catch {
      // keep the exposure already on screen; the dialog still states the scope
    } finally {
      setSaving(false);
    }
    setPending(confirmAs);
  };

  const scope = settings?.scope ?? "off";
  const tools = settings?.tools === true;
  const audience = keylessAudience(settings?.exposure ?? null);
  const busy = settings === null || saving;

  const request = (next: KeylessApiAccessScope, confirmAs: PendingGrant) => {
    if (next === "off" || scope === "full") {
      void save(next);
      return;
    }
    void confirm(confirmAs);
  };

  const requestTools = (on: boolean) => {
    if (!on) {
      void save(scope, false);
      return;
    }
    void confirm("tools");
  };

  const applyPending = () => {
    if (pending === "tools") {
      void save(scope, true);
      return;
    }
    if (pending) {
      void save(pending);
    }
  };

  return (
    <section
      data-settings-label="Keyless API access"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={LaptopIcon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <h2 className="text-base font-semibold font-heading text-foreground">
              Keyless API access
            </h2>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Use Unsloth from other apps without creating an API key. Signing
              in to Unsloth is unchanged.
            </p>
          </div>
        </div>
      </div>

      {error ? (
        <p className="border-t border-border/60 px-4 py-2.5 text-xs leading-snug text-destructive">
          {error}
        </p>
      ) : null}

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label="Chat and inference"
          description="Serve the OpenAI-compatible endpoints without a key, the way LM Studio and Ollama do."
          alignTop={true}
        >
          <Switch
            checked={scope !== "off"}
            disabled={busy}
            onCheckedChange={(on) =>
              request(on ? "inference" : "off", "inference")
            }
            aria-label="Chat and inference"
          />
        </SettingsRow>

        <SettingsRow
          label="Everything else"
          description="Also serve training, files and settings without a key."
          alignTop={true}
        >
          <Switch
            checked={scope === "full"}
            disabled={busy}
            onCheckedChange={(on) => request(on ? "full" : "inference", "full")}
            aria-label="Everything else"
          />
        </SettingsRow>

        <SettingsRow
          label="Allow tools"
          description="Let keyless callers use the built-in Python, terminal and web search tools. Off unless you turn it on."
          alignTop={true}
        >
          <Switch
            checked={tools}
            disabled={busy || scope === "off"}
            onCheckedChange={requestTools}
            aria-label="Allow tools"
          />
        </SettingsRow>
      </div>

      <Dialog
        open={pending !== null}
        onOpenChange={(open) => !open && setPending(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {pending ? CONFIRM_COPY[pending].title : ""}
            </DialogTitle>
            <DialogDescription>
              {pending ? CONFIRM_COPY[pending].body(audience) : ""}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button
              disabled={saving}
              className={cn(
                (pending === "full" || pending === "tools") &&
                  "bg-destructive hover:bg-destructive/90 text-destructive-foreground",
              )}
              onClick={applyPending}
            >
              {pending ? CONFIRM_COPY[pending].action : ""}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
