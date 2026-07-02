// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { translate, useT } from "@/i18n";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useState } from "react";
import { fetchApiKeys, revokeApiKey, type ApiKey } from "../api/api-keys";
import { ApiMonitorConsole } from "../components/api-monitor-console";
import { ApiKeyRow } from "../components/api-key-row";
import { CreateKeyForm } from "../components/create-key-form";
import { ModelAutoSwitchSection } from "../components/model-auto-switch-section";
import { KeyRevealCard } from "../components/key-reveal-card";
import { UsageExamples } from "../components/usage-examples";

export function ApiKeysTab() {
  const t = useT();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<ApiKey | null>(null);
  const [revoking, setRevoking] = useState(false);
  const [revealed, setRevealed] = useState<string | null>(null);
  const reduced = useReducedMotion();
  const transition = reduced
    ? { duration: 0 }
    : { duration: 0.18, ease: [0.165, 0.84, 0.44, 1] as const };

  // ../api/api-keys.ts throws generic English errors; use the translated
  // message so zh-CN users don't see English strings bleed through.
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setKeys(await fetchApiKeys());
    } catch {
      setError(translate("settings.apiKeys.loadError"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadInitialApiKeys() {
      try {
        const apiKeys = await fetchApiKeys();
        if (cancelled) return;
        setKeys(apiKeys);
        setError(null);
      } catch {
        if (cancelled) return;
        setError(translate("settings.apiKeys.loadError"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void loadInitialApiKeys();
    return () => {
      cancelled = true;
    };
  }, []);

  const confirmRevoke = async () => {
    if (!revokeTarget) return;
    setRevoking(true);
    try {
      await revokeApiKey(revokeTarget.id);
      await load();
      setRevokeTarget(null);
    } catch {
      setError(translate("settings.apiKeys.revokeError"));
    } finally {
      setRevoking(false);
    }
  };

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <header className="flex min-w-0 flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.apiKeys.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.apiKeys.description")}{" "}
          <a
            href="https://unsloth.ai/docs/basics/api"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-foreground underline decoration-border underline-offset-2 transition-colors hover:decoration-foreground"
          >
            {t("settings.apiKeys.readDocs")}
          </a>
          .
        </p>
      </header>

      <AnimatePresence mode="wait" initial={false}>
        {revealed !== null ? (
          <motion.div
            key="reveal"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={transition}
          >
            <KeyRevealCard
              rawKey={revealed}
              onDone={() => setRevealed(null)}
            />
          </motion.div>
        ) : (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={transition}
          >
            <CreateKeyForm
              onCreated={(raw) => {
                setRevealed(raw);
                void load();
              }}
              onError={setError}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <section className="flex min-w-0 flex-col">
        <h2 className="mb-2 text-sm font-semibold text-foreground">
          {t("settings.apiKeys.accessTokens")}
        </h2>
        {error ? (
          <div className="rounded-md border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
            {error}
          </div>
        ) : loading ? (
          <div className="flex flex-col gap-2 py-2">
            {[0, 1].map((i) => (
              <div
                key={i}
                className="h-12 animate-pulse rounded-md bg-muted/40"
              />
            ))}
          </div>
        ) : keys.length === 0 ? (
          <p className="py-6 text-center text-xs text-muted-foreground">
            {t("settings.apiKeys.noAccess")}
          </p>
        ) : (
          <div className="flex min-w-0 flex-col">
            {keys.map((k) => (
              <ApiKeyRow key={k.id} apiKey={k} onRevoke={setRevokeTarget} />
            ))}
          </div>
        )}
      </section>

      <ApiMonitorConsole />

      <UsageExamples apiKey={revealed} />

      <ModelAutoSwitchSection />

      <Dialog open={revokeTarget !== null} onOpenChange={(o) => !o && setRevokeTarget(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.apiKeys.revokeTitle", {
                name: revokeTarget?.name ?? "",
              })}
            </DialogTitle>
            <DialogDescription>
              {t("settings.apiKeys.revokeDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRevokeTarget(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={confirmRevoke}
              disabled={revoking}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {revoking
                ? t("settings.apiKeys.revoking")
                : t("settings.apiKeys.revokeAction", {
                    name: revokeTarget?.name ?? "",
                  })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
