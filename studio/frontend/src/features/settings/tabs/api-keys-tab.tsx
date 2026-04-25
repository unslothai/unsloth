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
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useState } from "react";
import { useI18n } from "@/features/i18n";
import { fetchApiKeys, revokeApiKey, type ApiKey } from "../api/api-keys";
import { ApiKeyRow } from "../components/api-key-row";
import { CreateKeyForm } from "../components/create-key-form";
import { KeyRevealCard } from "../components/key-reveal-card";
import { UsageExamples } from "../components/usage-examples";

export function ApiKeysTab() {
  const { t } = useI18n();
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

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setKeys(await fetchApiKeys());
    } catch (e) {
      setError(e instanceof Error ? e.message : t("settings.apiKeys.error.loadFallback"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const confirmRevoke = async () => {
    if (!revokeTarget) return;
    setRevoking(true);
    try {
      await revokeApiKey(revokeTarget.id);
      await load();
      setRevokeTarget(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("settings.apiKeys.error.revokeFallback"));
    } finally {
      setRevoking(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.apiKeys.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.apiKeys.subtitle")}
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

      <section className="flex flex-col">
        <h2 className="mb-2 text-sm font-semibold text-foreground">
          {t("settings.apiKeys.yourKeys")}
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
            {t("settings.apiKeys.empty")}
          </p>
        ) : (
          <div className="flex flex-col">
            {keys.map((k) => (
              <ApiKeyRow key={k.id} apiKey={k} onRevoke={setRevokeTarget} />
            ))}
          </div>
        )}
      </section>

      <UsageExamples />

      <Dialog open={revokeTarget !== null} onOpenChange={(o) => !o && setRevokeTarget(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.apiKeys.revoke.confirmTitle").replace(
                "{name}",
                revokeTarget?.name ?? "",
              )}
            </DialogTitle>
            <DialogDescription>
              {t("settings.apiKeys.revoke.confirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRevokeTarget(null)}>
              {t("settings.apiKeys.revoke.cancel")}
            </Button>
            <Button
              onClick={confirmRevoke}
              disabled={revoking}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {revoking
                ? t("settings.apiKeys.revoke.revoking")
                : t("settings.apiKeys.revoke.cta").replace(
                    "{name}",
                    revokeTarget?.name ?? "",
                  )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
