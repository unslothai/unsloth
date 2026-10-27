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
import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { cn } from "@/lib/utils";
import { translate, useT } from "@/i18n";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useState } from "react";
import {
  type ApiKey,
  createApiKey,
  fetchApiKeys,
  revokeApiKey,
} from "../api/api-keys";
import {
  type KeylessApiAccessExposure,
  type KeylessApiAccessScope,
  loadKeylessApiAccess,
} from "../api/keyless-api-access";
import { apiKeyActionMessage } from "../components/api-key-action-message";
import {
  ApiKeyRow,
  ApiKeyTableHeader,
  TOKEN_TABLE,
} from "../components/api-key-row";
import { CreateKeyForm } from "../components/create-key-form";
import { KeyRevealCard } from "../components/key-reveal-card";
import { KeylessApiAccessLink } from "../components/keyless-api-access-link";
import { McpAccessSection } from "../components/mcp-access-section";
import { ModelAliasesSection } from "../components/model-aliases-section";
import { ModelAutoSwitchSection } from "../components/model-auto-switch-section";
import { ModelMemorySection } from "../components/model-memory-section";
import { MonitorLink } from "../components/monitor-link";
import { SettingsSection } from "../components/settings-section";
import { ApiExamplesPanel } from "../components/api-example-tabs";
import { useApiTunnelPref } from "../components/usage-examples";
import { usePublishAccessUrls } from "../hooks/use-publish-access-urls";

function expiryDays(key: ApiKey): number | null {
  if (!key.expires_at) return null;
  const ms =
    new Date(key.expires_at).getTime() - new Date(key.created_at).getTime();
  return Math.max(1, Math.round(ms / 86400000));
}

export function ApiKeysTab() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<ApiKey | null>(null);
  const [revoking, setRevoking] = useState(false);
  const [regenerate, setRegenerate] = useState(false);
  const [revealed, setRevealed] = useState<string | null>(null);
  const [useTunnel, setUseTunnel] = useApiTunnelPref();
  // Owner-only routes, as the Remote & LAN tab is.
  usePublishAccessUrls(isOwner);
  const [keyless, setKeyless] = useState<{
    scope: KeylessApiAccessScope;
    tools: boolean;
    exposure: KeylessApiAccessExposure | null;
  }>({ scope: "off", tools: false, exposure: null });
  // Read for the examples' placeholder key; a tunnel coming or going changes exposure.
  const cloudflareUrl = usePlatformStore((state) => state.cloudflareUrl);
  // biome-ignore lint/correctness/useExhaustiveDependencies: cloudflareUrl is a trigger, not an input
  useEffect(() => {
    if (!isOwner) {
      return;
    }
    let cancelled = false;
    loadKeylessApiAccess()
      .then((next) => {
        if (!cancelled) {
          setKeyless({
            scope: next.suspendedByAccounts ? "off" : next.scope,
            tools: next.tools,
            exposure: next.exposure,
          });
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [isOwner, cloudflareUrl]);
  const reduced = useReducedMotion();
  const transition = reduced
    ? { duration: 0 }
    : { duration: 0.18, ease: [0.165, 0.84, 0.44, 1] as const };

  // api-keys.ts throws generic English errors; show the translated message.
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
    setActionError(null);
    try {
      if (regenerate) {
        // Created before the old one is revoked, so a failed create loses nothing.
        const created = await createApiKey(
          revokeTarget.name,
          expiryDays(revokeTarget),
        );
        setRevealed(created.key);
      }
      await revokeApiKey(revokeTarget.id);
      await load();
      setRevokeTarget(null);
    } catch (cause) {
      setActionError(
        apiKeyActionMessage(
          translate(
            regenerate
              ? "settings.apiKeys.regenerateError"
              : "settings.apiKeys.revokeError",
          ),
          cause,
        ),
      );
      void load();
      setRevokeTarget(null);
    } finally {
      setRevoking(false);
    }
  };

  return (
    <div className="settings-page">
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

      <section className="flex min-w-0 flex-col gap-4">
        <h2 className="settings-heading text-base font-semibold font-heading">
          {t("settings.apiKeys.accessTokens")}
        </h2>
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
                onError={setActionError}
                existingNames={keys.map((k) => k.name)}
              />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex min-w-0 flex-col">
          {actionError ? (
            <div className="mb-2 rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
              {actionError}
            </div>
          ) : null}
          {error ? (
            <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-destructive">
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
            <div className="hover-scrollbar flex max-h-72 min-w-0 flex-col overflow-y-auto pr-1 [scrollbar-gutter:stable]">
              <div className={cn("overlay-scrollbar-gutter", TOKEN_TABLE)}>
                <ApiKeyTableHeader />
                {keys.map((k) => (
                  <ApiKeyRow
                    key={k.id}
                    apiKey={k}
                    onRevoke={(key) => {
                      setRegenerate(false);
                      setRevokeTarget(key);
                    }}
                    onRegenerate={(key) => {
                      setRegenerate(true);
                      setRevokeTarget(key);
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <ApiExamplesPanel
        apiKey={revealed}
        useTunnel={useTunnel}
        onUseTunnelChange={setUseTunnel}
        keylessScope={keyless.scope}
        keylessTools={keyless.tools}
        keylessExposure={keyless.exposure}
      />

      {/* Installation-wide, so owner-only routes. */}
      <SettingsSection title="Access & monitoring">
        {isOwner ? <McpAccessSection apiKey={revealed} /> : null}
        {isOwner ? <KeylessApiAccessLink /> : null}
        <MonitorLink hideHeading={true} />
      </SettingsSection>

      {isOwner ? (
        <>
          <ModelMemorySection variant="api" />

          <ModelAutoSwitchSection />

          <ModelAliasesSection />
        </>
      ) : null}

      <Dialog
        open={revokeTarget !== null}
        onOpenChange={(o) => !o && setRevokeTarget(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t(
                regenerate
                  ? "settings.apiKeys.regenerateTitle"
                  : "settings.apiKeys.revokeTitle",
                { name: revokeTarget?.name ?? "" },
              )}
            </DialogTitle>
            <DialogDescription>
              {t(
                regenerate
                  ? "settings.apiKeys.regenerateDescription"
                  : "settings.apiKeys.revokeDescription",
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRevokeTarget(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={confirmRevoke}
              disabled={revoking}
              variant={regenerate ? "dark" : "destructive"}
            >
              {regenerate
                ? t(
                    revoking
                      ? "settings.apiKeys.regenerating"
                      : "settings.apiKeys.regenerateAction",
                  )
                : revoking
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
