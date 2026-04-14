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
import { useCallback, useEffect, useState } from "react";
import { fetchApiKeys, revokeApiKey, type ApiKey } from "../api/api-keys";
import { ApiKeyRow } from "../components/api-key-row";
import { CreateKeyForm } from "../components/create-key-form";
import { RevealKeyDialog } from "../components/reveal-key-dialog";
import { UsageExamples } from "../components/usage-examples";

export function ApiKeysTab() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<ApiKey | null>(null);
  const [revoking, setRevoking] = useState(false);
  const [revealed, setRevealed] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setKeys(await fetchApiKeys());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load API keys.");
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
      setError(e instanceof Error ? e.message : "Couldn't revoke key.");
    } finally {
      setRevoking(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">API Keys</h1>
        <p className="text-xs text-muted-foreground">
          Access Unsloth Studio programmatically via the OpenAI-compatible API.
        </p>
      </header>

      <CreateKeyForm
        onCreated={(raw) => {
          setRevealed(raw);
          void load();
        }}
        onError={setError}
      />

      <section className="flex flex-col">
        <h2 className="mb-2 text-sm font-semibold text-foreground">Your keys</h2>
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
            No API keys yet.
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
            <DialogTitle>Revoke key "{revokeTarget?.name}"?</DialogTitle>
            <DialogDescription>
              Applications using this key will immediately lose access. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRevokeTarget(null)}>
              Cancel
            </Button>
            <Button
              onClick={confirmRevoke}
              disabled={revoking}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {revoking ? "Revoking…" : `Revoke "${revokeTarget?.name}"`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <RevealKeyDialog rawKey={revealed} onClose={() => setRevealed(null)} />
    </div>
  );
}
