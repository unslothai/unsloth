// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
import { Card } from "@/components/ui/card";
import { transitionBrowserAccount } from "@/lib/account-transition";
import { apiUrl } from "@/lib/api-base";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { sessionAccount } from "./account-session";
import { type TokenResponse, setLoginMode } from "./login-client";
import { setMustChangePassword, storeAuthTokens } from "./session";

export function OIDCCallbackPage() {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function complete(): Promise<void> {
      const handoff = new URLSearchParams(window.location.search).get(
        "handoff",
      );
      // Remove the one-time code before any subsequent navigation or third-party resource load.
      window.history.replaceState({}, document.title, "/auth/oidc/callback");
      if (!handoff) throw new Error("The SSO login response is incomplete.");

      const response = await fetch(apiUrl("/api/auth/oidc/handoff"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handoff }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(payload?.detail ?? "SSO login could not be completed.");
      }
      const token = (await response.json()) as TokenResponse;
      if (cancelled) return;
      const username =
        sessionAccount(token.access_token)?.username ?? "oidc-user";
      const finishSession = () => {
        setLoginMode("multi");
        setMustChangePassword(false);
        storeAuthTokens(token.access_token, token.refresh_token);
      };
      const replaced = await transitionBrowserAccount(
        { username, accountId: token.account_id },
        "/chat",
        finishSession,
      );
      if (!replaced && !cancelled) navigate({ to: "/chat" });
    }

    void complete().catch((reason: unknown) => {
      if (!cancelled)
        setError(
          reason instanceof Error ? reason.message : "SSO login failed.",
        );
    });
    return () => {
      cancelled = true;
    };
  }, [navigate]);

  return (
    <div className="flex min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] items-center justify-center bg-background p-6">
      <Card className="w-full max-w-sm space-y-4 rounded-[3rem] px-8 py-10 text-center">
        <MascotImg
          src="Sloth emojis/large sloth wave.png"
          className="mx-auto h-20 w-20 object-contain"
        />
        <h1 className="text-2xl font-semibold">Completing sign in</h1>
        <p
          className={
            error ? "text-sm text-destructive" : "text-sm text-muted-foreground"
          }
        >
          {error ?? "Validating your SSO session..."}
        </p>
        {error && (
          <a
            href="/login"
            className="inline-block text-sm text-primary hover:underline"
          >
            Back to login
          </a>
        )}
      </Card>
    </div>
  );
}
