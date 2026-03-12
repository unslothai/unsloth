// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link, useNavigate } from "@tanstack/react-router";
import { Eye, EyeOff } from "lucide-react";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { ReactElement } from "react";
import { refreshSession } from "../api";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  resetOnboardingDone,
  storeAuthTokens,
} from "../session";

type AuthMode = "login" | "signup";

type AuthStatusResponse = {
  initialized: boolean;
};

type TokenResponse = {
  access_token: string;
  refresh_token: string;
};

type AuthFormProps = {
  mode: AuthMode;
};

export function AuthForm({ mode }: AuthFormProps): ReactElement | null {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [username, setUsername] = useState("admin");
  const [setupToken, setSetupToken] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;

    async function initializeAuthForm(): Promise<void> {
      if (hasRefreshToken()) {
        const refreshed = await refreshSession();
        if (refreshed) {
          if (!canceled) setStatusLoading(false);
          navigate({ to: getPostAuthRoute() });
          return;
        }
      }
      if (hasAuthToken()) {
        if (!canceled) setStatusLoading(false);
        navigate({ to: getPostAuthRoute() });
        return;
      }

      try {
        const response = await fetch("/api/auth/status");
        if (!response.ok) throw new Error("Failed to load auth status.");
        const result = (await response.json()) as AuthStatusResponse;
        if (!canceled) {
          setInitialized(result.initialized);
          // Auto-redirect to the correct page based on init state
          if (mode === "login" && result.initialized === false) {
            navigate({ to: "/signup" });
            return;
          }
          if (mode === "signup" && result.initialized === true) {
            navigate({ to: "/login" });
            return;
          }
        }
      } catch (err: unknown) {
        if (!canceled) {
          setError(err instanceof Error ? err.message : "Failed to load.");
        }
      } finally {
        if (!canceled) setStatusLoading(false);
      }
    }

    void initializeAuthForm();

    return () => {
      canceled = true;
    };
  }, [navigate]);

  const blockedByState =
    (mode === "login" && initialized === false) ||
    (mode === "signup" && initialized === true);

  const isLoginMode = mode === "login";
  let helperText: string | null = null;
  if (isLoginMode && initialized === false) {
    helperText = "Auth not initialized. go setup first.";
  } else if (!isLoginMode && initialized === true) {
    helperText = "Auth already initialized. use login.";
  }
  const title = isLoginMode ? "Welcome back" : "Welcome to Unsloth Studio!";
  const subtitle = isLoginMode
    ? "Sign in to continue"
    : "Create first admin account";
  const submitLabel = isLoginMode ? "Login" : "Create account";
  const switchText = isLoginMode ? "Need setup first? " : "Already initialized? ";
  const switchLinkTo = isLoginMode ? "/signup" : "/login";
  const switchLinkText = isLoginMode ? "Setup account" : "Login";

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!isLoginMode && !setupToken.trim()) {
      setError("Setup token required.");
      return;
    }

    setLoading(true);
    try {
      const endpoint = isLoginMode ? "/api/auth/login" : "/api/auth/setup";
      const payload: { username: string; password: string; setup_token?: string } = {
        username: username.trim(),
        password,
      };
      if (!isLoginMode) {
        payload.setup_token = setupToken.trim();
      }
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        let message = "Auth failed.";
        const errorPayload = (await response
          .json()
          .catch(() => null)) as { detail?: string } | null;
        if (errorPayload?.detail) message = errorPayload.detail;
        throw new Error(message);
      }
      const token = (await response.json()) as TokenResponse;

      if (!isLoginMode) resetOnboardingDone();
      storeAuthTokens(token.access_token, token.refresh_token);
      navigate({ to: getPostAuthRoute() });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Auth failed.");
    } finally {
      setLoading(false);
    }
  }

  if (statusLoading && initialized === null && error === null) return null;

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1.5 text-center">
        <img
          src="/Sloth emojis/large sloth wave.png"
          alt="Unsloth waving mascot"
          className="mx-auto mb-2 h-20 w-20 object-contain"
        />
        <h2 className="text-2xl font-semibold text-foreground">{title}</h2>
        <p className="text-muted-foreground">{subtitle}</p>
      </div>
      <form className="space-y-5" onSubmit={handleSubmit}>
        <div className="space-y-2">
          <Label htmlFor="username">Username</Label>
          <Input
            id="username"
            autoComplete="username"
            placeholder="admin"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? "text" : "password"}
              className="pr-10"
              autoComplete={
                mode === "login" ? "current-password" : "new-password"
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={8}
              required
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:bg-transparent"
              onClick={() => setShowPassword((prev) => !prev)}
            >
              {showPassword ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </Button>
          </div>
          {!isLoginMode && (
            <p className="text-xs text-muted-foreground">Must be at least 8 characters</p>
          )}
        </div>

        {!isLoginMode && (
          <div className="space-y-2">
            <Label htmlFor="setup-token">Setup token</Label>
            <Input
              id="setup-token"
              autoComplete="off"
              placeholder="Paste token from backend console"
              value={setupToken}
              onChange={(event) => setSetupToken(event.target.value)}
              required
            />
          </div>
        )}

        {helperText && (
          <p className="text-center text-sm text-amber-600">{helperText}</p>
        )}
        {error && <p className="text-center text-sm text-destructive">{error}</p>}

        <Button
          type="submit"
          className="w-full"
          disabled={loading || statusLoading || blockedByState || (!isLoginMode && password.length < 8)}
        >
          {loading ? "Please wait..." : submitLabel}
        </Button>
      </form>

      <p className="text-center text-sm text-muted-foreground">
        {switchText}
        <Link to={switchLinkTo} className="text-primary hover:underline">
          {switchLinkText}
        </Link>
      </p>
    </div>
  );
}
