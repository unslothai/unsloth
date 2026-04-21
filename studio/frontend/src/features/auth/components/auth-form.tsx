// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link, useNavigate } from "@tanstack/react-router";
import { Eye, EyeOff } from "lucide-react";
import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import type { SyntheticEvent } from "react";
import { usePlatformStore } from "@/config/env";
import { refreshSession } from "../api";

// Bootstrap credentials injected into index.html by the backend
// (only present while default admin must_change_password is true)
declare global {
  interface Window {
    __UNSLOTH_BOOTSTRAP__?: { username: string; password: string };
  }
}

import {
  clearAuthTokens,
  getAuthToken,
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  resetOnboardingDone,
  setMustChangePassword,
  storeAuthTokens,
} from "../session";

type AuthMode = "login" | "change-password";

type AuthStatusResponse = {
  initialized: boolean;
  requires_password_change: boolean;
};

type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

async function loginWithPassword(
  username: string,
  password: string,
): Promise<TokenResponse> {
  const response = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username: username.trim(),
      password,
    }),
  });

  if (!response.ok) {
    const errorPayload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(errorPayload?.detail ?? "Login failed.");
  }

  return (await response.json()) as TokenResponse;
}

type AuthFormProps = {
  mode: AuthMode;
};

const HIDDEN_LOGIN_USERNAME = "unsloth";

export function AuthForm({ mode }: AuthFormProps): ReactElement | null {
  const navigate = useNavigate();
  const isLoginMode = mode === "login";
  const [showPassword, setShowPassword] = useState(false);
  const username = HIDDEN_LOGIN_USERNAME;
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [requiresPasswordChange, setRequiresPasswordChange] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;

    async function initializeAuthForm(): Promise<void> {
      // Always check the server first — localStorage flags can be stale
      // (e.g. tokens from a previous install attempt).  The server's
      // /api/auth/status is the source of truth for requires_password_change.
      try {
        const response = await fetch(apiUrl("/api/auth/status"));
        if (!response.ok) throw new Error("Failed to load auth status.");
        const result = (await response.json()) as AuthStatusResponse;
        if (!canceled) {
          setInitialized(result.initialized);
          setRequiresPasswordChange(result.requires_password_change);

          // Redirect between login ↔ change-password based on server state
          if (mode === "login" && result.requires_password_change) {
            navigate({ to: "/change-password" });
            return;
          }
          if (mode === "change-password" && !result.requires_password_change && !mustChangePassword()) {
            navigate({ to: "/login" });
            return;
          }

          // On login page, if user already has a valid session and no
          // password change is required, skip straight to the app.
          if (isLoginMode && !result.requires_password_change) {
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

  // Seed password from bootstrap credentials injected into HTML by web CLI.
  useEffect(() => {
    function loadBootstrap() {
      const bootstrap = window.__UNSLOTH_BOOTSTRAP__;
      if (bootstrap && !isLoginMode && !password) {
        setPassword(bootstrap.password);
      }
    }
    loadBootstrap();
  }, []);

  const blockedByState =
    initialized === false ||
    (mode === "login" && requiresPasswordChange) ||
    (mode === "change-password" && !requiresPasswordChange && !mustChangePassword());

  let helperText: string | null = null;
  if (initialized === false) {
    helperText = "Auth is still bootstrapping the default admin account.";
  } else if (isLoginMode && requiresPasswordChange) {
    helperText = "Sign in once with the seeded credentials to change the password.";
  } else if (!isLoginMode && !requiresPasswordChange && !mustChangePassword()) {
    helperText = "Password already updated. Use the login screen.";
  }
  const title = isLoginMode ? "Welcome back" : "Setup your account";
  const subtitle = isLoginMode  
    ? "Sign in with your password."
    : "Choose a new password";
  const submitLabel = isLoginMode ? "Login" : "Change password";
  const showSwitchLink = !isLoginMode;
  const switchText = "Password already setup? ";
  const switchLinkTo = "/login";
  const switchLinkText = "Back to login";
  const currentPassword = password || window.__UNSLOTH_BOOTSTRAP__?.password || "";
  const invalidChangePasswordForm =
    !isLoginMode &&
    (newPassword.length < 8 || newPassword !== confirmPassword || currentPassword === newPassword);
  const showPasswordMismatchWarning =
    !isLoginMode &&
    newPassword.length > 0 &&
    confirmPassword.length > 0 &&
    newPassword !== confirmPassword;

  async function handleSubmit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!isLoginMode) {
      if (!currentPassword) {
        setError("Unable to initialize setup. Reload the page and try again.");
        return;
      }
      if (newPassword.length < 8) {
        setError("New password must be at least 8 characters.");
        return;
      }
      if (newPassword !== confirmPassword) {
        setError("Passwords do not match.");
        return;
      }
      if (currentPassword === newPassword) {
        setError("New password must be different from your current password.");
        return;
      }
    }

    setLoading(true);
    try {
      let token: TokenResponse;

      if (isLoginMode) {
        token = await loginWithPassword(username, password);
      } else {
        let accessToken = getAuthToken();

        if (hasRefreshToken()) {
          const refreshed = await refreshSession();
          accessToken = getAuthToken();
          if (!refreshed) {
            clearAuthTokens();
            accessToken = null;
          }
        }

        if (!accessToken) {
          const bootstrapToken = await loginWithPassword(username, currentPassword);
          storeAuthTokens(
            bootstrapToken.access_token,
            bootstrapToken.refresh_token,
            bootstrapToken.must_change_password,
          );
          setMustChangePassword(bootstrapToken.must_change_password);
          accessToken = bootstrapToken.access_token;
        }

        const response = await fetch(apiUrl("/api/auth/change-password"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({
            current_password: currentPassword,
            new_password: newPassword,
          }),
        });

        if (!response.ok) {
          let message = "Password update failed.";
          const errorPayload = (await response
            .json()
            .catch(() => null)) as { detail?: string } | null;
          if (errorPayload?.detail) message = errorPayload.detail;
          throw new Error(message);
        }

        token = (await response.json()) as TokenResponse;
      }

      if (!isLoginMode) {
        resetOnboardingDone();
        setRequiresPasswordChange(false);
        setMustChangePassword(false);
      } else {
        setMustChangePassword(token.must_change_password);
      }
      storeAuthTokens(
        token.access_token,
        token.refresh_token,
        token.must_change_password,
      );
      navigate({ to: getPostAuthRoute() });
    } catch (err: unknown) {
      let msg = err instanceof Error ? err.message : "Auth failed.";
      if (msg.includes("unsloth studio reset-password") && usePlatformStore.getState().deviceType === "windows") {
        msg = msg.replace(
          "unsloth studio reset-password",
          ".\\unsloth_studio\\Scripts\\unsloth.exe studio reset-password",
        );
      }
      setError(msg);
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
        {isLoginMode && (
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? "text" : "password"}
                className="pr-10"
                autoComplete="current-password"
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
          </div>
        )}

        {!isLoginMode && (
          <>
            <div className="space-y-2">
              <Label htmlFor="new-password">New password</Label>
              <div className="relative">
                <Input
                  id="new-password"
                  type={showPassword ? "text" : "password"}
                  className="pr-10"
                  autoComplete="new-password"
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
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
            </div>
            <div className="space-y-2">
              <Label htmlFor="confirm-password">Confirm password</Label>
              <Input
                id="confirm-password"
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                minLength={8}
                required
              />
            </div>
            <p
              className={`min-h-4 text-xs ${
                showPasswordMismatchWarning ? "text-destructive" : "text-muted-foreground"
              }`}
              aria-live="polite"
            >
              {showPasswordMismatchWarning
                ? "Please ensure passwords match."
                : "Must be at least 8 characters."}
            </p>
          </>
        )}

        {helperText && (
          <p className="text-center text-sm text-amber-600">{helperText}</p>
        )}
        {error && <p className="text-center text-sm text-destructive">{error}</p>}

        <Button
          type="submit"
          className="w-full"
          disabled={
            loading ||
            statusLoading ||
            blockedByState ||
            (isLoginMode && password.length < 8) ||
            invalidChangePasswordForm
          }
        >
          {loading ? "Please wait..." : submitLabel}
        </Button>
      </form>

      {showSwitchLink && (
        <p className="text-center text-sm text-muted-foreground">
          {switchText}
          <Link to={switchLinkTo} className="text-primary hover:underline">
            {switchLinkText}
          </Link>
        </p>
      )}
    </div>
  );
}
