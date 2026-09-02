// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { apiUrl } from "@/lib/api-base";
import { Button } from "@/components/ui/button";
import { MascotImg } from "@/components/mascot-img";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Link, useNavigate } from "@tanstack/react-router";
import { Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactElement } from "react";
import type { SyntheticEvent } from "react";
import { noteAuthSessionReplaced, refreshSession } from "../api";
import { notifyAccountAuthenticated } from "../../../lib/account-transition.ts";
import {
  deadlineFromStatus,
  formatCountdown,
  hasExpired,
} from "../bootstrap-deadline";

// Bootstrap credentials injected into index.html by the backend (only present
// while default admin must_change_password is true)
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
  setMustChangePassword,
  storeAuthTokens,
} from "../session";

type AuthMode = "login" | "change-password";

type AuthStatusResponse = {
  initialized: boolean;
  default_username: string;
  requires_password_change: boolean;
  bootstrap_deadline_seconds?: number | null;
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

export function AuthForm({ mode }: AuthFormProps): ReactElement | null {
  const navigate = useNavigate();
  const isLoginMode = mode === "login";
  const [showPassword, setShowPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [username, setUsername] = useState(
    () => window.__UNSLOTH_BOOTSTRAP__?.username ?? "",
  );
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [initialized, setInitialized] = useState<boolean | null>(null);
  // Who the server says owns this installation, so the pre-accounts browser data
  // is attributed to them rather than to whoever signs in first.
  const [installationOwner, setInstallationOwner] = useState<string | null>(null);
  const [requiresPasswordChange, setRequiresPasswordChange] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deadlineAt, setDeadlineAt] = useState<number | null>(null);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  const reloadReadySent = useRef(false);

  useEffect(() => {
    if (deadlineAt === null) {
      return;
    }
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [deadlineAt]);

  useEffect(() => {
    let canceled = false;

    async function initializeAuthForm(): Promise<void> {
      // Always check the server first; localStorage flags can be stale (e.g.
      // tokens from a previous install). /api/auth/status is the source of
      // truth for requires_password_change.
      try {
        const response = await fetch(apiUrl("/api/auth/status"));
        if (!response.ok) throw new Error("Failed to load auth status.");
        const result = (await response.json()) as AuthStatusResponse;
        if (!canceled) {
          setInitialized(result.initialized);
          setInstallationOwner(result.default_username ?? null);
          setUsername((current) => current || result.default_username);
          const accountRequiresPasswordChange =
            hasAuthToken() && mustChangePassword();
          // requires_password_change is the OWNER's, and this page is reached by
          // every account. Only the seeded first boot may act on it, which is
          // the one case the credential to complete it is on the page: without
          // that guard an owner reset shut every managed account out of /login
          // until the owner had finished, since the form blocked and redirected
          // before a username could even be typed.
          const ownerBootstrapPending =
            result.requires_password_change &&
            Boolean(window.__UNSLOTH_BOOTSTRAP__?.password);
          const effectivePasswordChange =
            ownerBootstrapPending || accountRequiresPasswordChange;
          setRequiresPasswordChange(effectivePasswordChange);
          // One clock sample for both: nowMs is otherwise still the mount time
          // until the first tick, which adds the request duration to the figure
          // and renders a 0 from the server as "shuts down in 0 seconds".
          const sampledNow = Date.now();
          setNowMs(sampledNow);
          setDeadlineAt(
            deadlineFromStatus(result.bootstrap_deadline_seconds, sampledNow),
          );

          // Server truth wins; keep localStorage in sync both ways. Only from
          // the seeded boot, for the same reason: this flag describes whoever
          // holds the session, and the owner's is not theirs to inherit.
          if (ownerBootstrapPending && !mustChangePassword()) {
            setMustChangePassword(true);
          }

          // Redirect between login / change-password per server state
          if (mode === "login" && ownerBootstrapPending) {
            navigate({ to: "/change-password" });
            return;
          }
          if (mode === "change-password" && !effectivePasswordChange) {
            navigate({ to: "/login" });
            return;
          }

          // On login, skip to the app if a valid session exists and no
          // password change is required.
          if (isLoginMode && !ownerBootstrapPending) {
            if (hasRefreshToken()) {
              const refreshed = await refreshSession();
              if (refreshed) {
                if (!canceled) setStatusLoading(false);
                navigate({ to: getPostAuthRoute() });
                return;
              }
              // The failed refresh cleared local storage, but the flag was read
              // before it ran. Recompute, or a revoked session leaves the login
              // form disabled with nothing left to recompute it and the user
              // cannot enter their replacement setup code without a reload.
              if (!canceled) {
                setRequiresPasswordChange(
                  ownerBootstrapPending || mustChangePassword(),
                );
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
  }, [isLoginMode, mode, navigate]);

  useEffect(() => {
    if (statusLoading || reloadReadySent.current) return;
    reloadReadySent.current = true;
    window.dispatchEvent(new Event("unsloth:app-shell-ready"));
  }, [statusLoading]);

  // Seed password from bootstrap credentials injected into HTML by web CLI.
  useEffect(() => {
    function loadBootstrap() {
      const bootstrap = window.__UNSLOTH_BOOTSTRAP__;
      if (bootstrap && !isLoginMode && !password) {
        setPassword(bootstrap.password);
      }
    }
    loadBootstrap();
  }, [isLoginMode, password]);

  const blockedByState =
    initialized === false ||
    (mode === "login" && requiresPasswordChange) ||
    (mode === "change-password" && !requiresPasswordChange);

  let helperText: string | null = null;
  if (initialized === false) {
    helperText = "Auth is still bootstrapping the default admin account.";
  } else if (isLoginMode && requiresPasswordChange) {
    helperText = "Sign in once with the seeded credentials to change the password.";
  } else if (!isLoginMode && !requiresPasswordChange) {
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
  // On first boot the backend injects __UNSLOTH_BOOTSTRAP__ and we silently
  // reuse that password; the Current password input is only rendered for the
  // admin-forced must_change_password path where no bootstrap is available.
  const hasBootstrapPassword = Boolean(window.__UNSLOTH_BOOTSTRAP__?.password);
  const invalidChangePasswordForm =
    !isLoginMode &&
    (currentPassword.length < 8 ||
      newPassword.length < 8 ||
      /\s/.test(newPassword) ||
      newPassword !== confirmPassword ||
      currentPassword === newPassword);
  const showWhitespaceWarning = !isLoginMode && /\s/.test(newPassword);
  const showPasswordMismatchWarning =
    !isLoginMode &&
    newPassword.length > 0 &&
    confirmPassword.length > 0 &&
    newPassword !== confirmPassword;

  async function handleSubmit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!isLoginMode) {
      // Mirror the disable gate: Enter / autofill can bypass the button.
      if (currentPassword.length < 8) {
        setError(
          currentPassword
            ? "Current password must be at least 8 characters."
            : "Unable to initialize setup. Reload the page and try again.",
        );
        return;
      }
      if (newPassword.length < 8) {
        setError("New password must be at least 8 characters.");
        return;
      }
      if (/\s/.test(newPassword)) {
        setError("New password cannot contain spaces.");
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
        setRequiresPasswordChange(false);
        setMustChangePassword(false);
      } else {
        setMustChangePassword(token.must_change_password);
      }
      // Before the new session mounts: this browser's localStorage is origin-wide,
      // so an account that is not the one that left it there would otherwise read
      // the previous account's drafts, dictation history and provider metadata,
      // and have its legacy chats, chat settings and Hugging Face token migrated
      // into its own workspace.
      //
      // The login step ONLY. In change-password mode this form has remounted and
      // its username state comes from /api/auth/status.default_username, which is
      // the installation owner, not the account whose password is being changed:
      // notifying with that moved the browser marker to the owner and released
      // the quarantined legacy state into a managed user's live session. The
      // login that produced this session already recorded the right account.
      let accountChanged = false;
      if (isLoginMode) {
        accountChanged = notifyAccountAuthenticated(username, installationOwner);
        // A request still waiting on a 401 belongs to the account that sent it.
        if (accountChanged) noteAuthSessionReplaced();
      }
      storeAuthTokens(token.access_token, token.refresh_token);
      if (accountChanged) {
        // A different account on the same browser. Purging storage does not touch
        // anything already hydrated, and enumerating the stores that hold account
        // content is what kept missing one; a document load resets all of them,
        // including the Dexie handles whose databases the purge just deleted.
        try {
          window.location.replace(getPostAuthRoute());
          return;
        } catch {
          // No navigation (tests): fall through to the router.
        }
      }
      navigate({ to: getPostAuthRoute() });
    } catch (err: unknown) {
      // The backend returns the correct PATH-based command ("unsloth studio
      // reset-password"), which the installer puts on PATH on every platform.
      // Do NOT rewrite it to a relative Windows path like
      // ".\unsloth_studio\Scripts\unsloth.exe ..." -- that only resolves inside
      // the Unsloth home dir and fails with CommandNotFoundException elsewhere.
      // Show the backend message as-is.
      const msg = err instanceof Error ? err.message : "Auth failed.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  if (statusLoading && initialized === null && error === null) return null;

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1.5 text-center">
        <MascotImg
          src="Sloth emojis/large sloth wave.png"
          className="mx-auto mb-2 h-20 w-20 object-contain"
        />
        <h2 className="text-2xl font-semibold text-foreground">{title}</h2>
        <p className="text-muted-foreground">{subtitle}</p>
      </div>
      {/* Not a live region: it re-renders every second, so it would be read aloud on every tick. */}
      {deadlineAt !== null && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-center text-sm text-amber-600">
          {hasExpired(deadlineAt - nowMs) ? (
            <>
              This instance is shutting down: it was reachable on the network
              and its default password was never changed.
            </>
          ) : (
            <>
              This instance is reachable on the network and still uses its
              default password, so it shuts down in{" "}
              {formatCountdown(deadlineAt - nowMs)}. Setting a password here
              keeps it running.
            </>
          )}
        </p>
      )}
      <form className="space-y-5" onSubmit={handleSubmit}>
        {isLoginMode && (
          <>
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value.toLowerCase())}
                minLength={3}
                maxLength={64}
                pattern="[a-z0-9][a-z0-9._-]*"
                spellCheck={false}
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
          </>
        )}

        {!isLoginMode && (
          <>
            {!hasBootstrapPassword && (
              <div className="space-y-2">
                <Label htmlFor="current-password">Current password</Label>
                <div className="relative">
                  <Input
                    id="current-password"
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
            <div className="space-y-2">
              <Label htmlFor="new-password">New password</Label>
              <div className="relative">
                <Input
                  id="new-password"
                  type={showNewPassword ? "text" : "password"}
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
                  onClick={() => setShowNewPassword((prev) => !prev)}
                >
                  {showNewPassword ? (
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
                showWhitespaceWarning || showPasswordMismatchWarning
                  ? "text-destructive"
                  : "text-muted-foreground"
              }`}
              aria-live="polite"
            >
              {showWhitespaceWarning
                ? "New password cannot contain spaces."
                : showPasswordMismatchWarning
                  ? "Please ensure passwords match."
                  : "Must be at least 8 characters."}
            </p>
          </>
        )}

        {helperText && (
          <p className="text-center text-sm text-amber-600">{helperText}</p>
        )}
        {error && (
          <p className="text-center text-sm text-destructive [overflow-wrap:anywhere]">
            {error}
          </p>
        )}

        <Button
          type="submit"
          className="mx-auto flex w-fit px-4"
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
