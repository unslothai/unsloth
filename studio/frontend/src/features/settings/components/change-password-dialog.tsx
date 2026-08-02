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
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  getAuthToken,
  refreshSession,
  setMustChangePassword,
  storeAuthTokens,
} from "@/features/auth";
import { useT } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { type FormEvent, useState } from "react";

const MIN_PASSWORD_LENGTH = 8;
const WRONG_CURRENT_PASSWORD_DETAIL = "Current password is incorrect";

type T = ReturnType<typeof useT>;

function stringField(payload: Record<string, unknown>, key: string): string {
  const value = payload[key];
  return typeof value === "string" ? value : "";
}

function booleanField(payload: Record<string, unknown>, key: string): boolean {
  return payload[key] === true;
}

function changePasswordBody(
  currentPassword: string,
  nextPassword: string,
): string {
  return JSON.stringify(
    Object.fromEntries([
      ["current_password", currentPassword],
      ["new_password", nextPassword],
    ]),
  );
}

function hasStartedTooShortPassword(value: string): boolean {
  return value.length > 0 && value.length < MIN_PASSWORD_LENGTH;
}

function hasReusablePassword(currentPassword: string, nextPassword: string) {
  return (
    currentPassword.length >= MIN_PASSWORD_LENGTH &&
    nextPassword.length >= MIN_PASSWORD_LENGTH &&
    currentPassword === nextPassword
  );
}

function passwordValidationMessage(
  t: T,
  currentPassword: string,
  nextPassword: string,
  confirmPassword: string,
): string {
  if (currentPassword.length < MIN_PASSWORD_LENGTH) {
    return t("settings.general.passwordDialog.currentTooShort", {
      minLength: MIN_PASSWORD_LENGTH,
    });
  }
  if (nextPassword.length < MIN_PASSWORD_LENGTH) {
    return t("settings.general.passwordDialog.newTooShort", {
      minLength: MIN_PASSWORD_LENGTH,
    });
  }
  if (/\s/.test(nextPassword)) {
    return t("settings.general.passwordDialog.newHasSpaces");
  }
  if (nextPassword !== confirmPassword) {
    return t("settings.general.passwordDialog.mismatch");
  }
  if (currentPassword === nextPassword) {
    return t("settings.general.passwordDialog.samePassword");
  }
  return "";
}

async function unauthorizedDetail(response: Response): Promise<string | null> {
  if (response.status !== 401) {
    return null;
  }
  const payload = (await response
    .clone()
    .json()
    .catch(() => null)) as {
    detail?: string;
  } | null;
  return payload?.detail ?? null;
}

function postChangePassword(
  currentPassword: string,
  nextPassword: string,
): Promise<Response> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(apiUrl("/api/auth/change-password"), {
    method: "POST",
    headers,
    body: changePasswordBody(currentPassword, nextPassword),
  });
}

async function requestPasswordChange(
  currentPassword: string,
  nextPassword: string,
): Promise<Record<string, unknown>> {
  let response = await postChangePassword(currentPassword, nextPassword);
  const detail = await unauthorizedDetail(response);
  if (response.status === 401 && detail !== WRONG_CURRENT_PASSWORD_DETAIL) {
    // Retry token/session 401s, but never turn the endpoint's
    // "wrong current password" validation into a session refresh/logout.
    if (await refreshSession()) {
      response = await postChangePassword(currentPassword, nextPassword);
    }
  }
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(payload?.detail || "");
  }
  return (await response.json()) as Record<string, unknown>;
}

/**
 * Change the signed-in account's password from Settings, reusing the existing
 * POST /api/auth/change-password endpoint. The forced first-login flow lives at
 * /change-password and bounces non-forced users to /login, so day-to-day changes
 * need their own self-contained entry point here.
 */
export function ChangePasswordDialog() {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const reset = () => {
    setCurrent("");
    setNext("");
    setConfirm("");
  };

  const currentTooShort = hasStartedTooShortPassword(current);
  const nextTooShort = hasStartedTooShortPassword(next);
  const nextHasSpaces = /\s/.test(next);
  const mismatch = confirm.length > 0 && next !== confirm;
  const samePassword = hasReusablePassword(current, next);
  const validationMessage = passwordValidationMessage(
    t,
    current,
    next,
    confirm,
  );
  const disabled = submitting || Boolean(validationMessage);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (validationMessage) {
      toast.error(validationMessage);
      return;
    }
    setSubmitting(true);
    try {
      const data = await requestPasswordChange(current, next);
      const accessToken = stringField(data, "access_token");
      const refreshToken = stringField(data, "refresh_token");
      if (!(accessToken && refreshToken)) {
        throw new Error(t("settings.general.passwordDialog.updateFailed"));
      }
      // The endpoint rotates the JWT secret and returns fresh tokens.
      storeAuthTokens(accessToken, refreshToken);
      setMustChangePassword(booleanField(data, "must_change_password"));
      toast.success(t("settings.general.passwordDialog.updated"));
      reset();
      setOpen(false);
    } catch (err) {
      toast.error(
        err instanceof Error && err.message
          ? err.message
          : t("settings.general.passwordDialog.updateFailed"),
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (submitting && !o) {
          return;
        }
        setOpen(o);
        if (!o) {
          reset();
        }
      }}
    >
      <DialogTrigger asChild={true}>
        <Button variant="outline" size="sm" className="h-8">
          {t("settings.general.passwordDialog.trigger")}
        </Button>
      </DialogTrigger>
      <DialogContent
        className="sm:max-w-sm"
        showCloseButton={!submitting}
        onEscapeKeyDown={(event) => {
          if (submitting) {
            event.preventDefault();
          }
        }}
        onInteractOutside={(event) => {
          if (submitting) {
            event.preventDefault();
          }
        }}
      >
        <form onSubmit={submit}>
          <DialogHeader>
            <DialogTitle>
              {t("settings.general.passwordDialog.title")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.general.passwordDialog.description", {
                minLength: MIN_PASSWORD_LENGTH,
              })}
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4 space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="cp-current">
                {t("settings.general.passwordDialog.currentPassword")}
              </Label>
              <Input
                id="cp-current"
                type="password"
                autoComplete="current-password"
                value={current}
                onChange={(e) => setCurrent(e.target.value)}
                minLength={MIN_PASSWORD_LENGTH}
                disabled={submitting}
              />
              {currentTooShort ? (
                <p className="text-xs text-destructive" aria-live="polite">
                  {t("settings.general.passwordDialog.currentTooShort", {
                    minLength: MIN_PASSWORD_LENGTH,
                  })}
                </p>
              ) : null}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cp-new">
                {t("settings.general.passwordDialog.newPassword")}
              </Label>
              <Input
                id="cp-new"
                type="password"
                autoComplete="new-password"
                value={next}
                onChange={(e) => setNext(e.target.value)}
                minLength={MIN_PASSWORD_LENGTH}
                disabled={submitting}
              />
              {nextTooShort || nextHasSpaces || samePassword ? (
                <p className="text-xs text-destructive" aria-live="polite">
                  {nextTooShort
                    ? t("settings.general.passwordDialog.newTooShort", {
                        minLength: MIN_PASSWORD_LENGTH,
                      })
                    : nextHasSpaces
                      ? t("settings.general.passwordDialog.newHasSpaces")
                      : t("settings.general.passwordDialog.samePassword")}
                </p>
              ) : null}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cp-confirm">
                {t("settings.general.passwordDialog.confirmPassword")}
              </Label>
              <Input
                id="cp-confirm"
                type="password"
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                minLength={MIN_PASSWORD_LENGTH}
                disabled={submitting}
              />
              {mismatch ? (
                <p className="text-xs text-destructive" aria-live="polite">
                  {t("settings.general.passwordDialog.mismatch")}
                </p>
              ) : null}
            </div>
          </div>
          <DialogFooter className="mt-5">
            <Button type="submit" disabled={disabled}>
              {submitting
                ? t("settings.general.passwordDialog.updating")
                : t("settings.general.passwordDialog.update")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
