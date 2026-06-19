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
import { authFetch } from "@/features/auth";
import {
  setMustChangePassword,
  storeAuthTokens,
} from "@/features/auth/session";
import { toast } from "@/lib/toast";
import { type FormEvent, useState } from "react";

type TokenResponse = {
  access_token: string;
  refresh_token: string;
  must_change_password?: boolean;
};

const MIN_PASSWORD_LENGTH = 8;

/**
 * Change the signed-in account's password from Settings, reusing the existing
 * POST /api/auth/change-password endpoint. The forced first-login flow lives at
 * /change-password and bounces non-forced users to /login, so day-to-day changes
 * need their own self-contained entry point here.
 */
export function ChangePasswordDialog() {
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

  const mismatch = confirm.length > 0 && next !== confirm;
  const disabled =
    submitting ||
    current.length < MIN_PASSWORD_LENGTH ||
    next.length < MIN_PASSWORD_LENGTH ||
    next !== confirm ||
    current === next;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (disabled) return;
    setSubmitting(true);
    try {
      // authFetch attaches the current access token and, on a 401 from an
      // expired token, refreshes the session and retries -- so a user who left
      // Studio open past the access-token lifetime can still change their
      // password instead of getting a spurious expired-token error.
      const response = await authFetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_password: current,
          new_password: next,
        }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(payload?.detail || "Password update failed.");
      }
      const data = (await response.json()) as TokenResponse;
      // The endpoint rotates the JWT secret and returns fresh tokens.
      storeAuthTokens(data.access_token, data.refresh_token);
      setMustChangePassword(data.must_change_password ?? false);
      toast.success("Password updated.");
      reset();
      setOpen(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Password update failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (!o) reset();
      }}
    >
      <DialogTrigger asChild={true}>
        <Button variant="outline" size="sm" className="h-8">
          Change password
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <form onSubmit={submit}>
          <DialogHeader>
            <DialogTitle>Change password</DialogTitle>
            <DialogDescription>
              Enter your current password and choose a new one (at least{" "}
              {MIN_PASSWORD_LENGTH} characters).
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4 space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="cp-current">Current password</Label>
              <Input
                id="cp-current"
                type="password"
                autoComplete="current-password"
                value={current}
                onChange={(e) => setCurrent(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cp-new">New password</Label>
              <Input
                id="cp-new"
                type="password"
                autoComplete="new-password"
                value={next}
                onChange={(e) => setNext(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="cp-confirm">Confirm new password</Label>
              <Input
                id="cp-confirm"
                type="password"
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
              />
              {mismatch ? (
                <p className="text-xs text-destructive">Passwords do not match.</p>
              ) : null}
            </div>
          </div>
          <DialogFooter className="mt-5">
            <Button type="submit" disabled={disabled}>
              {submitting ? "Updating..." : "Update password"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
