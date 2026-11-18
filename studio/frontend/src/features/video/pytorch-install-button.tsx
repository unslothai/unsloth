// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { authFetch } from "@/features/auth";
import { useIsAccountOwner } from "@/features/auth/account-session";

type InstallStatus = {
  status: "idle" | "installing" | "failed" | "restart_required";
  message: string | null;
};
const endpoint = "/api/inference/video/pytorch-install";

async function readStatus(method: "GET" | "POST"): Promise<InstallStatus> {
  const response = await authFetch(endpoint, { method });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : "Could not contact the PyTorch installer. Try again.",
    );
  }
  return response.json();
}

export function PytorchInstallButton() {
  const isOwner = useIsAccountOwner();
  const [job, setJob] = useState<InstallStatus>({ status: "idle", message: null });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  // Resume progress when returning to this page or refreshing during installation.
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await readStatus("GET");
        if (!disposed) {
          setJob(next);
          setPollError(null);
        }
      } catch (cause) {
        if (!disposed) setPollError(cause instanceof Error ? cause.message : "Could not read installation progress.");
      } finally {
        if (!disposed) timer = setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, []);

  async function install() {
    setPending(true);
    setError(null);
    try {
      setJob(await readStatus("POST"));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start installation.");
    } finally {
      setPending(false);
    }
  }

  const installing = pending || job.status === "installing";
  return (
    <div className="flex max-w-sm flex-col items-center gap-3">
      {job.status !== "restart_required" && isOwner && (
        <Button onClick={() => void install()} disabled={installing}>
          {installing && <Spinner className="size-4" />}
          {installing ? "Installing PyTorch…" : job.status === "failed" ? "Retry PyTorch installation" : "Install PyTorch"}
        </Button>
      )}
      {!isOwner && job.status === "idle" && (
        <p>Ask the owner of this Unsloth installation to install PyTorch.</p>
      )}
      {job.message && <p role="status">{job.message}</p>}
      {(error || pollError) && <p role="alert">{error || pollError}</p>}
    </div>
  );
}
