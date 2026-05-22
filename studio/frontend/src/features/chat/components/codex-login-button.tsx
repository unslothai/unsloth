// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * "Sign in to Codex" button + streamed log surface.
 *
 * Renders in place of the regular API-key field in the provider config
 * dialog when ``/api/codex/status`` reports ``logged_in=false``. Click
 * fires POST ``/api/codex/login``, which spawns
 * ``codex auth login --device-auth`` server-side. The first SSE event
 * carries the verification URL -- as soon as it arrives we open it in
 * a new tab via ``window.open`` so the user doesn't have to copy-paste
 * a long URL out of a log pane.
 *
 * The button stays mounted while the CLI is exchanging the device
 * code: the streamed ``log`` events accumulate into the visible
 * progress area until the ``done`` event closes the stream. The
 * caller passes ``onLoggedIn`` so the dialog can refetch the status
 * probe and flip back into the "ready" state automatically.
 */

import { useCallback, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  streamCodexDeviceLogin,
  type CodexLoginEvent,
} from "../api/codex-api";

interface Props {
  /** Called when the device-auth flow finishes successfully so the
   * parent can re-probe ``/api/codex/status`` and switch UI states. */
  onLoggedIn?: () => void;
}

export function CodexLoginButton({ onLoggedIn }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [deviceUrl, setDeviceUrl] = useState<string | null>(null);
  // Track the active stream's abort controller so a second click
  // (or an unmount) tears the SSE reader down cleanly. Without this
  // the long-running login subprocess would keep streaming into a
  // detached component.
  const abortRef = useRef<AbortController | null>(null);

  const startLogin = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setLogs([]);
    setDeviceUrl(null);
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    try {
      let lastOk: boolean | undefined;
      for await (const event of streamCodexDeviceLogin(
        controller.signal,
      ) as AsyncGenerator<CodexLoginEvent>) {
        if (event.type === "device_url" && event.url) {
          setDeviceUrl(event.url);
          // Open the verification page eagerly so the user doesn't
          // have to copy the URL out of the log surface. ``noopener``
          // prevents the auth-tab from controlling the Studio window.
          try {
            window.open(event.url, "_blank", "noopener,noreferrer");
          } catch {
            // Ignore -- the URL is still visible in the log.
          }
        } else if (event.type === "log" && event.line) {
          setLogs((prev) => [...prev, event.line as string]);
        } else if (event.type === "error" && event.message) {
          setError(event.message);
        } else if (event.type === "done") {
          lastOk = event.ok;
        }
      }
      if (lastOk) {
        onLoggedIn?.();
      } else if (!error) {
        setError("Codex login did not complete -- see log for details.");
      }
    } catch (exc) {
      if ((exc as { name?: string } | null)?.name !== "AbortError") {
        setError(String((exc as Error)?.message ?? exc));
      }
    } finally {
      setBusy(false);
    }
  }, [busy, error, onLoggedIn]);

  return (
    <div className="space-y-2">
      <Button type="button" disabled={busy} onClick={startLogin}>
        {busy ? "Signing in to Codex…" : "Sign in to Codex"}
      </Button>
      {deviceUrl && (
        <p className="text-xs text-muted-foreground">
          Verification URL:{" "}
          <a
            href={deviceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            {deviceUrl}
          </a>
        </p>
      )}
      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}
      {logs.length > 0 && (
        <pre className="max-h-40 overflow-auto rounded bg-muted/50 p-2 text-[11px]">
          {logs.join("\n")}
        </pre>
      )}
    </div>
  );
}
