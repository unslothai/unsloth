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

import { useCallback, useEffect, useRef, useState } from "react";
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
  const [deviceCode, setDeviceCode] = useState<string | null>(null);
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
    setDeviceCode(null);
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    // Track the specific backend error inside the closure so the
    // generic fallback message does not overwrite it: setError is
    // async and reading `error` after `setError(event.message)` would
    // still see the stale pre-stream value.
    let lastStreamError: string | null = null;
    try {
      let lastOk: boolean | undefined;
      for await (const event of streamCodexDeviceLogin(
        controller.signal,
      ) as AsyncGenerator<CodexLoginEvent>) {
        if (event.type === "device_url" && event.url) {
          setDeviceUrl(event.url);
          // Do NOT auto-open the verification URL with `window.open`.
          // The click handler that started this flow has already
          // awaited an SSE event, so the call is no longer in a user
          // gesture and most browsers (Firefox, Safari, Chrome with
          // strict popup settings) will silently block the popup.
          // The URL is rendered as a prominent link below so the
          // user can open it in one click without depending on the
          // popup heuristic.
        } else if (event.type === "device_code" && event.code) {
          setDeviceCode(event.code);
        } else if (event.type === "log" && event.line) {
          setLogs((prev) => [...prev, event.line as string]);
        } else if (event.type === "error" && event.message) {
          lastStreamError = event.message;
          setError(event.message);
        } else if (event.type === "done") {
          lastOk = event.ok;
        }
      }
      if (lastOk) {
        onLoggedIn?.();
      } else if (!lastStreamError) {
        setError("Codex login did not complete -- see log for details.");
      }
    } catch (exc) {
      if ((exc as { name?: string } | null)?.name !== "AbortError") {
        setError(String((exc as Error)?.message ?? exc));
      }
    } finally {
      setBusy(false);
    }
  }, [busy, onLoggedIn]);

  // Abort the in-flight SSE stream on unmount so the underlying
  // `codex login --device-auth` subprocess does not keep streaming
  // (and consuming a device-auth session) after the dialog closes.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return (
    <div className="space-y-2">
      <Button type="button" disabled={busy} onClick={startLogin}>
        {busy ? "Signing in to Codex…" : "Sign in to Codex"}
      </Button>
      {deviceUrl && (
        <div className="space-y-1">
          <Button
            type="button"
            variant="outline"
            size="sm"
            asChild
          >
            {/* Opens via a real anchor click so popup blockers cannot
                interfere -- the popup-block path used to apply when
                `window.open` was triggered from inside an awaited
                event handler instead of a fresh user gesture. */}
            <a
              href={deviceUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open verification page
            </a>
          </Button>
          <p className="break-all text-[11px] text-muted-foreground">
            Or copy: {deviceUrl}
          </p>
        </div>
      )}
      {deviceCode && (
        <p className="text-xs text-muted-foreground">
          One-time code:{" "}
          <code className="font-mono text-foreground">{deviceCode}</code>
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
