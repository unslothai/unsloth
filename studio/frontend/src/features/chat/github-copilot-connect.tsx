// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard, copyToClipboardFrom } from "@/lib/copy-to-clipboard";
import { openExternalLink } from "@/lib/open-link";
import { useEffect, useRef, useState } from "react";
import {
  cancelGitHubCopilot,
  disconnectGitHubCopilot,
  pollGitHubCopilot,
  startGitHubCopilot,
  type GitHubCopilotFlow,
  type ProviderAuthStatus,
} from "./api/providers-api";
import { SubscriptionConnectionSection } from "./subscription-connection-section";

const GITHUB_DEVICE_URL = "https://github.com/login/device";
type DeviceFlow = GitHubCopilotFlow & { providerId: string | null };

export function GitHubCopilotConnect({
  providerId,
  authStatus,
  chatEnabled,
  onChanged,
  onStartChat,
}: {
  providerId: string | null;
  authStatus?: ProviderAuthStatus;
  chatEnabled: boolean;
  onChanged: (id: string, connected: boolean) => Promise<void>;
  onStartChat: () => void;
}) {
  const [flow, setFlow] = useState<DeviceFlow | null>(null);
  const [busy, setBusy] = useState<"starting" | "cancelling" | "disconnecting" | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const connected = authStatus === "connected";
  const expired = authStatus === "reauthorization_required";
  const generation = useRef(0);
  const signInTab = useRef<Window | null>(null);
  const pendingFlow = useRef<DeviceFlow | null>(null);
  const previousProviderId = useRef(providerId);
  const changed = useRef(onChanged);
  changed.current = onChanged;

  useEffect(() => {
    if (previousProviderId.current && previousProviderId.current !== providerId) {
      generation.current += 1;
      setFlow(null);
      setBusy(null);
      setLoadingModels(false);
      setCancelled(false);
      setError("");
    }
    previousProviderId.current = providerId;
  }, [providerId]);

  useEffect(() => () => {
    generation.current += 1;
    signInTab.current?.close();
    const attempt = pendingFlow.current;
    if (attempt) {
      void cancelGitHubCopilot(attempt.providerId, attempt.flow_id).catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (!flow) return;
    const { providerId: id, flow_id, interval, expires_in } = flow;
    const expires = Date.now() + expires_in * 1000;
    const request = generation.current;
    let active = true;
    const isCurrent = () => active && request === generation.current;
    let timer = setTimeout(poll, interval * 1000);
    async function poll() {
      try {
        if (Date.now() >= expires) throw new Error("The code expired. Sign in again for a new one.");
        const result = await pollGitHubCopilot(id, flow_id);
        if (!isCurrent()) return;
        if (result.status === "pending") {
          timer = setTimeout(poll, Math.max(5, result.interval ?? interval) * 1000);
          return;
        }
        if (result.status === "expired") throw new Error("The code expired. Sign in again for a new one.");
        if (result.status === "denied") throw new Error("GitHub sign-in was declined. You can try again.");
        const connectedId = result.provider_id ?? id;
        if (!connectedId) throw new Error("Studio did not return the new connection.");
        pendingFlow.current = null;
        setLoadingModels(true);
        await changed.current(connectedId, true);
      } catch (err) {
        if (isCurrent()) setError(err instanceof Error ? err.message : "Could not connect to GitHub.");
      }
      if (isCurrent()) {
        pendingFlow.current = null;
        setLoadingModels(false);
        setFlow(null);
      }
    }
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [flow]);

  async function start() {
    const request = ++generation.current;
    setBusy("starting");
    setCancelled(false);
    setCopied(false);
    setError("");
    const pending = startGitHubCopilot(providerId).then((next) => {
      if (request !== generation.current) {
        void cancelGitHubCopilot(providerId, next.flow_id).catch(() => {});
        return null;
      }
      return { ...next, providerId };
    });
    // Start both operations during the original click so Safari allows them.
    const copy = copyToClipboardFrom(async () => (await pending)?.user_code ?? "");
    const tab = isTauri ? null : window.open("about:blank", "_blank");
    if (tab) tab.opener = null;
    signInTab.current = tab;
    try {
      const next = await pending;
      if (!next) {
        tab?.close();
        return;
      }
      pendingFlow.current = next;
      setFlow(next);
      const didCopy = await copy;
      if (request !== generation.current) {
        tab?.close();
        return;
      }
      setCopied(didCopy);
      if (tab && !tab.closed) tab.location.replace(GITHUB_DEVICE_URL);
      else if (isTauri) openExternalLink(GITHUB_DEVICE_URL);
    } catch (err) {
      tab?.close();
      if (request === generation.current) {
        setError(err instanceof Error ? err.message : "Could not start GitHub sign-in.");
      }
    } finally {
      if (request === generation.current) setBusy(null);
    }
  }

  async function cancel() {
    if (!flow) return;
    const request = ++generation.current;
    setFlow(null);
    pendingFlow.current = null;
    signInTab.current?.close();
    signInTab.current = null;
    setBusy("cancelling");
    setError("");
    try {
      const completed = await cancelGitHubCopilot(flow.providerId, flow.flow_id);
      if (request !== generation.current) return;
      if (completed) {
        setLoadingModels(true);
        await changed.current(completed.provider_id, true);
      } else {
        setCancelled(true);
      }
    } catch (err) {
      if (request === generation.current) {
        setError(err instanceof Error ? err.message : "Could not cancel GitHub sign-in.");
      }
    } finally {
      if (request === generation.current) {
        setLoadingModels(false);
        setBusy(null);
      }
    }
  }

  async function disconnect() {
    if (!providerId) return;
    const request = ++generation.current;
    setBusy("disconnecting");
    setError("");
    try {
      await disconnectGitHubCopilot(providerId);
      if (request === generation.current) await changed.current(providerId, false);
    } catch (err) {
      if (request === generation.current) {
        setError(err instanceof Error ? err.message : "Could not disconnect GitHub.");
      }
    } finally {
      if (request === generation.current) setBusy(null);
    }
  }

  return (
    <SubscriptionConnectionSection
      title="GitHub Copilot subscription"
      description={
        connected
          ? "Connected securely on this Unsloth installation."
          : expired
            ? "Your saved authorization is no longer valid. Reconnect to continue."
            : "Authorize in your system browser with a GitHub device code. Tokens never enter browser storage."
      }
    >
      {flow && !loadingModels ? (
        <div data-reload-snapshot-sensitive className="space-y-2 text-sm">
          <p>{copied ? "Paste the copied code on GitHub." : "Enter this code on GitHub:"}</p>
          <code aria-label="GitHub device code" className="block w-fit select-all rounded-full bg-muted px-3 py-2 font-mono text-base">
            {flow.user_code}
          </code>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void copyToClipboard(flow.user_code).then(setCopied)}
          >
            {copied ? "Copied" : "Copy code"}
          </Button>
        </div>
      ) : null}
      {loadingModels ? (
        <div role="status" className="flex items-center gap-2 text-xs text-muted-foreground">
          <Spinner className="size-3.5" />
          Loading Copilot models…
        </div>
      ) : null}
      {cancelled ? (
        <p role="status" className="text-xs text-muted-foreground">
          Sign-in cancelled. Close the GitHub tab if it’s still open.
        </p>
      ) : null}
      {error ? <p role="alert" className="text-xs text-destructive">{error}</p> : null}
      {!loadingModels ? (
        <div className="flex flex-wrap gap-2">
          {connected ? (
            <>
              {chatEnabled ? (
                <Button type="button" size="sm" variant="dark" disabled={busy !== null} onClick={onStartChat}>
                  Start chat
                </Button>
              ) : null}
              <Button type="button" size="sm" variant="outline" disabled={busy !== null} onClick={() => void disconnect()}>
                Disconnect locally
              </Button>
            </>
          ) : flow ? (
            <>
              <Button asChild size="sm" variant="dark">
                <a
                  href={GITHUB_DEVICE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(event) => {
                    if (isTauri) {
                      event.preventDefault();
                      openExternalLink(GITHUB_DEVICE_URL);
                    }
                  }}
                >
                  Open GitHub
                </a>
              </Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => void cancel()}>
                Cancel sign-in
              </Button>
            </>
          ) : (
            <Button type="button" size="sm" variant="dark" disabled={busy !== null} onClick={() => void start()}>
              {busy ? <Spinner className="size-3.5" data-icon="inline-start" /> : null}
              {busy === "starting"
                ? "Opening GitHub…"
                : busy === "cancelling"
                  ? "Cancelling sign-in…"
                  : expired
                    ? "Reconnect in browser"
                    : "Connect in browser"}
            </Button>
          )}
        </div>
      ) : null}
    </SubscriptionConnectionSection>
  );
}
