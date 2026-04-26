// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { beginRestart } from "@/lib/restart-state";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import {
  type ServerSslSettings,
  fetchServerSettings,
  restartServer,
  testCertificate,
  updateServerSettings,
} from "../api/server-settings";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  applyDiscoveredBackend,
  fetchBackendHealth,
  pollUntilBackendReady,
} from "../lib/restart-poll";

type CertMode = "self_signed" | "byo";

/**
 * Pick the radio default based on saved settings. When SSL has never
 * been configured (the fresh-install case) we land on self-signed so
 * toggling "Enable TLS" on shows the recommended option pre-selected.
 * BYO is only the default when the user has already saved a non-self-
 * signed configuration.
 */
function defaultCertMode(settings: ServerSslSettings): CertMode {
  if (settings.ssl_enabled && !settings.ssl_self_signed) {
    return "byo";
  }
  return "self_signed";
}

function describeActiveSource(settings: ServerSslSettings | null): string {
  if (!settings) {
    return "";
  }
  const httpsActive = settings.active_scheme === "https";
  switch (settings.active_source) {
    case "cli_no_ssl":
      return "Running in plain HTTP because --no-ssl was passed. Saved settings will resume on next restart without that flag.";
    case "cli_paths":
      return "TLS is forced on by --ssl-certfile / --ssl-keyfile CLI flags. Saved settings below take over when the flags are dropped.";
    case "cli_self_signed":
      return "TLS is forced on by --ssl-self-signed. Saved settings below take over when the flag is dropped — toggle Enable TLS and save to make this persistent.";
    case "env_paths":
      return "TLS is forced on by UNSLOTH_SSL_CERTFILE / UNSLOTH_SSL_KEYFILE env vars. Saved settings take over when those are unset.";
    case "env_self_signed":
      return "TLS is forced on by UNSLOTH_SSL_SELF_SIGNED env var. Saved settings take over when it's unset.";
    case "db_paths":
    case "db_self_signed":
      return httpsActive
        ? "TLS is enabled from your saved settings."
        : "Plain HTTP is currently in use.";
    default:
      return httpsActive
        ? "TLS is currently enabled. Self-signed certs trigger a one-time browser warning on first visit."
        : "Plain HTTP is currently in use. Enable TLS below to encrypt logins and API traffic.";
  }
}

export function ServerTab() {
  const [settings, setSettings] = useState<ServerSslSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [sslEnabled, setSslEnabled] = useState(false);
  const [certMode, setCertMode] = useState<CertMode>("self_signed");
  const [certfile, setCertfile] = useState("");
  const [keyfile, setKeyfile] = useState("");

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedHint, setSavedHint] = useState<string | null>(null);

  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    error: string | null;
  } | null>(null);

  const [restarting, setRestarting] = useState(false);
  const [restartError, setRestartError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchServerSettings()
      .then((s) => {
        if (cancelled) {
          return;
        }
        setSettings(s);
        setSslEnabled(s.ssl_enabled);
        setCertMode(defaultCertMode(s));
        setCertfile(s.ssl_certfile ?? "");
        setKeyfile(s.ssl_keyfile ?? "");
      })
      .catch((e) => {
        if (cancelled) {
          return;
        }
        setLoadError(String(e?.message ?? e));
      })
      .finally(() => {
        if (cancelled) {
          return;
        }
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // When TLS is off, certMode and the path inputs are meaningless —
  // comparing them against the saved state would leave the form
  // permanently "dirty" after disabling TLS.
  const dirty =
    settings !== null &&
    (sslEnabled !== settings.ssl_enabled ||
      (sslEnabled &&
        ((certMode === "self_signed") !== settings.ssl_self_signed ||
          (certMode === "byo" && certfile !== (settings.ssl_certfile ?? "")) ||
          (certMode === "byo" && keyfile !== (settings.ssl_keyfile ?? "")))));

  const requiresRestart =
    settings !== null && sslEnabled !== (settings.active_scheme === "https");

  async function handleTest() {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testCertificate({
        ssl_certfile: certfile,
        ssl_keyfile: keyfile,
      });
      setTestResult(result);
    } catch (e) {
      setTestResult({ ok: false, error: String((e as Error)?.message ?? e) });
    } finally {
      setTesting(false);
    }
  }

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    setSavedHint(null);
    try {
      const next = await updateServerSettings({
        ssl_enabled: sslEnabled,
        ssl_self_signed: sslEnabled && certMode === "self_signed",
        ssl_certfile: sslEnabled && certMode === "byo" ? certfile : null,
        ssl_keyfile: sslEnabled && certMode === "byo" ? keyfile : null,
      });
      setSettings(next);
      setSavedHint("Saved. Restart the server to apply SSL changes.");
    } catch (e) {
      setSaveError(String((e as Error)?.message ?? e));
    } finally {
      setSaving(false);
    }
  }

  async function handleRestart() {
    if (!settings?.active_port) {
      return;
    }
    setRestartError(null);
    setSavedHint(null);
    setRestarting(true);

    // Capture the pre-restart instance_id so the poller can confirm the
    // process was actually replaced (vs. answering us from the same
    // uvicorn that hasn't shut down yet).
    const before = await fetchBackendHealth({ port: settings.active_port });
    const previousInstanceId = before?.instanceId ?? null;

    // Open the gate so authFetch parks (rather than detonating) every
    // request that races the restart. Released in `finally` so a thrown
    // error or early return can never leave the rest of the UI frozen.
    const releaseGate = beginRestart();
    try {
      const r = await restartServer();
      if (!r.supported) {
        setRestartError(
          "This platform cannot self-restart. Please restart `unsloth studio` manually.",
        );
        return;
      }

      const result = await pollUntilBackendReady({
        port: settings.active_port,
        timeoutMs: 90_000,
        previousInstanceId,
      });
      if (!result) {
        setRestartError(
          "The server did not come back online within 90 seconds. " +
            "It may still be starting — try refreshing in a moment. " +
            "If the UI is unreachable, run `unsloth studio --no-ssl` to recover.",
        );
        return;
      }
      const { reloaded } = applyDiscoveredBackend(result);
      if (reloaded) {
        // Page is bouncing to the new scheme; nothing else to do.
        return;
      }
      // Same-scheme restart: refresh the displayed settings against the
      // freshly booted server so "Active" reflects the new state.
      try {
        const next = await fetchServerSettings();
        setSettings(next);
        setSslEnabled(next.ssl_enabled);
        setCertMode(defaultCertMode(next));
        setCertfile(next.ssl_certfile ?? "");
        setKeyfile(next.ssl_keyfile ?? "");
      } catch {
        // Non-fatal — the next render will fall back to the previous
        // settings until the user navigates.
      }
      setSavedHint("Server restarted.");
    } catch (e) {
      setRestartError(String((e as Error)?.message ?? e));
    } finally {
      releaseGate();
      setRestarting(false);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-2">
        <h1 className="text-lg font-semibold font-heading">Server</h1>
        <p className="text-xs text-muted-foreground">
          Loading server settings…
        </p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex flex-col gap-3">
        <h1 className="text-lg font-semibold font-heading">Server</h1>
        <Alert variant="destructive">
          <AlertTitle>Could not load settings</AlertTitle>
          <AlertDescription>{loadError}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">Server</h1>
        <p className="text-xs text-muted-foreground">
          Configure how the Unsloth Studio backend serves the web UI and API.
          Changes apply on server restart.
        </p>
      </header>

      <Alert>
        <AlertTitle>
          Active: {settings?.active_scheme === "https" ? "HTTPS" : "HTTP"}
          {settings?.active_port ? ` on port ${settings.active_port}` : ""}
        </AlertTitle>
        <AlertDescription>
          {describeActiveSource(settings)}
        </AlertDescription>
      </Alert>

      <SettingsSection
        title="TLS / SSL"
        description="Encrypts the web UI, /api/*, and /v1/* endpoints. Off by default."
      >
        <SettingsRow
          label="Enable TLS"
          description="Turn on to serve HTTPS. Requires a server restart to take effect."
        >
          <Switch checked={sslEnabled} onCheckedChange={setSslEnabled} />
        </SettingsRow>

        {sslEnabled ? (
          <div className="flex flex-col gap-4 py-3">
            <RadioGroup
              value={certMode}
              onValueChange={(v) => setCertMode(v as CertMode)}
              className="flex flex-col gap-3"
            >
              <div className="flex items-start gap-3">
                <RadioGroupItem
                  value="self_signed"
                  id="ssl-self-signed"
                  className="mt-0.5"
                />
                <Label
                  htmlFor="ssl-self-signed"
                  className="flex flex-col items-start gap-0.5 cursor-pointer text-left"
                >
                  <span className="text-sm font-medium">
                    Self-signed certificate
                  </span>
                  <span className="text-xs text-muted-foreground leading-snug">
                    Studio generates and reuses a cert in{" "}
                    <code>~/.unsloth/studio/ssl/</code>. Browsers will show a
                    one-time warning you can click through.
                  </span>
                </Label>
              </div>
              <div className="flex items-start gap-3">
                <RadioGroupItem value="byo" id="ssl-byo" className="mt-0.5" />
                <Label
                  htmlFor="ssl-byo"
                  className="flex flex-col items-start gap-0.5 cursor-pointer text-left"
                >
                  <span className="text-sm font-medium">
                    Provide certificate files
                  </span>
                  <span className="text-xs text-muted-foreground leading-snug">
                    Use a real cert (e.g. from Let's Encrypt) by pointing to PEM
                    files.
                  </span>
                </Label>
              </div>
            </RadioGroup>

            {certMode === "byo" ? (
              <div className="flex flex-col gap-3 pl-7">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="ssl-certfile" className="text-xs">
                    Certificate file (PEM)
                  </Label>
                  <Input
                    id="ssl-certfile"
                    placeholder="/etc/ssl/certs/studio.pem"
                    value={certfile}
                    onChange={(e) => {
                      setCertfile(e.target.value);
                      setTestResult(null);
                    }}
                    className="h-8 font-mono text-xs"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="ssl-keyfile" className="text-xs">
                    Private key file (PEM)
                  </Label>
                  <Input
                    id="ssl-keyfile"
                    placeholder="/etc/ssl/private/studio.key"
                    value={keyfile}
                    onChange={(e) => {
                      setKeyfile(e.target.value);
                      setTestResult(null);
                    }}
                    className="h-8 font-mono text-xs"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    disabled={testing || !certfile || !keyfile}
                    onClick={handleTest}
                  >
                    {testing ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      "Test certificate"
                    )}
                  </Button>
                  {testResult ? (
                    <span
                      className={
                        testResult.ok
                          ? "text-xs text-emerald-500"
                          : "text-xs text-destructive"
                      }
                    >
                      {testResult.ok
                        ? "Valid certificate/key pair."
                        : `Error: ${testResult.error ?? "invalid"}`}
                    </span>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </SettingsSection>

      {saveError ? (
        <Alert variant="destructive">
          <AlertTitle>Could not save</AlertTitle>
          <AlertDescription>{saveError}</AlertDescription>
        </Alert>
      ) : null}

      {restartError ? (
        <Alert variant="destructive">
          <AlertTitle>Restart did not complete</AlertTitle>
          <AlertDescription>{restartError}</AlertDescription>
        </Alert>
      ) : null}

      <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-4">
        <span className="text-xs text-muted-foreground">
          {savedHint ??
            (dirty
              ? "Unsaved changes — save before restarting to apply them."
              : requiresRestart
                ? "Saved settings differ from the running server. Restart to apply."
                : "Settings match the running server.")}
        </span>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            disabled={saving || !dirty}
          >
            {saving ? <Loader2 className="size-3.5 animate-spin" /> : "Save"}
          </Button>
          <Button
            size="sm"
            onClick={handleRestart}
            disabled={restarting || !settings?.restart_supported}
            title={
              dirty
                ? "Restart will use your last saved settings. Click Save first to include unsaved changes."
                : "Restart the server now."
            }
          >
            {restarting ? (
              <>
                <Loader2 className="size-3.5 animate-spin" />
                <span className="ml-1.5">Restarting…</span>
              </>
            ) : (
              "Restart server"
            )}
          </Button>
        </div>
      </div>

      <p className="text-xs text-foreground">
        <span className="font-semibold text-emerald-500">Note:</span>{" "}
        HTTP and HTTPS are separate browser origins, so your chat history,
        theme, and login session won't follow you across the switch.
        Nothing is deleted — switching back makes them reappear — but on
        the new scheme you'll re-sign in and re-pick preferences.
      </p>

      <p className="text-xs text-muted-foreground">
        Locked out after a misconfigured cert? Run{" "}
        <code>unsloth studio --no-ssl</code> from a terminal to start the server
        in plain HTTP and recover.
      </p>
    </div>
  );
}
