// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  cancelNamedTunnelSetup,
  confirmNamedTunnelDns,
  removeNamedTunnel,
  setupNamedTunnel,
  updateRemoteAccessConfig,
} from "@/features/settings/api/remote-access";
import {
  type NamedTunnelMethod,
  type RemoteAccessStatus,
  cloudflareLoginUrl,
} from "@/features/settings/api/remote-access-state";
import { useT } from "@/i18n";
import { openExternalLink } from "@/lib/open-link";
import { useEffect, useRef, useState } from "react";
import { SettingsRow } from "./settings-row";

export function namedTunnelErrorMessage(
  error: string,
  t: ReturnType<typeof useT>,
): string {
  switch (error) {
    case "invalid_hostname":
      return t("settings.namedTunnel.errors.hostname");
    case "token_required":
    case "invalid_tunnel_token":
      return t("settings.namedTunnel.errors.token");
    case "remove_configuration_first":
      return t("settings.namedTunnel.errors.removeFirst");
    case "named_tunnel_busy":
    case "stop_tunnel_first":
    case "operation_in_progress":
      return t("settings.namedTunnel.errors.busy");
    case "cloudflare_login_failed":
    case "cloudflare_login_required":
      return t("settings.namedTunnel.errors.login");
    case "command_timed_out":
    case "setup_timed_out":
      return t("settings.namedTunnel.errors.timeout");
    case "setup_cancelled":
      return t("settings.namedTunnel.errors.cancelled");
    case "dns_route_failed":
    case "dns_result_unrecognized":
      return t("settings.namedTunnel.errors.dns");
    case "dns_hostname_mismatch":
      return t("settings.namedTunnel.errors.zone");
    case "tunnel_delete_failed":
      return t("settings.namedTunnel.errors.remove");
    case "tunnel_still_active":
      return t("settings.namedTunnel.errors.stillActive");
    case "tunnel_status_unavailable":
      return t("settings.namedTunnel.errors.statusUnavailable");
    case "credentials_unreadable":
    case "configuration_invalid":
    case "tunnel_credentials_invalid":
      return t("settings.namedTunnel.errors.credentials");
    case "named_setup_required":
      return t("settings.namedTunnel.prepareHint");
    default:
      return t("settings.namedTunnel.errors.generic");
  }
}

export function NamedTunnelSettings({
  status,
  busy,
  perform,
}: {
  status: RemoteAccessStatus;
  busy: boolean;
  perform: (request: () => Promise<RemoteAccessStatus>) => Promise<boolean>;
}) {
  const t = useT();
  const configuration = status.named.configuration;
  const setup = status.named.setup;
  const [method, setMethod] = useState<NamedTunnelMethod>(
    configuration?.method ?? "automatic",
  );
  const [hostname, setHostname] = useState(configuration?.hostname ?? "");
  const [token, setToken] = useState("");
  const focusTarget = useRef<HTMLFieldSetElement>(null);
  const [removeOpen, setRemoveOpen] = useState(false);
  const [dismissedConflict, setDismissedConflict] = useState<string | null>(
    null,
  );
  const disabled = busy || !status.canConfigure;
  const authUrl = cloudflareLoginUrl(setup?.authUrl ?? null);

  // Reset secret drafts on rotation without remounting the focused controls.
  // biome-ignore lint/correctness/useExhaustiveDependencies: configurationId also resets the token on rotation
  useEffect(() => {
    setHostname(configuration?.hostname ?? "");
    setMethod(configuration?.method ?? "automatic");
    setToken("");
    setRemoveOpen(false);
  }, [
    configuration?.configurationId,
    configuration?.hostname,
    configuration?.method,
  ]);

  const restoreFocus = (event: Event) => {
    event.preventDefault();
    focusTarget.current?.focus();
  };

  const submit = async () => {
    const success = await perform(() =>
      method === "automatic"
        ? setupNamedTunnel(hostname)
        : updateRemoteAccessConfig({
            mode: "named",
            hostname,
            ...(token ? { token } : {}),
          }),
    );
    if (success) {
      setToken("");
    }
  };
  const cleanupHostname =
    setup?.dnsCleanupHostname ?? configuration?.dnsCleanupHostname;
  const conflictOpen =
    setup?.state === "dns_conflict" &&
    setup.active &&
    dismissedConflict !== setup.operationId;

  return (
    <fieldset
      ref={focusTarget}
      tabIndex={-1}
      aria-label={t("settings.namedTunnel.named")}
      className="min-w-0 border-t border-border/60 px-4 py-1"
    >
      <SettingsRow label={t("settings.namedTunnel.method")}>
        <Select
          value={method}
          disabled={disabled || configuration !== null}
          onValueChange={(value) => setMethod(value as NamedTunnelMethod)}
        >
          <SelectTrigger
            className="w-80 max-w-full"
            aria-label={t("settings.namedTunnel.method")}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="automatic">
              {t("settings.namedTunnel.automatic")}
            </SelectItem>
            <SelectItem value="token">
              {t("settings.namedTunnel.existing")}
            </SelectItem>
          </SelectContent>
        </Select>
      </SettingsRow>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <SettingsRow label={t("settings.namedTunnel.hostname")}>
          <Input
            aria-label={t("settings.namedTunnel.hostname")}
            placeholder="studio.example.com"
            value={hostname}
            onChange={(event) => setHostname(event.target.value)}
            disabled={disabled || configuration !== null}
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            required={true}
            className="w-64 max-w-full"
          />
        </SettingsRow>
        {method === "token" ? (
          <>
            <SettingsRow
              label={t("settings.namedTunnel.token")}
              description={
                configuration?.method === "token"
                  ? t("settings.namedTunnel.keepToken")
                  : undefined
              }
            >
              <Input
                aria-label={t("settings.namedTunnel.token")}
                type="password"
                value={token}
                onChange={(event) => setToken(event.target.value)}
                disabled={disabled}
                autoComplete="off"
                data-reload-snapshot-sensitive="true"
                required={configuration?.method !== "token"}
                className="w-64 max-w-full"
              />
            </SettingsRow>
            <p className="mb-3 break-words text-xs text-muted-foreground">
              {t("settings.namedTunnel.originHint", {
                origin: status.localOrigin ?? "—",
              })}
            </p>
          </>
        ) : (
          <p className="mb-3 text-xs text-muted-foreground">
            {t("settings.namedTunnel.prepareHint")}
          </p>
        )}
        <div className="flex flex-wrap justify-end gap-2 pb-3">
          {configuration ? (
            <AlertDialog open={removeOpen} onOpenChange={setRemoveOpen}>
              <AlertDialogTrigger asChild={true}>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={disabled}
                >
                  {t("settings.namedTunnel.remove")}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent onCloseAutoFocus={restoreFocus}>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    {t("settings.namedTunnel.removeTitle")}
                  </AlertDialogTitle>
                  <AlertDialogDescription className="break-words">
                    {t(
                      configuration.method === "automatic"
                        ? "settings.namedTunnel.removeAuto"
                        : "settings.namedTunnel.removeToken",
                      { hostname: configuration.hostname },
                    )}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
                  <AlertDialogAction
                    disabled={disabled}
                    onClick={() =>
                      void perform(() =>
                        removeNamedTunnel(configuration.configurationId),
                      )
                    }
                  >
                    {t("settings.namedTunnel.remove")}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          ) : null}
          {method === "automatic" &&
          !setup?.active &&
          (setup?.error === "cloudflare_login_failed" ||
            setup?.error === "cloudflare_login_required") ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={disabled}
              onClick={() =>
                void perform(() => setupNamedTunnel(hostname, true))
              }
            >
              {t("settings.namedTunnel.authorizeAgain")}
            </Button>
          ) : null}
          <Button
            type="submit"
            size="sm"
            disabled={
              disabled ||
              !hostname ||
              (method === "automatic" && configuration?.ready)
            }
          >
            {method === "token"
              ? t("common.save")
              : t("settings.namedTunnel.setup")}
          </Button>
        </div>
      </form>
      {setup ? (
        <div className="space-y-2 border-t border-border/60 py-3">
          <output className="block break-words text-xs text-muted-foreground">
            {setup.error
              ? namedTunnelErrorMessage(setup.error, t)
              : t(`settings.namedTunnel.states.${setup.state}`)}
          </output>
          {authUrl ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void openExternalLink(authUrl)}
            >
              {t("settings.namedTunnel.openLogin")}
            </Button>
          ) : null}
          {setup.active ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() =>
                void perform(() => cancelNamedTunnelSetup(setup.operationId))
              }
            >
              {t("common.cancel")}
            </Button>
          ) : null}
        </div>
      ) : null}
      {cleanupHostname &&
      (setup?.state === "removed" ||
        setup?.error === "dns_hostname_mismatch") ? (
        <p className="break-words pb-3 text-xs text-muted-foreground">
          {t("settings.namedTunnel.dnsCleanup", { hostname: cleanupHostname })}
        </p>
      ) : null}
      <AlertDialog
        open={conflictOpen}
        onOpenChange={(open) => {
          if (!open && setup?.state === "dns_conflict") {
            setDismissedConflict(setup.operationId);
            void perform(() => cancelNamedTunnelSetup(setup.operationId));
          }
        }}
      >
        <AlertDialogContent onCloseAutoFocus={restoreFocus}>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.namedTunnel.dnsConflictTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription className="break-words">
              {t("settings.namedTunnel.dnsConflict", {
                hostname: setup?.hostname ?? "",
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("common.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={busy}
              onClick={(event) => {
                event.preventDefault();
                if (setup?.tunnelId)
                  void perform(() =>
                    confirmNamedTunnelDns(
                      setup.operationId,
                      setup.hostname,
                      setup.tunnelId as string,
                    ),
                  );
              }}
            >
              {t("settings.namedTunnel.overwrite")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </fieldset>
  );
}
