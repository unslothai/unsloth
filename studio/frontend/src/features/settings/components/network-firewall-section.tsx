// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { formatBytes } from "@/features/hub";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { type ReactNode, useCallback, useEffect, useId, useState } from "react";
import {
  NETWORK_MODES,
  type NetworkConnection,
  type NetworkFirewall,
  type NetworkMode,
  addNetworkSite,
  loadNetworkConnections,
  loadNetworkFirewall,
  parseSites,
  updateNetworkFirewall,
} from "../api/network-firewall";
import { isSettingsRouteAbsent } from "../api/settings-route-absent";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const MODE_TEXT = {
  open: { name: "settings.sandbox.network.modes.open", detail: "settings.sandbox.network.modeDetails.open" },
  ask: { name: "settings.sandbox.network.modes.ask", detail: "settings.sandbox.network.modeDetails.ask" },
  allowlist: {
    name: "settings.sandbox.network.modes.allowlist",
    detail: "settings.sandbox.network.modeDetails.allowlist",
  },
} as const satisfies Record<NetworkMode, { name: TranslationKey; detail: TranslationKey }>;

const LIST_CLASS = "w-full max-w-[calc(26rem*var(--ui-space-scale,1))]";

function SitesRow({
  label,
  description,
  value,
  disabled,
  onChange,
  below,
}: {
  label: string;
  description: string;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  below?: ReactNode;
}) {
  return (
    <SettingsRow label={label} description={description} wideControl={true} below={below}>
      <Textarea
        value={value}
        disabled={disabled}
        spellCheck={false}
        aria-label={label}
        placeholder="example.com"
        onChange={(event) => onChange(event.target.value)}
        className={cn(LIST_CLASS, "max-h-[calc(13rem*var(--ui-space-scale,1))] overflow-y-auto font-mono text-xs")}
      />
    </SettingsRow>
  );
}

function RecentConnections({
  connections,
  stored,
  busy,
  onMove,
  onAllowReached,
}: {
  connections: NetworkConnection[];
  stored: NetworkFirewall;
  busy: boolean;
  onMove: (kind: "allowed" | "blocked", host: string) => void;
  onAllowReached: (hosts: string[]) => void;
}) {
  const t = useT();
  const reached = connections
    .filter((row) => row.verdict === "allowed" && !stored.allowedSites.includes(row.host))
    .map((row) => row.host);
  return (
    <SettingsRow
      label={t("settings.sandbox.network.recentLabel")}
      description={t("settings.sandbox.network.recentDescription")}
      wideControl={true}
      below={
        reached.length > 1 ? (
          <Button
            variant="outline"
            size="sm"
            disabled={busy}
            onClick={() => onAllowReached(reached)}
          >
            {t("settings.sandbox.network.recentAllowReached")}
          </Button>
        ) : null
      }
    >
      <ul className={cn(LIST_CLASS, "flex flex-col gap-1.5")}>
        {connections.slice(0, 12).map((row) => (
          <li key={row.host} className="flex items-center justify-end gap-2 text-xs">
            <span className="min-w-0 truncate font-mono" title={row.host}>
              {row.host}
            </span>
            <span
              className={cn(
                "shrink-0",
                row.verdict === "refused" ? "text-destructive" : "text-muted-foreground",
              )}
            >
              {row.verdict === "refused"
                ? t("settings.sandbox.network.recentRefused")
                : t("settings.sandbox.network.recentSent", { size: formatBytes(row.sent) })}
            </span>
            {stored.allowedSites.includes(row.host) ? null : (
              <Button
                variant="outline"
                size="xs"
                disabled={busy}
                onClick={() => onMove("allowed", row.host)}
              >
                {t("settings.sandbox.network.recentAllow")}
              </Button>
            )}
            {stored.blockedSites.includes(row.host) ? null : (
              <Button
                variant="outline"
                size="xs"
                disabled={busy}
                onClick={() => onMove("blocked", row.host)}
              >
                {t("settings.sandbox.network.recentBlock")}
              </Button>
            )}
          </li>
        ))}
      </ul>
    </SettingsRow>
  );
}

export function NetworkFirewallSection() {
  const t = useT();
  const pillId = useId();
  const modeDescriptionId = useId();
  const [saved, setSaved] = useState<NetworkFirewall | null>(null);
  const [mode, setMode] = useState<NetworkMode>("open");
  const [allowed, setAllowed] = useState("");
  const [blocked, setBlocked] = useState("");
  const [connections, setConnections] = useState<NetworkConnection[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const show = useCallback((stored: NetworkFirewall) => {
    setSaved(stored);
    setMode(stored.mode);
    setAllowed(stored.allowedSites.join("\n"));
    setBlocked(stored.blockedSites.join("\n"));
    setError(null);
  }, []);

  useEffect(() => {
    let live = true;
    loadNetworkFirewall(t("settings.sandbox.network.loadError"))
      .then((stored) => {
        if (live) show(stored);
      })
      .catch((loadError) => {
        if (!live || isSettingsRouteAbsent(loadError)) return;
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      });
    void loadNetworkConnections().then((rows) => {
      if (live) setConnections(rows);
    });
    return () => {
      live = false;
    };
  }, [show, t]);

  if (saved === null && error === null) return null;
  const same = (text: string, list: string[]) => parseSites(text).join("\n") === list.join("\n");
  const unsaveable =
    saved === null ||
    (same(allowed, saved.allowedSites) && same(blocked, saved.blockedSites));
  const run = (request: Promise<NetworkFirewall>) => {
    setSaving(true);
    request
      .then(show)
      .catch((saveError) => {
        setError(saveError instanceof Error ? saveError.message : String(saveError));
      })
      .finally(() => setSaving(false));
  };
  const save = () =>
    run(
      updateNetworkFirewall(
        {
          mode,
          allowedSites: parseSites(allowed),
          blockedSites: parseSites(blocked),
        },
        t("settings.sandbox.network.saveError"),
      ),
    );
  // The mode applies as it is picked; drafts typed into the lists stay until Save.
  const pickMode = (next: NetworkMode) => {
    if (!saved || next === saved.mode) return;
    setMode(next);
    setSaving(true);
    updateNetworkFirewall({ ...saved, mode: next }, t("settings.sandbox.network.saveError"))
      .then((stored) => {
        setSaved(stored);
        setMode(stored.mode);
        setError(null);
      })
      .catch((saveError) => {
        setMode(saved.mode);
        setError(saveError instanceof Error ? saveError.message : String(saveError));
      })
      .finally(() => setSaving(false));
  };
  const locked = saving || saved === null;
  const strict = mode !== "open";

  return (
    <SettingsSection title={t("settings.sandbox.network.title")}>
      <SettingsRow
        label={t("settings.sandbox.network.modeLabel")}
        description={
          <span id={modeDescriptionId}>
            <span className="font-medium text-foreground">{t(MODE_TEXT[mode].name)}:</span>{" "}
            {t(MODE_TEXT[mode].detail)}
            {strict && saved ? (
              <span className="mt-1 block">
                {t(
                  saved.level === "locked"
                    ? "settings.sandbox.network.locked"
                    : "settings.sandbox.network.watched",
                )}
              </span>
            ) : null}
          </span>
        }
      >
        <div
          className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
          role="group"
          aria-label={t("settings.sandbox.network.modeLabel")}
          aria-describedby={modeDescriptionId}
        >
          {NETWORK_MODES.map((option) => {
            const selected = option === mode;
            return (
              <button
                key={option}
                type="button"
                aria-pressed={selected}
                disabled={locked}
                onClick={() => pickMode(option)}
                className={cn(
                  "relative inline-flex h-8 cursor-pointer items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
                  selected
                    ? "cursor-default text-foreground"
                    : "text-muted-foreground hover:text-foreground disabled:hover:text-muted-foreground",
                )}
              >
                {selected && <SegmentedPill group={`${pillId}-mode`} />}
                {t(MODE_TEXT[option].name)}
              </button>
            );
          })}
        </div>
      </SettingsRow>
      {strict ? (
        <SitesRow
          label={t("settings.sandbox.network.allowedLabel")}
          description={t("settings.sandbox.network.allowedDescription")}
          value={allowed}
          disabled={locked}
          onChange={setAllowed}
        />
      ) : null}
      <SitesRow
        label={t("settings.sandbox.network.blockedLabel")}
        description={t("settings.sandbox.network.blockedDescription")}
        value={blocked}
        disabled={locked}
        onChange={setBlocked}
        below={
          <div className="flex flex-col items-end gap-2">
            {error ? (
              <span className="max-w-[calc(26rem*var(--ui-space-scale,1))] text-right text-xs text-destructive">
                {error}
              </span>
            ) : null}
            <Button variant="outline" size="sm" disabled={saving || unsaveable} onClick={save}>
              {saving ? t("common.saving") : t("common.save")}
            </Button>
          </div>
        }
      />
      {saved && connections.length ? (
        <RecentConnections
          connections={connections}
          stored={saved}
          busy={saving}
          onMove={(kind, host) =>
            run(addNetworkSite(kind, host, t("settings.sandbox.network.saveError")))
          }
          onAllowReached={(hosts) =>
            run(
              updateNetworkFirewall(
                { ...saved, allowedSites: [...saved.allowedSites, ...hosts] },
                t("settings.sandbox.network.saveError"),
              ),
            )
          }
        />
      ) : null}
    </SettingsSection>
  );
}
