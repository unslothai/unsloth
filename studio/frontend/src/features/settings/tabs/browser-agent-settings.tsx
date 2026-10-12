// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  DEFAULT_AGENT_RULE,
  type HistoryAccess,
  type SiteRule,
  normalizeSitePattern,
  useBrowserAgentStore,
} from "@/features/browser";
import { useT } from "@/i18n";
import { Add01Icon, Cancel01Icon, CursorInWindowIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

type Column = keyof SiteRule;

function RuleSelect({
  column,
  value,
  label,
  isDefault,
  onChange,
}: {
  column: Column;
  value: string;
  label: string;
  isDefault: boolean;
  onChange: (value: string) => void;
}) {
  const t = useT();
  const options: Array<[string, string]> =
    column === "browsing"
      ? [
          ["allow", t("browser.agentSettings.alwaysAllow")],
          ["ask", t(isDefault ? "browser.agentSettings.askNewSites" : "browser.agentSettings.requiresApproval")],
          ["block", t("browser.agentSettings.blocked")],
        ]
      : column === "actions"
        ? [
            ["ask", t("browser.agentSettings.requiresApproval")],
            ["allow", t("browser.agentSettings.alwaysAllow")],
          ]
        : [
            ["ask", t("browser.agentSettings.requiresApproval")],
            ["allow", t("browser.agentSettings.alwaysAllow")],
            ["block", t("browser.agentSettings.blocked")],
          ];
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-full min-w-0" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map(([option, text]) => (
          <SelectItem key={option} value={option}>
            {text}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

// Wide enough for "Requires approval"; a narrow dialog scrolls the table sideways instead of cutting it.
const GRID =
  "grid min-w-[calc(42rem*var(--ui-space-scale,1))] grid-cols-[minmax(calc(8rem*var(--ui-space-scale,1)),1.2fr)_repeat(3,minmax(calc(10.5rem*var(--ui-space-scale,1)),1fr))_2rem] items-center gap-2";

function PermissionsTable() {
  const t = useT();
  const permissions = useBrowserAgentStore((state) => state.permissions);
  const { setDefaultRule, setSiteRule } = useBrowserAgentStore.getState();
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const sites = Object.keys(permissions.sites).sort();
  const columns: Array<[Column, string]> = [
    ["browsing", t("browser.agentSettings.browsing")],
    ["actions", t("browser.agentSettings.actions")],
    ["downloads", t("browser.agentSettings.downloads")],
  ];

  const add = () => {
    const site = normalizeSitePattern(draft ?? "");
    if (!site) {
      setInvalid(true);
      return;
    }
    // A site added here is one the user trusts: open it without asking, keep the action prompts.
    setSiteRule(site, permissions.sites[site] ?? { ...DEFAULT_AGENT_RULE, browsing: "allow" });
    setDraft(null);
    setInvalid(false);
  };

  const row = (name: string, rule: SiteRule, site: string | null) => (
    <div key={site ?? "default"} className={`${GRID} py-1.5`}>
      <span className="truncate text-sm text-foreground" title={name}>
        {name}
      </span>
      {columns.map(([column, title]) => (
        <RuleSelect
          key={column}
          column={column}
          value={rule[column]}
          isDefault={site === null}
          label={`${title}: ${name}`}
          onChange={(value) =>
            site === null
              ? setDefaultRule({ [column]: value } as Partial<SiteRule>)
              : setSiteRule(site, { ...rule, [column]: value } as SiteRule)
          }
        />
      ))}
      {site === null ? (
        <span />
      ) : (
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          aria-label={t("browser.agentSettings.removeSite", { site })}
          onClick={() => setSiteRule(site, null)}
        >
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
        </Button>
      )}
    </div>
  );

  return (
    <SettingsSection
      title={t("browser.agentSettings.permissionsTitle")}
      description={t("browser.agentSettings.permissionsDescription")}
      action={
        <Button variant="outline" size="sm" onClick={() => setDraft("")} disabled={draft !== null}>
          <HugeiconsIcon icon={Add01Icon} strokeWidth={1.75} className="size-4" />
          {t("browser.agentSettings.addSite")}
        </Button>
      }
    >
      <div className="mt-2 flex flex-col overflow-x-auto rounded-lg border border-border/80 px-3 py-1">
        <div className={`${GRID} border-b border-border/60 py-2 text-xs text-muted-foreground`}>
          <span>{t("browser.agentSettings.site")}</span>
          {columns.map(([column, title]) => (
            <span key={column}>{title}</span>
          ))}
          <span />
        </div>
        {row(t("browser.agentSettings.default"), permissions.defaults, null)}
        {sites.map((site) => row(site, permissions.sites[site] ?? DEFAULT_AGENT_RULE, site))}
        {draft !== null ? (
          <form
            className="flex items-center gap-2 py-2"
            onSubmit={(event) => {
              event.preventDefault();
              add();
            }}
          >
            <Input
              autoFocus={true}
              value={draft}
              aria-invalid={invalid}
              aria-label={t("browser.agentSettings.site")}
              placeholder={t("browser.agentSettings.sitePlaceholder")}
              onChange={(event) => {
                setDraft(event.target.value);
                setInvalid(false);
              }}
              onKeyDown={(event) => {
                if (event.key === "Escape") setDraft(null);
              }}
              className="h-8 flex-1"
            />
            <Button type="submit" size="sm">
              {t("browser.agentSettings.add")}
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={() => setDraft(null)}>
              {t("browser.agentSettings.cancel")}
            </Button>
          </form>
        ) : null}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        {invalid ? t("browser.agentSettings.siteInvalid") : t("browser.agentSettings.permissionsNote")}
      </p>
    </SettingsSection>
  );
}

/** Settings > Browser: browser use on or off, history access, and the agent's site permissions. */
export function BrowserAgentSettings() {
  const t = useT();
  const enabled = useBrowserAgentStore((state) => state.enabled);
  const historyAccess = useBrowserAgentStore((state) => state.historyAccess);
  const { setEnabled, setHistoryAccess } = useBrowserAgentStore.getState();
  return (
    <>
      <SettingsSection title={t("browser.agentSettings.title")}>
        <SettingsRow
          icon={<HugeiconsIcon icon={CursorInWindowIcon} strokeWidth={1.75} className="size-6" />}
          label={t("browser.agentSettings.enable")}
          description={t("browser.agentSettings.enableDescription")}
        >
          <Switch aria-label={t("browser.agentSettings.enable")} checked={enabled} onCheckedChange={setEnabled} />
        </SettingsRow>
        <SettingsRow
          label={t("browser.agentSettings.history")}
          description={t("browser.agentSettings.historyDescription")}
        >
          <Select value={historyAccess} onValueChange={(value) => setHistoryAccess(value as HistoryAccess)}>
            <SelectTrigger className="w-40" aria-label={t("browser.agentSettings.history")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ask">{t("browser.agentSettings.alwaysAsk")}</SelectItem>
              <SelectItem value="allow">{t("browser.agentSettings.alwaysAllow")}</SelectItem>
              <SelectItem value="never">{t("browser.agentSettings.never")}</SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
      </SettingsSection>
      <PermissionsTable />
    </>
  );
}
