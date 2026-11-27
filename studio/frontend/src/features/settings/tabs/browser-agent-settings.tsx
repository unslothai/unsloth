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
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  DEFAULT_AGENT_RULE,
  type HistoryAccess,
  type ScriptAccess,
  type SiteRule,
  normalizeSitePattern,
  useBrowserAgentStore,
} from "@/features/browser";
import { useT } from "@/i18n";
import { CursorInWindowIcon } from "@/lib/browser-use-icon";
import { cn } from "@/lib/utils";
import { Cancel01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

type Column = Exclude<keyof SiteRule, "hostOnly">;

export const BROWSER_USE_DOCS_URL = "https://unsloth.ai/docs/basics/browser-use";

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
      : column === "actions" || column === "unlabelled" || column === "scripts"
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

/** Asked before scripts run unprompted on every site: Run scripts set to Always allow, or the Default row's Scripts rule. */
function ConfirmAllScripts({ open, onConfirm, onCancel }: { open: boolean; onConfirm: () => void; onCancel: () => void }) {
  const t = useT();
  return (
    <AlertDialog open={open} onOpenChange={(next) => !next && onCancel()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.agentSettings.scriptsConfirmTitle")}</AlertDialogTitle>
          <AlertDialogDescription>{t("browser.agentSettings.scriptsConfirmDescription")}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("browser.agentSettings.cancel")}</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>
            {t("browser.agentSettings.scriptsConfirmAction")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// Fixed columns, not auto-fit: auto-fit left a choice alone on its own row at odd widths.
const CHOICES = "grid grid-cols-1 gap-x-4 gap-y-3 @[30rem]/perms:grid-cols-2 @[44rem]/perms:grid-cols-3";
const SITE_LIST = "relative -mx-4 max-h-[min(28rem,55vh)] overflow-y-auto overscroll-contain px-4";

function PermissionsTable() {
  const t = useT();
  const permissions = useBrowserAgentStore((state) => state.permissions);
  const { setDefaultRule, setSiteRule } = useBrowserAgentStore.getState();
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const [confirmDefaultScripts, setConfirmDefaultScripts] = useState(false);
  const added = useRef<string | null>(null);
  const siteList = useRef<HTMLDivElement>(null);
  const sites = Object.keys(permissions.sites).sort();

  // Sorted, so a new site can land out of view. "nearest" also scrolls the settings panel: focusing the add form scrolled it.
  useEffect(() => {
    const site = added.current;
    added.current = null;
    if (site !== null) siteList.current?.querySelector(`[data-site="${CSS.escape(site)}"]`)?.scrollIntoView({ block: "nearest" });
  }, [permissions.sites]);
  const columns: Array<[Column, string]> = [
    ["browsing", t("browser.agentSettings.browsing")],
    ["actions", t("browser.agentSettings.actions")],
    ["downloads", t("browser.agentSettings.downloads")],
    ["unlabelled", t("browser.agentSettings.unlabelled")],
    ["scripts", t("browser.agentSettings.scriptsColumn")],
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
    added.current = site;
  };

  const row = (name: string, rule: SiteRule, site: string | null) => (
    <div
      key={site ?? "default"}
      data-site={site ?? undefined}
      className={cn(
        "flex flex-col gap-2.5 border-b border-border/60 py-4",
        site === null ? "pt-3" : null,
        draft === null && (site === null ? sites.length === 0 : site === sites[sites.length - 1]) ? "border-b-0 pb-5" : null,
      )}
    >
      <div className="flex min-h-8 items-center gap-2">
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-sm font-medium text-foreground" title={name}>
            {name}
          </span>
          {rule.hostOnly ? <span className="truncate text-xs text-muted-foreground">{t("browser.agentSettings.hostOnly")}</span> : null}
        </span>
        {site === null ? null : (
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
      <div className={CHOICES}>
        {columns.map(([column, title]) => (
          <div key={column} className="flex min-w-0 flex-col gap-1.5">
            <span className="px-1 text-xs text-muted-foreground">{title}</span>
            <RuleSelect
              column={column}
              value={rule[column]}
              isDefault={site === null}
              label={`${title}: ${name}`}
              onChange={(value) => {
                if (site !== null) setSiteRule(site, { ...rule, [column]: value } as SiteRule);
                else if (column === "scripts" && value === "allow") setConfirmDefaultScripts(true);
                else setDefaultRule({ [column]: value } as Partial<SiteRule>);
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <SettingsSection
      title={t("browser.agentSettings.permissionsTitle")}
      description={t("browser.agentSettings.permissionsDescription")}
      action={
        <Button variant="outline" size="sm" onClick={() => setDraft("")} disabled={draft !== null}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-4" />
          {t("browser.agentSettings.addSite")}
        </Button>
      }
    >
      <div className="@container/perms mt-2 flex flex-col rounded-2xl border border-border/80 px-4">
        {row(t("browser.agentSettings.default"), permissions.defaults, null)}
        {sites.length > 0 ? (
          <div ref={siteList} className={SITE_LIST}>
            {sites.map((site) => row(site, permissions.sites[site] ?? DEFAULT_AGENT_RULE, site))}
          </div>
        ) : null}
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
            <Button variant="secondary" type="submit" size="sm">
              <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
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
      <ConfirmAllScripts
        open={confirmDefaultScripts}
        onConfirm={() => {
          setDefaultRule({ scripts: "allow" });
          setConfirmDefaultScripts(false);
        }}
        onCancel={() => setConfirmDefaultScripts(false)}
      />
    </SettingsSection>
  );
}

/** Settings > Browser: browser use on or off, history and script access, and the agent's site permissions. */
export function BrowserAgentSettings() {
  const t = useT();
  const enabled = useBrowserAgentStore((state) => state.enabled);
  const historyAccess = useBrowserAgentStore((state) => state.historyAccess);
  const scriptAccess = useBrowserAgentStore((state) => state.scriptAccess);
  const showCursor = useBrowserAgentStore((state) => state.showCursor);
  const { setEnabled, setHistoryAccess, setScriptAccess, setShowCursor } = useBrowserAgentStore.getState();
  const [confirmScripts, setConfirmScripts] = useState(false);
  return (
    <>
      <SettingsSection
        title={t("browser.agentSettings.title")}
      >
        <SettingsRow
          icon={<HugeiconsIcon icon={CursorInWindowIcon} strokeWidth={1.75} className="me-1.5 size-7" />}
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
        <SettingsRow
          label={t("browser.agentSettings.scripts")}
          description={t(scriptAccess === "allow" ? "browser.agentSettings.scriptsAllowDescription" : "browser.agentSettings.scriptsDescription")}
        >
          <Select
            value={scriptAccess}
            onValueChange={(value) => (value === "allow" ? setConfirmScripts(true) : setScriptAccess(value as ScriptAccess))}
          >
            <SelectTrigger className="w-40" aria-label={t("browser.agentSettings.scripts")}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ask">{t("browser.agentSettings.askEveryTime")}</SelectItem>
              <SelectItem value="allow">{t("browser.agentSettings.alwaysAllow")}</SelectItem>
              <SelectItem value="never">{t("browser.agentSettings.never")}</SelectItem>
            </SelectContent>
          </Select>
          <ConfirmAllScripts
            open={confirmScripts}
            onConfirm={() => {
              setScriptAccess("allow");
              setConfirmScripts(false);
            }}
            onCancel={() => setConfirmScripts(false)}
          />
        </SettingsRow>
        <SettingsRow label={t("browser.agentSettings.cursor")} description={t("browser.agentSettings.cursorDescription")}>
          <Switch aria-label={t("browser.agentSettings.cursor")} checked={showCursor} onCheckedChange={setShowCursor} />
        </SettingsRow>
      </SettingsSection>
      <PermissionsTable />
    </>
  );
}
