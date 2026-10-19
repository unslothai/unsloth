// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice mode's Decide list, built from the model picker's own parts: the search field, the
// Recommended / On Device pills, the filter and sort menus, ListLabel groups, ModelRow rows and
// the Custom Folders section.

import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { type HubOption, HubOptionMenu } from "@/features/hub";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  ArrowUpDownIcon,
  Download01Icon,
  Search01Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useMemo, useState } from "react";
import { CustomFoldersControls, useScanFolders } from "./custom-folders";
import { PillTabs } from "./pill-tabs";
import { formatFootprintBytes, ListLabel, ModelRow } from "./pickers";

export interface DecisionPickerModel {
  name: string;
  label: string;
  description: string;
  kind: "catalog" | "fine_tune" | "folder";
  path: string | null;
  available: boolean;
  unavailable_reason: string | null;
  family?: string | null;
  params?: string | null;
  gguf?: string | null;
  size_bytes?: number | null;
  mtime?: number | null;
}

type Section = "recommended" | "downloaded";
type Family = "all" | "laya" | "clef";
type Sort = "recent" | "name";

const NO_CAPABILITIES = {
  vision: false,
  reasoning: false,
  audio: false,
  imageGen: false,
  videoGen: false,
};

// The picker's own control classes (pickers.tsx), so the menus line up the same way.
const SORT_TRIGGER =
  "h-(--picker-control-h) w-(--picker-control-w) shrink-0 justify-between gap-1.5 pl-4 pr-3.5 !border-0 text-xs [&>span]:!text-clip";
const SORT_MENU =
  "picker-option-menu !p-1 !rounded-[14px] [&_[role=option]]:!pl-2 [&_[role=option]]:!py-1.5 [&_[role=option]]:!text-xs [&_[role=option]]:!rounded-[10px]";

function rowMeta(m: DecisionPickerModel): string {
  const format = m.kind === "catalog" ? null : "Safetensors";
  return [
    format,
    m.params,
    m.size_bytes ? formatFootprintBytes(m.size_bytes) : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

export function DecisionPicker({
  models,
  current,
  state,
  error,
  launchDefault,
  fromSetting,
  onPick,
  onFoldersChange,
}: {
  models: DecisionPickerModel[];
  current: string | null;
  state: "loaded" | "loading" | "not_loaded" | "error" | null;
  error: string | null;
  launchDefault: string | null;
  /** The pick is stored, so the launch default can be put back. */
  fromSetting: boolean;
  onPick: (name: string | null) => void;
  onFoldersChange: () => void;
}) {
  const t = useT();
  const currentModel = models.find((m) => m.name === current);
  const [section, setSection] = useState<Section>(
    currentModel && currentModel.kind !== "catalog"
      ? "downloaded"
      : "recommended",
  );
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState<Family>("all");
  const [sort, setSort] = useState<Sort>("recent");
  const [fineTunedCollapsed, setFineTunedCollapsed] = useState(false);
  const [foldersCollapsed, setFoldersCollapsed] = useState(false);
  const folders = useScanFolders(onFoldersChange);

  const familyOptions: HubOption<Family>[] = [
    { value: "all", label: t("voicePanel.filterAll") },
    { value: "laya", label: "Laya" },
    { value: "clef", label: "Clef" },
  ];
  const sortOptions: HubOption<Sort>[] = [
    { value: "recent", label: t("voicePanel.sortRecent") },
    { value: "name", label: t("voicePanel.sortName") },
  ];
  const tabs = [
    {
      value: "recommended",
      label: t("voicePanel.tabRecommended"),
      icon: <HugeiconsIcon icon={StarIcon} className="size-3.5 shrink-0" />,
    },
    {
      value: "downloaded",
      label: t("picker.onDevice"),
      icon: (
        <HugeiconsIcon icon={Download01Icon} className="size-3.5 shrink-0" />
      ),
    },
  ];

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = models.filter(
      (m) =>
        (!q ||
          m.label.toLowerCase().includes(q) ||
          m.name.toLowerCase().includes(q)) &&
        (family === "all" ||
          (family === "clef" ? m.family !== "Laya" : m.family === "Laya")),
    );
    return rows.sort((a, b) =>
      sort === "name" || a.mtime == null || b.mtime == null
        ? a.label.localeCompare(b.label)
        : b.mtime - a.mtime,
    );
  }, [models, query, family, sort]);

  const renderRow = (m: DecisionPickerModel): ReactNode => {
    const selected = m.name === current;
    const tags = [m.family ?? "Laya"];
    if (launchDefault && m.name === launchDefault)
      tags.push(t("voicePanel.defaultBadge"));
    return (
      <div key={m.name}>
        <ModelRow
          label={m.label}
          meta={rowMeta(m)}
          selected={selected}
          loaded={selected && state === "loaded"}
          onClick={() => m.available && onPick(m.name)}
          capabilities={NO_CAPABILITIES}
          hideOwner={true}
          tags={tags}
          quantChip={m.gguf ? `GGUF ${m.gguf}` : null}
          alignMeta={m.kind === "catalog" ? undefined : "device"}
          tooltipText={m.unavailable_reason ?? m.path ?? m.description}
          className={cn(!m.available && "opacity-50")}
        />
        {selected && state === "error" && error ? (
          <p className="px-2.5 pb-1 text-ui-10 text-destructive">{error}</p>
        ) : null}
        {!m.available && m.unavailable_reason ? (
          <p className="px-2.5 pb-1 text-ui-10 text-muted-foreground">
            {m.unavailable_reason}
          </p>
        ) : null}
      </div>
    );
  };

  const stock = visible.filter((m) => m.kind === "catalog");
  const fineTunes = visible.filter((m) => m.kind === "fine_tune");
  const inFolders = visible.filter((m) => m.kind === "folder");
  const empty = (
    <div className="px-2.5 py-2 text-xs text-muted-foreground">
      {t("voicePanel.noMatches")}
    </div>
  );

  return (
    <TooltipProvider
      delayDuration={0}
      skipDelayDuration={0}
      disableHoverableContent={true}
    >
      <div className="relative space-y-2">
        <div className="flex items-center gap-2 pb-1 pr-2">
          <div className="relative flex-1">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={t("voicePanel.searchPlaceholder")}
              data-model-picker-search-input={true}
              className="field-soft h-(--picker-control-h) border-0 pl-8 pr-8 text-sm"
            />
          </div>
        </div>
        <div className="-mr-2 flex flex-wrap items-center gap-2">
          <PillTabs
            className="[&_[role=tab]]:px-[calc(0.75rem*var(--ui-space-scale,1)_+_var(--picker-tab-pad)/2)]"
            ariaLabel={t("picker.hubSectionAriaLabel")}
            tabs={tabs}
            value={section}
            onValueChange={(next) => setSection(next as Section)}
            fit={true}
          />
          <div className="flex max-w-full min-w-0 flex-wrap items-center gap-2">
            <HubOptionMenu
              value={family}
              options={familyOptions}
              onValueChange={setFamily}
              ariaLabel={t("voicePanel.filterAria")}
              align="end"
              className={SORT_TRIGGER}
              contentClassName={SORT_MENU}
              composer={false}
            />
            <HubOptionMenu
              value={sort}
              options={sortOptions}
              onValueChange={setSort}
              ariaLabel={t("voicePanel.sortAria")}
              align="end"
              className={SORT_TRIGGER}
              contentClassName={SORT_MENU}
              composer={false}
              triggerContent={
                <span className="flex items-center gap-1">
                  <HugeiconsIcon
                    icon={ArrowUpDownIcon}
                    strokeWidth={1.75}
                    className="size-3.5 shrink-0 text-foreground"
                  />
                  <span className="truncate">
                    {sortOptions.find((o) => o.value === sort)?.label}
                  </span>
                </span>
              }
              footer={
                fromSetting ? (
                  <button
                    type="button"
                    onClick={() => onPick(null)}
                    className="flex w-full items-center rounded-[10px] px-2 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {t("voicePanel.useLaunchDefault")}
                  </button>
                ) : undefined
              }
            />
          </div>
        </div>
        <p className="px-2.5 text-ui-10 text-muted-foreground">
          {t("voicePanel.decideHint")}
        </p>
        <div className="model-list-scroll max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto scroll-py-1.5 pl-0.5">
          <div className="model-list-gutter pb-4">
            {section === "recommended" ? (
              <>
                <ListLabel>Unsloth</ListLabel>
                {stock.length > 0 ? stock.map(renderRow) : empty}
              </>
            ) : (
              <>
                {fineTunes.length > 0 ? (
                  <>
                    <ListLabel
                      collapsed={fineTunedCollapsed}
                      onToggle={() => setFineTunedCollapsed((v) => !v)}
                    >
                      {t("voicePanel.decideGroupFineTunes")}
                    </ListLabel>
                    {!fineTunedCollapsed && fineTunes.map(renderRow)}
                  </>
                ) : null}
                <CustomFoldersControls
                  folders={folders}
                  collapsed={foldersCollapsed}
                  onCollapsedChange={setFoldersCollapsed}
                />
                {!foldersCollapsed &&
                  (inFolders.length > 0 ? inFolders.map(renderRow) : empty)}
              </>
            )}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
