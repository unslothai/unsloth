// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown, Hand, ShieldCheck } from "lucide-react";
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui";
import type { ComponentType } from "react";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { useFullAccessAllowed } from "@/features/auth/account-session";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";

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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSubContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownStandardIcon, ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { ShieldAlertGlyph } from "@/lib/shield-alert-icon";
import { SparklesGlyph } from "@/lib/sparkles-icon";
import { MenuTickIcon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ComputerTerminal01Icon,
  Folder01Icon,
  InternetIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type SandboxCapability,
  loadSandboxCapability,
  onSandboxCapabilityChange,
} from "./api/sandbox-capability";
import { sandboxSwitchState } from "./sandbox-level";
import {
  pickSandboxLevel,
} from "./sandbox-pick";
import {
  SandboxSetupDialog,
  useSandboxSetupDialogStore,
} from "./sandbox-setup-dialog";
import {
  type PermissionMode,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";

/** Full access stays last: it disables approval prompts and the sandbox. */
export const PERMISSION_MODE_OPTIONS: readonly {
  value: PermissionMode;
  label: string;
  description: string;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}[] = [
  {
    value: "ask",
    label: "Ask for approval",
    description: "Ask before using tools or the internet",
    icon: Hand,
  },
  {
    value: "auto",
    label: "Approve for me",
    description: "Only ask for actions that look risky",
    icon: ShieldCheck,
  },
  {
    value: "off",
    label: "Run automatically",
    description: "Never ask, but keep code in the sandbox",
    icon: SparklesGlyph,
  },
  {
    value: "full",
    label: "Full access",
    description: "Full access to your computer",
    icon: ShieldAlertGlyph,
  },
] as const;

const FULL_ACCESS_SCOPES = [
  {
    icon: Folder01Icon,
    title: "Files",
    description: "Read, change or delete any file you can access",
  },
  {
    icon: ComputerTerminal01Icon,
    title: "Commands",
    description: "Run terminal and Python code, install packages",
  },
  {
    icon: InternetIcon,
    title: "Internet",
    description: "Browse, send data and use MCP tools",
  },
] as const;

export function permissionModeOption(mode: PermissionMode) {
  return (
    PERMISSION_MODE_OPTIONS.find((option) => option.value === mode) ??
    // Unknown values fall back to the default ("Approve for me"), not row 0.
    PERMISSION_MODE_OPTIONS.find((option) => option.value === "auto") ??
    PERMISSION_MODE_OPTIONS[0]
  );
}

/** Menu heading. `sandboxControls` adds the Sandbox Low/High picker; Settings shows its own row.
 *  `onOsSandboxMissing` opens the install popup; by default the one at the chat-page root. */
export function PermissionMenuLabel({
  sandboxControls,
  onOsSandboxMissing,
}: {
  sandboxControls: boolean;
  onOsSandboxMissing?: () => void;
}) {
  const t = useT();
  return (
    <DropdownMenuLabel className="flex items-center justify-between gap-3">
      <span>{t("settings.general.permissions.sectionTitle")}</span>
      {sandboxControls ? <SandboxLevelMenuPicker onOsSandboxMissing={onOsSandboxMissing} /> : null}
    </DropdownMenuLabel>
  );
}

/** Space between the permission menu and the sandbox picker beside it. */
const SANDBOX_PICKER_GAP = 8;

/** Sandbox levels, weakest first. Disabled only shows under Full access and is never picked. */
const SANDBOX_LEVEL_OPTIONS = [
  { value: "off", labelKey: "settings.sandbox.levelOff", descriptionKey: "settings.sandbox.levelOffShort" },
  { value: "low", labelKey: "settings.sandbox.levelLow", descriptionKey: "settings.sandbox.levelLowShort" },
  { value: "high", labelKey: "settings.sandbox.levelHigh", descriptionKey: "settings.sandbox.levelHighShort" },
] as const;

/** "Sandbox High ›" chip that opens the level picker. High without an OS sandbox reads Low and
 *  picking it opens the setup popup. Full access shows Disabled and locks Low and High. */
function SandboxLevelMenuPicker({ onOsSandboxMissing }: { onOsSandboxMissing?: () => void }) {
  const t = useT();
  const sandboxLevel = useChatRuntimeStore((s) => s.sandboxLevel);
  const setSandboxLevel = useChatRuntimeStore((s) => s.setSandboxLevel);
  const setSandboxSetupOpen = useSandboxSetupDialogStore((s) => s.setOpen);
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const { permissionMode } = useAccountPermissionMode();
  const capability = useSandboxCapability(sandboxLevel === "high");
  const { checked, disabled } = sandboxSwitchState(sandboxLevel, permissionMode, capability);
  const active = disabled ? "off" : checked ? "high" : "low";
  const options = SANDBOX_LEVEL_OPTIONS.filter((option) => disabled || option.value !== "off");
  const activeOption = SANDBOX_LEVEL_OPTIONS.find((option) => option.value === active)!;
  const descriptionId = useId();
  const chipRef = useRef<HTMLDivElement | null>(null);
  const [offsets, setOffsets] = useState({ side: SANDBOX_PICKER_GAP, align: 0 });
  // State, not a ref: the portal mounts the picker a render after `open` flips.
  const [picker, setPicker] = useState<HTMLDivElement | null>(null);
  // Ours, not Radix's: it closes a submenu once the pointer or focus leaves it.
  const [open, setOpen] = useState(false);

  // Stays open until a click outside the picker. The chip toggles it itself.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (picker?.contains(target) || chipRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [open, picker]);

  // Places the picker a gap clear of the menu, tops aligned. Radix applies one sideOffset to
  // whichever side it lands on, so pick the side here: right if it fits, else left.
  useLayoutEffect(() => {
    const chip = chipRef.current;
    const menu = chip?.closest<HTMLElement>('[role="menu"]');
    if (!open || !chip || !menu || !picker) return;
    const chipBox = chip.getBoundingClientRect();
    const menuBox = menu.getBoundingClientRect();
    const width = picker.offsetWidth;
    const fitsRight = menuBox.right + SANDBOX_PICKER_GAP + width <= window.innerWidth;
    const fitsLeft = menuBox.left - SANDBOX_PICKER_GAP - width >= 0;
    setOffsets({
      side:
        !fitsRight && fitsLeft
          ? chipBox.left - menuBox.left + SANDBOX_PICKER_GAP
          : menuBox.right - chipBox.right + SANDBOX_PICKER_GAP,
      align: menuBox.top - chipBox.top,
    });
  }, [open, picker]);

  return (
    <DropdownMenuPrimitive.Sub
      open={open}
      // Close requests are ignored; see the pointerdown effect.
      onOpenChange={(next) => {
        if (next) setOpen(true);
      }}
    >
      <DropdownMenuPrimitive.SubTrigger
        ref={chipRef}
        aria-describedby={descriptionId}
        // Opens on click or arrow key, not hover.
        onPointerMove={(event) => event.preventDefault()}
        onClick={(event) => {
          if (!open) return;
          event.preventDefault();
          setOpen(false);
        }}
        // Negative margins cancel the hover pill's padding so nothing shifts.
        className="menu-heading-chip -my-1 -mr-1.5 flex shrink-0 cursor-pointer items-center gap-1 rounded-full border-0 py-1 pr-1.5 pl-2.5 text-ui-12 font-medium outline-none transition-colors"
      >
        <span className="text-foreground">{t("settings.sandbox.levelLabel")}</span>
        <span className={active === "off" ? "text-muted-foreground" : "text-primary"}>
          {t(activeOption.labelKey)}
        </span>
        <HugeiconsIcon
          icon={ChevronRightStandardIcon}
          strokeWidth={1.75}
          className="-ml-0.5 size-[calc(13px*var(--ui-space-scale,1))] text-foreground"
        />
        <span id={descriptionId} className="sr-only">
          {t(activeOption.descriptionKey)}
        </span>
      </DropdownMenuPrimitive.SubTrigger>
      <DropdownMenuSubContent
        ref={setPicker}
        sideOffset={offsets.side}
        alignOffset={offsets.align}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setOpen(false);
        }}
        className="unsloth-plus-menu w-[calc(312px*var(--ui-space-scale,1))]"
      >
        <DropdownMenuLabel className="flex items-start justify-between gap-3">
          <span className="min-w-0">{t("settings.sandbox.levelPickerTitle")}</span>
          <DropdownMenuPrimitive.Item
            // my-0!: drops the menu item margin so it lines up with the question.
            className="my-0! shrink-0 cursor-pointer rounded-sm font-normal text-muted-foreground underline decoration-muted-foreground/50 underline-offset-[3px] outline-hidden transition-colors hover:text-foreground hover:decoration-foreground/60 data-[highlighted]:text-foreground data-[highlighted]:decoration-foreground/60"
            // Deferred past the menu's focus restore.
            onSelect={() =>
              setTimeout(
                () => openSettings("sandbox", { scrollTarget: "sandbox-permissions" }),
                0,
              )
            }
          >
            {t("settings.sandbox.learnMore")}
          </DropdownMenuPrimitive.Item>
        </DropdownMenuLabel>
        {options.map((option) => (
          <DropdownMenuItem
            key={option.value}
            disabled={disabled && option.value !== "off"}
            onSelect={() => {
              if (option.value === active || option.value === "off") return;
              void pickSandboxLevel(option.value, setSandboxLevel, () =>
                // Deferred past the menu's focus restore.
                setTimeout(onOsSandboxMissing ?? (() => setSandboxSetupOpen(true)), 0),
              );
            }}
            className={cn("items-start gap-2 py-2", active === option.value && "font-medium")}
          >
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="text-ui-13 leading-tight">{t(option.labelKey)}</span>
              <span className="text-xs font-normal leading-snug text-muted-foreground">
                {t(option.descriptionKey)}
              </span>
            </span>
            {active === option.value ? (
              <HugeiconsIcon
                icon={MenuTickIcon}
                strokeWidth={2}
                className="permission-mode-tick ml-auto size-4 shrink-0 self-center"
              />
            ) : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuPrimitive.Sub>
  );
}

/** The option rows shared by every permission dropdown or submenu. Non-full levels apply
 *  directly; picking Full access must go through the caller's danger confirmation. */
function useAccountPermissionMode() {
  const fullAccessAllowed = useFullAccessAllowed();
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);
  useEffect(() => {
    if (!fullAccessAllowed && permissionMode === "full") setPermissionMode("auto");
  }, [fullAccessAllowed, permissionMode, setPermissionMode]);
  return {
    permissionMode: !fullAccessAllowed && permissionMode === "full" ? "auto" : permissionMode,
    fullAccessAllowed,
  };
}

/** Null while unknown, when the server is too old to say, or when `enabled` is false (Low needs no
 *  OS sandbox answer, and the read probes it); follows later reads and resets. */
export function useSandboxCapability(enabled: boolean): SandboxCapability | null {
  const [capability, setCapability] = useState<SandboxCapability | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    // Only the newest read applies, so a late older answer cannot undo it.
    let reads = 0;
    const read = () => {
      const id = ++reads;
      void loadSandboxCapability().then((next) => {
        if (live && id === reads) setCapability(next);
      });
    };
    read();
    const stop = onSandboxCapabilityChange(read);
    return () => {
      live = false;
      stop();
    };
  }, [enabled]);
  return enabled ? capability : null;
}

export function PermissionModeMenuItems({
  onRequestFullAccess,
}: {
  onRequestFullAccess: () => void;
}) {
  const { permissionMode, fullAccessAllowed } = useAccountPermissionMode();
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <>
      {PERMISSION_MODE_OPTIONS.filter((option) => fullAccessAllowed || option.value !== "full").map((option) => (
        <DropdownMenuItem
          key={option.value}
          onSelect={() => {
            if (option.value === permissionMode) return;
            // Run automatically applies at once: without an OS sandbox the switch reads Low and
            // risky Python and Terminal calls still ask.
            if (option.value === "full") {
              onRequestFullAccess();
            } else {
              setPermissionMode(option.value);
            }
          }}
          className={cn(
            "items-start gap-2 py-2",
            permissionMode === option.value && "font-medium",
          )}
        >
          <option.icon className="mt-0.5 size-4 shrink-0" strokeWidth={2} />
          <span className="flex min-w-0 flex-1 flex-col gap-0.5">
            <span className="text-ui-13 leading-tight">{option.label}</span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {option.description}
            </span>
          </span>
          {permissionMode === option.value ? (
            <HugeiconsIcon
              icon={MenuTickIcon}
              strokeWidth={2}
              className="permission-mode-tick ml-auto size-4 shrink-0 self-center"
            />
          ) : null}
        </DropdownMenuItem>
      ))}
    </>
  );
}

export function useActivePermissionMode() {
  return permissionModeOption(useAccountPermissionMode().permissionMode);
}

function lastFocusOutsideMenus(): HTMLElement | null {
  let element = document.activeElement;
  for (
    let menu = element?.closest('[role="menu"]');
    menu;
    menu = element?.closest('[role="menu"]')
  ) {
    const triggerId = menu.getAttribute("aria-labelledby");
    element = triggerId ? document.getElementById(triggerId) : null;
  }
  return element instanceof HTMLElement && element !== document.body ? element : null;
}

export function FullAccessConfirmContent({
  onConfirm,
  onClose,
}: {
  onConfirm: () => void;
  onClose: () => void;
}) {
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  return (
    <AlertDialogContent
      className="gap-5 p-7 ring-0 data-[size=default]:sm:max-w-[calc(500px*var(--ui-space-scale,1))]"
      onOverlayClick={onClose}
      // Focus the card, not Cancel, so Cancel shows no focus ring until tabbed to.
      onOpenAutoFocus={(event) => {
        event.preventDefault();
        returnFocusRef.current = lastFocusOutsideMenus();
        (event.currentTarget as HTMLElement).focus();
      }}
    >
      <AlertDialogHeader className="gap-2">
        <AlertDialogTitle className="flex items-center gap-2.5">
          <ShieldAlertGlyph className="size-5 shrink-0" strokeWidth={2} />
          Turn on Full access?
        </AlertDialogTitle>
        <AlertDialogDescription className="text-pretty leading-relaxed">
          Tools will run without asking and outside the sandbox, including:
        </AlertDialogDescription>
      </AlertDialogHeader>
      <ul className="flex flex-col rounded-2xl bg-muted/50 px-5 py-1.5">
        {FULL_ACCESS_SCOPES.map((scope) => (
          <li key={scope.title} className="flex items-center gap-4 py-3">
            <HugeiconsIcon
              icon={scope.icon}
              strokeWidth={1.75}
              className="size-5 shrink-0 text-muted-foreground"
            />
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="text-sm font-medium">{scope.title}</span>
              <span className="text-xs leading-relaxed text-muted-foreground">
                {scope.description}
              </span>
            </span>
          </li>
        ))}
      </ul>
      <p className="text-pretty text-xs leading-relaxed text-muted-foreground">
        Risks include data loss or exposure and prompt injection. You can turn this off
        anytime.{" "}
        <button
          type="button"
          className="cursor-pointer text-foreground underline underline-offset-2"
          onClick={() => {
            onClose();
            openSettings("sandbox", {
              scrollTarget: "sandbox-permissions",
              opener: returnFocusRef.current,
            });
          }}
        >
          Learn more
        </button>
      </p>
      <AlertDialogFooter className="mt-1">
        <AlertDialogCancel variant="muted">Cancel</AlertDialogCancel>
        <AlertDialogAction variant="destructive" onClick={onConfirm}>
          Turn on
        </AlertDialogAction>
      </AlertDialogFooter>
    </AlertDialogContent>
  );
}

export function FullAccessConfirmDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);
  const { fullAccessAllowed } = useAccountPermissionMode();
  if (!fullAccessAllowed) return null;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <FullAccessConfirmContent
        onConfirm={() => {
          setPermissionMode("full");
          onOpenChange(false);
        }}
        onClose={() => onOpenChange(false)}
      />
    </AlertDialog>
  );
}

/** Select-style dropdown (like the MCP composer menu) for picking the permission level. Used in
 *  Settings > Sandbox and the chat settings sheet. */
export function PermissionModeDropdown({
  side = "bottom",
  align = "end",
  triggerClassName,
  sandboxControls = true,
}: {
  side?: "top" | "bottom";
  align?: "start" | "end";
  triggerClassName?: string;
  /** Off in Settings, which shows the level and the setup in their own rows. */
  sandboxControls?: boolean;
} = {}) {
  const { permissionMode, fullAccessAllowed } = useAccountPermissionMode();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [sandboxSetupOpen, setSandboxSetupOpen] = useState(false);
  const active = permissionModeOption(permissionMode);
  const ActiveIcon = active.icon;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className={cn("gap-1.5", triggerClassName)}
            aria-label="Permission level for tool calls"
          >
            <ActiveIcon className="size-3.5 shrink-0" strokeWidth={2} />
            <span className="min-w-0 flex-1 truncate text-left">
              {active.label}
            </span>
            <ChevronDown className="size-3.5 shrink-0 opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align={align}
          className="w-[calc(330px*var(--ui-space-scale,1))]"
          avoidCollisions={true}
        >
          <PermissionMenuLabel
            sandboxControls={sandboxControls}
            onOsSandboxMissing={() => setSandboxSetupOpen(true)}
          />
          <PermissionModeMenuItems
            // Defer past menu-close focus restore so the dialog focus trap holds.
            onRequestFullAccess={() =>
              setTimeout(() => setConfirmOpen(true), 0)
            }
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <FullAccessConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
      />
      <SandboxSetupDialog
        open={sandboxSetupOpen}
        onOpenChange={setSandboxSetupOpen}
      />
    </>
  );
}

/** Full access routes through a confirm dialog at the chat-page root so it survives unmount. */
export function PermissionModeComposerPill({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const { permissionMode, fullAccessAllowed } = useAccountPermissionMode();
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );
  const active = permissionModeOption(permissionMode);
  const ActiveIcon = active.icon;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          className="composer-pill-btn composer-pill-permissions"
          data-pill-label={active.label}
          aria-label="Permission level for tool calls"
          title={`${active.label}: ${active.description}`}
        >
          <span className="composer-pill-glyph">
            <ActiveIcon className="size-[calc(15px*var(--ui-space-scale,1))]" strokeWidth={2} />
          </span>
          <span>{active.label}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.5}
            className="composer-pill-caret size-[calc(15px*var(--ui-space-scale,1))]"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align="start"
        sideOffset={0}
        avoidCollisions={true}
        className="unsloth-plus-menu w-[calc(330px*var(--ui-space-scale,1))]"
      >
        <PermissionMenuLabel sandboxControls={true} />
        <PermissionModeMenuItems
          onRequestFullAccess={() =>
            setTimeout(() => setBypassConfirmOpen(true), 0)
          }
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
