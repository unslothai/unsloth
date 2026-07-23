// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChevronDown,
  CircleAlert,
  CircleOff,
  Hand,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";

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
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type PermissionMode,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";

/**
 * Permission levels for tool calls. Full access stays last because it disables
 * both approval prompts and the code sandbox.
 */
export const PERMISSION_MODE_OPTIONS: readonly {
  value: PermissionMode;
  label: string;
  description: string;
  icon: typeof Hand;
}[] = [
  {
    value: "ask",
    label: "Ask for approval",
    description: "Always ask before tool calls edit files or use the internet",
    icon: Hand,
  },
  {
    value: "auto",
    label: "Approve for me",
    description: "Only ask for actions detected as potentially unsafe",
    icon: ShieldCheck,
  },
  {
    value: "off",
    label: "Run automatically",
    description: "Run tool calls without approval prompts inside the sandbox",
    icon: CircleOff,
  },
  {
    value: "full",
    label: "Full access",
    description:
      "Unrestricted: no approval prompts and the code sandbox is disabled",
    icon: CircleAlert,
  },
] as const;

export const FULL_ACCESS_WARNING =
  "Full access lets tool calls run without approval prompts or the code sandbox. They can modify or delete files, run commands, and make network requests. Enable it only when you trust the current task.";

export function permissionModeOption(mode: PermissionMode) {
  return (
    PERMISSION_MODE_OPTIONS.find((option) => option.value === mode) ??
    PERMISSION_MODE_OPTIONS[0]
  );
}

/** The option rows shared by every permission dropdown/submenu. Non-full
 *  levels apply directly; picking Full access must go through the caller's
 *  danger confirmation, so it's a separate callback. */
export function PermissionModeMenuItems({
  onRequestFullAccess,
}: {
  onRequestFullAccess: () => void;
}) {
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <>
      {PERMISSION_MODE_OPTIONS.map((option) => (
        <DropdownMenuItem
          key={option.value}
          onSelect={() => {
            if (option.value === permissionMode) {
              return;
            }
            if (option.value === "full") {
              onRequestFullAccess();
            } else {
              setPermissionMode(option.value);
            }
          }}
          className={cn(
            "items-start gap-2 py-2",
            permissionMode === option.value && "font-medium",
            option.value === "full" &&
              permissionMode === "full" &&
              "text-bypass",
          )}
        >
          <option.icon className="mt-0.5 size-4 shrink-0" strokeWidth={2} />
          <span className="flex min-w-0 flex-1 flex-col gap-0.5">
            <span className="text-[0.8125rem] leading-tight">{option.label}</span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {option.description}
            </span>
          </span>
          {permissionMode === option.value ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className="ml-auto mt-0.5 size-4 shrink-0"
            />
          ) : null}
        </DropdownMenuItem>
      ))}
    </>
  );
}

/** Danger confirmation shown before Full access turns on. Self-contained so
 *  the dropdown works outside the chat page (e.g. the Settings dialog). */
export function FullAccessConfirmDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Enable Full access?</AlertDialogTitle>
          <AlertDialogDescription>
            {FULL_ACCESS_WARNING}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
            onClick={() => {
              setPermissionMode("full");
              onOpenChange(false);
            }}
          >
            I understand
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/**
 * Select-style dropdown (like the MCP composer menu) for picking the
 * permission level. Used in General settings and the chat settings sheet.
 */
export function PermissionModeDropdown({
  side = "bottom",
  align = "end",
  triggerClassName,
}: {
  side?: "top" | "bottom";
  align?: "start" | "end";
  triggerClassName?: string;
} = {}) {
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const active = permissionModeOption(permissionMode);
  const ActiveIcon = active.icon;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className={cn(
              "gap-1.5",
              triggerClassName,
              // Last so a text color in triggerClassName cannot override it.
              permissionMode === "full" &&
                "text-bypass hover:text-bypass border-bypass/50",
            )}
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
          className="w-[300px]"
          avoidCollisions={true}
        >
          <DropdownMenuLabel>
            How should tool calls be approved?
          </DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration so the dialog's
            // focus trap isn't broken by the dropdown grabbing focus back.
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
    </>
  );
}

/**
 * Composer pill (mirrors the MCP pill) showing the current permission level
 * in the chat box; clicking opens the level dropdown. Danger-styled while
 * Full access is on. The Full access pick routes through the store-driven
 * BypassPermissionsConfirmDialog mounted at the chat-page root, so the
 * warning survives this menu unmounting.
 */
export function PermissionModeComposerPill({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );
  const active = permissionModeOption(permissionMode);
  const ActiveIcon = active.icon;
  const fullAccess = permissionMode === "full";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          className="composer-pill-btn composer-pill-permissions"
          data-pill-label={active.label}
          data-active={fullAccess ? "true" : "false"}
          data-variant={fullAccess ? "danger" : undefined}
          aria-label="Permission level for tool calls"
          title={`${active.label}: ${active.description}`}
        >
          <span className="composer-pill-glyph">
            <ActiveIcon className="size-[15px]" strokeWidth={2} />
          </span>
          <span>{active.label}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.5}
            className="composer-pill-caret size-[15px]"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        side={side}
        align="start"
        sideOffset={0}
        avoidCollisions={true}
        className="unsloth-plus-menu w-[300px]"
      >
        <DropdownMenuLabel>
          How should tool calls be approved?
        </DropdownMenuLabel>
        <PermissionModeMenuItems
          // Defer past the menu-close focus restoration (see PermissionModeDropdown).
          onRequestFullAccess={() =>
            setTimeout(() => setBypassConfirmOpen(true), 0)
          }
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
