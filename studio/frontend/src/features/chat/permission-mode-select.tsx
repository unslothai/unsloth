// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDown, CircleAlert, Hand, ShieldCheck } from "lucide-react";
import type { ComponentType } from "react";
import { useEffect, useState } from "react";
import { useFullAccessAllowed } from "@/features/auth/account-session";
import { type TranslationKey, useT } from "@/i18n";

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
import { SparklesGlyph } from "@/lib/sparkles-icon";
import { MenuTickIcon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  loadSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import {
  SandboxSetupDialog,
  useSandboxSetupDialogStore,
} from "./sandbox-setup-dialog";
import {
  type PermissionMode,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";

/** Permission levels for tool calls. The stored values never change; only what they are called.
 *  Bypass permissions stays last because it disables both approval prompts and the sandbox. */
export const PERMISSION_MODE_OPTIONS: readonly {
  value: PermissionMode;
  labelKey: TranslationKey;
  descriptionKey: TranslationKey;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}[] = [
  {
    value: "ask",
    labelKey: "permissionModes.ask.label",
    descriptionKey: "permissionModes.ask.description",
    icon: Hand,
  },
  {
    value: "auto",
    labelKey: "permissionModes.auto.label",
    descriptionKey: "permissionModes.auto.description",
    icon: ShieldCheck,
  },
  {
    value: "off",
    labelKey: "permissionModes.off.label",
    descriptionKey: "permissionModes.off.description",
    icon: SparklesGlyph,
  },
  {
    value: "full",
    labelKey: "permissionModes.full.label",
    descriptionKey: "permissionModes.full.description",
    icon: CircleAlert,
  },
] as const;

export function permissionModeOption(mode: PermissionMode) {
  return (
    PERMISSION_MODE_OPTIONS.find((option) => option.value === mode) ??
    // Unknown values fall back to the default ("Auto-approve"), not row 0 ("Ask every time").
    PERMISSION_MODE_OPTIONS.find((option) => option.value === "auto") ??
    PERMISSION_MODE_OPTIONS[0]
  );
}

/** The option rows shared by every permission dropdown or submenu. Non-full levels apply
 *  directly; picking Bypass permissions must go through the caller's danger confirmation. */
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

/** True when this computer's OS sandbox is known not to cover Python and Terminal. False while
 *  unknown or when the server is too old to say, so the picker then behaves as it always did. */
function useSandboxUnavailable(): boolean {
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    let live = true;
    void loadSandboxCapability().then((capability) => {
      if (live) setUnavailable(capability !== null && !sandboxReady(capability));
    });
    return () => {
      live = false;
    };
  }, []);
  return unavailable;
}

/** "Full access in sandbox" only holds with a working OS sandbox, so picking it without one
 *  offers the setup instead of applying it; nothing is installed until the owner asks. */
export function pickSandboxedMode(
  setPermissionMode: (mode: PermissionMode) => void,
  onRequestSandboxSetup: () => void,
): Promise<void> {
  return loadSandboxCapability().then((capability) => {
    if (capability !== null && !sandboxReady(capability)) {
      onRequestSandboxSetup();
    } else {
      setPermissionMode("off");
    }
  });
}

export function PermissionModeMenuItems({
  onRequestFullAccess,
  onRequestSandboxSetup,
}: {
  onRequestFullAccess: () => void;
  onRequestSandboxSetup: () => void;
}) {
  const t = useT();
  const { permissionMode, fullAccessAllowed } = useAccountPermissionMode();
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);
  const sandboxUnavailable = useSandboxUnavailable();

  return (
    <>
      {PERMISSION_MODE_OPTIONS.filter((option) => fullAccessAllowed || option.value !== "full").map((option) => (
        <DropdownMenuItem
          key={option.value}
          onSelect={() => {
            if (option.value === permissionMode) {
              return;
            }
            if (option.value === "full") {
              onRequestFullAccess();
            } else if (option.value === "off") {
              void pickSandboxedMode(setPermissionMode, onRequestSandboxSetup);
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
            <span className="text-ui-13 leading-tight">{t(option.labelKey)}</span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {t(option.descriptionKey)}
            </span>
            {option.value === "off" && sandboxUnavailable ? (
              <span className="text-xs font-normal leading-snug text-bypass">
                {t("permissionModes.off.sandboxUnavailable")}
              </span>
            ) : null}
          </span>
          {permissionMode === option.value ? (
            <HugeiconsIcon
              icon={MenuTickIcon}
              strokeWidth={2}
              className="ml-auto mt-0.5 size-4 shrink-0"
            />
          ) : null}
        </DropdownMenuItem>
      ))}
    </>
  );
}

/** Danger confirmation shown before Bypass permissions turns on. Self-contained so the dropdown works
 *  outside the chat page (e.g. the Settings dialog). */
export function FullAccessConfirmDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);
  const { fullAccessAllowed } = useAccountPermissionMode();
  if (!fullAccessAllowed) return null;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>{t("permissionModes.bypassTitle")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("permissionModes.bypassWarning")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("permissionModes.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
            onClick={() => {
              setPermissionMode("full");
              onOpenChange(false);
            }}
          >
            {t("permissionModes.bypassConfirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** Select-style dropdown (like the MCP composer menu) for picking the permission level. Used in
 *  General settings and the chat settings sheet. */
export function PermissionModeDropdown({
  side = "bottom",
  align = "end",
  triggerClassName,
}: {
  side?: "top" | "bottom";
  align?: "start" | "end";
  triggerClassName?: string;
} = {}) {
  const t = useT();
  const { permissionMode } = useAccountPermissionMode();
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
            className={cn(
              "gap-1.5",
              triggerClassName,
              // Last so a text color in triggerClassName cannot override it.
              permissionMode === "full" &&
                "text-bypass hover:text-bypass border-bypass/50",
            )}
            aria-label={t("permissionModes.triggerLabel")}
          >
            <ActiveIcon className="size-3.5 shrink-0" strokeWidth={2} />
            <span className="min-w-0 flex-1 truncate text-left">
              {t(active.labelKey)}
            </span>
            <ChevronDown className="size-3.5 shrink-0 opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align={align}
          className="w-[calc(300px*var(--ui-space-scale,1))]"
          avoidCollisions={true}
        >
          <DropdownMenuLabel>{t("permissionModes.menuLabel")}</DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration so the dialog's focus trap is not broken by the
            // dropdown grabbing focus back.
            onRequestFullAccess={() =>
              setTimeout(() => setConfirmOpen(true), 0)
            }
            onRequestSandboxSetup={() =>
              setTimeout(() => setSandboxSetupOpen(true), 0)
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

/** Composer pill showing the current permission level in the chat box; clicking opens the level
 *  dropdown. Danger-styled while Bypass permissions is on. That pick routes through the
 *  store-driven confirm dialog mounted at the chat-page root, so the warning survives this
 *  menu unmounting. */
export function PermissionModeComposerPill({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const t = useT();
  const { permissionMode } = useAccountPermissionMode();
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );
  const setSandboxSetupOpen = useSandboxSetupDialogStore((s) => s.setOpen);
  const active = permissionModeOption(permissionMode);
  const activeLabel = t(active.labelKey);
  const ActiveIcon = active.icon;
  const fullAccess = permissionMode === "full";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          className="composer-pill-btn composer-pill-permissions"
          data-pill-label={activeLabel}
          data-active={fullAccess ? "true" : "false"}
          data-variant={fullAccess ? "danger" : undefined}
          aria-label={t("permissionModes.triggerLabel")}
          title={`${activeLabel}: ${t(active.descriptionKey)}`}
        >
          <span className="composer-pill-glyph">
            <ActiveIcon className="size-[calc(15px*var(--ui-space-scale,1))]" strokeWidth={2} />
          </span>
          <span>{activeLabel}</span>
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
        className="unsloth-plus-menu w-[calc(300px*var(--ui-space-scale,1))]"
      >
        <DropdownMenuLabel>{t("permissionModes.menuLabel")}</DropdownMenuLabel>
        <PermissionModeMenuItems
          // Defer past the menu-close focus restoration (see PermissionModeDropdown).
          onRequestFullAccess={() =>
            setTimeout(() => setBypassConfirmOpen(true), 0)
          }
          onRequestSandboxSetup={() =>
            setTimeout(() => setSandboxSetupOpen(true), 0)
          }
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
