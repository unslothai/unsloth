// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useSettingsDialogStore } from "@/features/settings";
import { cn } from "@/lib/utils";
import {
  ChipIcon,
  CubeIcon,
  Database02Icon,
  Logout01Icon,
  PackageIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";

function HfLogo({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      fillRule="evenodd"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <path d="M16.781 3.277c2.997 1.704 4.844 4.851 4.844 8.258 0 .995-.155 1.955-.443 2.857a1.332 1.332 0 011.125.4 1.41 1.41 0 01.2 1.723c.204.165.352.385.428.632l.017.062c.06.222.12.69-.2 1.166.244.37.279.836.093 1.236-.255.57-.893 1.018-2.128 1.5l-.202.078-.131.048c-.478.173-.89.295-1.061.345l-.086.024c-.89.243-1.808.375-2.732.394-1.32 0-2.3-.36-2.923-1.067a9.852 9.852 0 01-3.18.018C9.778 21.647 8.802 22 7.494 22a11.249 11.249 0 01-2.541-.343l-.221-.06-.273-.08a16.574 16.574 0 01-1.175-.405c-1.237-.483-1.875-.93-2.13-1.501-.186-.4-.151-.867.093-1.236a1.42 1.42 0 01-.2-1.166c.069-.273.226-.516.447-.694a1.41 1.41 0 01.2-1.722c.233-.248.557-.391.917-.407l.078-.001a9.385 9.385 0 01-.44-2.85c0-3.407 1.847-6.554 4.844-8.258a9.822 9.822 0 019.687 0zM4.188 14.758c.125.687 2.357 2.35 2.14 2.707-.19.315-.796-.239-.948-.386l-.041-.04-.168-.147c-.561-.479-2.304-1.9-2.74-1.432-.43.46.119.859 1.055 1.42l.784.467.136.083c1.045.643 1.12.84.95 1.113-.188.295-3.07-2.1-3.34-1.083-.27 1.011 2.942 1.304 2.744 2.006-.2.7-2.265-1.324-2.685-.537-.425.79 2.913 1.718 2.94 1.725l.16.04.175.042c1.227.284 3.565.65 4.435-.604.673-.973.64-1.709-.248-2.61l-.057-.057c-.945-.928-1.495-2.288-1.495-2.288l-.017-.058-.025-.072c-.082-.22-.284-.639-.63-.584-.46.073-.798 1.21.12 1.933l.05.038c.977.721-.195 1.21-.573.534l-.058-.104-.143-.25c-.463-.799-1.282-2.111-1.739-2.397-.532-.332-.907-.148-.782.541zm14.842-.541c-.533.335-1.563 2.074-1.94 2.751a.613.613 0 01-.687.302.436.436 0 01-.176-.098.303.303 0 01-.049-.06l-.014-.028-.008-.02-.007-.019-.003-.013-.003-.017a.289.289 0 01-.004-.048c0-.12.071-.266.25-.427.026-.024.054-.047.084-.07l.047-.036c.022-.016.043-.032.063-.049.883-.71.573-1.81.131-1.917l-.031-.006-.056-.004a.368.368 0 00-.062.006l-.028.005-.042.014-.039.017-.028.015-.028.019-.036.027-.023.02c-.173.158-.273.428-.31.542l-.016.054s-.53 1.309-1.439 2.234l-.054.054c-.365.358-.596.69-.702 1.018-.143.437-.066.868.21 1.353.055.097.117.195.187.296.882 1.275 3.282.876 4.494.59l.286-.07.25-.074c.276-.084.736-.233 1.2-.42l.188-.077.065-.028.064-.028.124-.056.081-.038c.529-.252.964-.543.994-.827l.001-.036a.299.299 0 00-.037-.139c-.094-.176-.271-.212-.491-.168l-.045.01c-.044.01-.09.024-.136.04l-.097.035-.054.022c-.559.23-1.238.705-1.607.745h.006a.452.452 0 01-.05.003h-.024l-.024-.003-.023-.005c-.068-.016-.116-.06-.14-.142a.22.22 0 01-.005-.1c.062-.345.958-.595 1.713-.91l.066-.028c.528-.224.97-.483.985-.832v-.04a.47.47 0 00-.016-.098c-.048-.18-.175-.251-.36-.251-.785 0-2.55 1.36-2.92 1.36-.025 0-.048-.007-.058-.024a.6.6 0 01-.046-.088c-.1-.238.068-.462 1.06-1.066l.209-.126c.538-.32 1.01-.588 1.341-.831.29-.212.475-.406.503-.6l.003-.028c.008-.113-.038-.227-.147-.344a.266.266 0 00-.07-.054l-.034-.015-.013-.005a.403.403 0 00-.13-.02c-.162 0-.369.07-.595.18-.637.313-1.431.952-1.826 1.285l-.249.215-.033.033c-.08.078-.288.27-.493.386l-.071.037-.041.019a.535.535 0 01-.122.036h.005a.346.346 0 01-.031.003l.01-.001-.013.001c-.079.005-.145-.021-.19-.095a.113.113 0 01-.014-.065c.027-.465 2.034-1.991 2.152-2.642l.009-.048c.1-.65-.271-.817-.791-.493zM11.938 2.984c-4.798 0-8.688 3.829-8.688 8.55 0 .692.083 1.364.24 2.008l.008-.009c.252-.298.612-.46 1.017-.46.355.008.699.117.993.312.22.14.465.384.715.694.261-.372.69-.598 1.15-.605.852 0 1.367.728 1.562 1.383l.047.105.06.127c.192.396.595 1.139 1.143 1.68 1.06 1.04 1.324 2.115.8 3.266a8.865 8.865 0 002.024-.014c-.505-1.12-.26-2.17.74-3.186l.066-.066c.695-.684 1.157-1.69 1.252-1.912.195-.655.708-1.383 1.56-1.383.46.007.889.233 1.15.605.25-.31.495-.553.718-.694a1.87 1.87 0 01.99-.312c.357 0 .682.126.925.36.14-.61.215-1.245.215-1.898 0-4.722-3.89-8.55-8.687-8.55zm1.857 8.926l.439-.212c.553-.264.89-.383.89.152 0 1.093-.771 3.208-3.155 3.262h-.184c-2.325-.052-3.116-2.06-3.156-3.175l-.001-.087c0-1.107 1.452.586 3.25.586.716 0 1.379-.272 1.917-.526zm4.017-3.143c.45 0 .813.358.813.8 0 .441-.364.8-.813.8a.806.806 0 01-.812-.8c0-.442.364-.8.812-.8zm-11.624 0c.448 0 .812.358.812.8 0 .441-.364.8-.812.8a.806.806 0 01-.813-.8c0-.442.364-.8.813-.8zm7.79-.841c.32-.384.846-.54 1.33-.394.483.146.83.564.878 1.06.048.495-.212.97-.659 1.203-.322.168-.447-.477-.767-.585l.002-.003c-.287-.098-.772.362-.925.079a1.215 1.215 0 01.14-1.36zm-4.323 0c.322.384.377.92.14 1.36-.152.283-.64-.177-.925-.079l.003.003c-.108.036-.194.134-.273.24l-.118.165c-.11.15-.22.262-.377.18a1.226 1.226 0 01-.658-1.204c.048-.495.395-.913.878-1.059a1.262 1.262 0 011.33.394z" />
    </svg>
  );
}

function HfTokenIndicator() {
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const openDialog = useSettingsDialogStore((s) => s.openDialog);
  const hasToken = Boolean(hfToken && hfToken.trim());

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => openDialog("general")}
          aria-label={
            hasToken
              ? "Hugging Face token configured"
              : "Set Hugging Face token"
          }
          className={cn(
            "inline-flex items-center gap-1.5 rounded-[10px] px-2.5 py-1 text-[11.5px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            hasToken
              ? "bg-muted/60 text-muted-foreground hover:bg-muted"
              : "bg-[#b42323] text-white hover:bg-[#9e1e1e] dark:bg-[#5e1a1a] dark:hover:bg-[#4d1414]",
          )}
        >
          <HfLogo className="size-3.5" />
          <span>Token</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6}>
        {hasToken
          ? "Hugging Face token set"
          : "No Hugging Face token, click to set"}
      </TooltipContent>
    </Tooltip>
  );
}

function StatPill({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: IconSvgElement;
  label: string;
  value: string;
  tone?: "default" | "active";
}) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[10px] px-2.5 py-1 text-[11.5px]",
        tone === "active"
          ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          : "bg-muted/60 text-muted-foreground",
      )}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
      <span>{label}</span>
      <span className="font-semibold text-foreground/90">{value}</span>
    </div>
  );
}

export function ModelsHeader({
  cachedCount,
  localCount,
  gpuLabel,
  ramLabel,
  activeCheckpoint,
  activeGgufVariant,
  onEject,
}: {
  cachedCount: number;
  localCount: number;
  gpuLabel: string;
  ramLabel: string;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  onEject: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        <div className="inline-flex size-9 items-center justify-center rounded-[12px] bg-foreground text-background">
          <HugeiconsIcon icon={CubeIcon} strokeWidth={1.8} className="size-4" />
        </div>
        <div>
          <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-foreground">
            Hub
          </h1>
          <p className="text-[12px] text-muted-foreground">
            Discover, download, and run inference models locally.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <HfTokenIndicator />
        <StatPill
          icon={PackageIcon}
          label="Cache"
          value={String(cachedCount)}
        />
        <StatPill
          icon={Database02Icon}
          label="Local"
          value={String(localCount)}
        />
        <StatPill icon={ChipIcon} label="GPU" value={gpuLabel} />
        <StatPill icon={RamMemoryIcon} label="RAM" value={ramLabel} />

        {activeCheckpoint && (
          <div className="ml-1 inline-flex items-center gap-1.5 rounded-[10px] border border-emerald-500/20 bg-emerald-500/8 px-2 py-1 text-[11.5px] text-emerald-700 dark:text-emerald-300">
            <span
              className="size-1.5 rounded-full bg-emerald-500"
              aria-hidden="true"
            />
            <span className="max-w-[180px] truncate font-medium">
              {activeCheckpoint}
              {activeGgufVariant ? ` · ${activeGgufVariant}` : ""}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="-mr-1 h-6 gap-1 px-1.5 text-[11px] text-emerald-700 hover:bg-emerald-500/10 hover:text-emerald-800 dark:text-emerald-300"
              onClick={onEject}
            >
              <HugeiconsIcon
                icon={Logout01Icon}
                strokeWidth={1.75}
                className="size-3"
              />
              Eject
            </Button>
          </div>
        )}
      </div>
    </header>
  );
}
