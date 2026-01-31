import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  Book03Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link, useRouterState } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

const NAV_ITEMS = [
  { label: "Studio", href: "/studio", icon: ZapIcon, enabled: true },
  { label: "Evaluate", href: "/evaluate", enabled: false },
  { label: "Export", href: "/export", enabled: false },
  { label: "Chat", href: "/chat", enabled: true },
];

export function Navbar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const [logoHovered, setLogoHovered] = useState(false);

  return (
    <header className="sticky top-0 z-40 h-16 w-full">
      <div className="mx-auto flex h-full max-w-7xl items-center justify-between px-6">
        {/* Left: logo */}
        <div
          className="relative flex items-center gap-2.5 cursor-pointer select-none"
          onMouseEnter={() => setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
        >
          <motion.img
            src="https://unsloth.ai/cgi/image/unsloth_sticker_no_shadow_ldN4V4iydw00qSIIWDCUv.png?width=96&quality=80&format=auto"
            alt="unsloth"
            className="size-8"
            animate={{ rotate: logoHovered ? 360 : 0 }}
            transition={{ duration: 0.5, ease: [0.165, 0.84, 0.44, 1] }}
          />
          <span className="text-xl font-bold tracking-tight font-heading">
            unsloth
          </span>
          <AnimatePresence>
            {logoHovered && (
              <motion.img
                src="/Sloth emojis/large sloth wave.png"
                alt="hi!"
                className="absolute -bottom-10 left-1 size-10 pointer-events-none"
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: 10, opacity: 0 }}
                transition={{ duration: 0.25, ease: [0.215, 0.61, 0.355, 1] }}
              />
            )}
          </AnimatePresence>
        </div>

        {/* Center: pill nav */}
        <nav className="flex items-center rounded-full border border-border bg-card p-1 ring-1 ring-foreground/5">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            if (!item.enabled) {
              return (
                <span
                  key={item.href}
                  className="rounded-full px-4 py-1.5 text-sm font-medium text-muted-foreground/40 cursor-not-allowed"
                >
                  {item.label}
                </span>
              );
            }
            return (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  "rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "bg-foreground text-background"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {"icon" in item && item.icon && (
                  <HugeiconsIcon
                    icon={item.icon}
                    className="size-3.5 inline-block mr-1 -mt-px fill-current"
                  />
                )}
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Right: docs link */}
        <HoverCard openDelay={200} closeDelay={100}>
          <HoverCardTrigger asChild={true}>
            <a
              href="https://unsloth.ai/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm font-medium text-emerald-600 hover:text-emerald-700 transition-colors"
            >
              <HugeiconsIcon icon={Book03Icon} className="size-4" />
              Learn more
            </a>
          </HoverCardTrigger>
          <HoverCardContent align="end" className="w-80 p-0">
            <a
              href="https://unsloth.ai/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="group/card flex flex-col gap-1 p-4 no-underline"
            >
              <p className="text-sm font-semibold font-heading">
                Unsloth Documentation
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Guides on fine-tuning LLMs 2x faster with 70% less memory.
                Covers LoRA, QLoRA, data formatting, and deployment.
              </p>
              <span className="mt-1 flex items-center gap-1 text-xs font-medium text-emerald-600 group-hover/card:underline">
                Visit docs
                <HugeiconsIcon icon={ArrowRight01Icon} className="size-3" />
              </span>
            </a>
          </HoverCardContent>
        </HoverCard>
      </div>
    </header>
  );
}
