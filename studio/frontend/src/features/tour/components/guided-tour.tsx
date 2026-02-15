import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Cancel01Icon,
  CheckmarkCircle01Icon,
} from "@hugeicons/core-free-icons";
import { Dialog as DialogPrimitive } from "radix-ui";
import { AnimatePresence, motion } from "motion/react";
import {
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { cssEscape, toRect } from "../lib/dom";
import { computeCardPos, padded, pickPlacement } from "../lib/layout";
import type { Placement, Rect, TourStep } from "../types";

// (types + layout/dom helpers live in ../types and ../lib)

function SpotlightOverlay({
  rect,
  vw,
  vh,
  maskId,
}: {
  rect: Rect | null;
  vw: number;
  vh: number;
  maskId: string;
}) {
  const hole = rect ?? { x: vw / 2 - 140, y: vh / 2 - 90, w: 280, h: 180 };
  const r = 22;

  return (
    <svg
      className="absolute inset-0 size-full"
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      aria-hidden={true}
    >
      <defs>
        <radialGradient id={`${maskId}-v`} cx="50%" cy="45%" r="80%">
          <stop offset="0%" stopColor="rgba(6, 9, 15, 0.35)" />
          <stop offset="55%" stopColor="rgba(6, 9, 15, 0.65)" />
          <stop offset="100%" stopColor="rgba(6, 9, 15, 0.88)" />
        </radialGradient>
        <mask id={maskId}>
          <rect x="0" y="0" width={vw} height={vh} fill="white" />
          <motion.rect
            x={hole.x}
            y={hole.y}
            width={hole.w}
            height={hole.h}
            rx={r}
            fill="black"
            transition={{ type: "spring", stiffness: 260, damping: 30 }}
          />
        </mask>
      </defs>
      <rect
        x="0"
        y="0"
        width={vw}
        height={vh}
        fill={`url(#${maskId}-v)`}
        mask={`url(#${maskId})`}
      />
    </svg>
  );
}

export function GuidedTour({
  open,
  onOpenChange,
  steps,
  onSkip,
  onComplete,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  steps: TourStep[];
  onSkip: () => void;
  onComplete: () => void;
}) {
  const maskId = `${useId()}-tour-mask`;
  const [idx, setIdx] = useState(0);
  const [vw, setVw] = useState(0);
  const [vh, setVh] = useState(0);
  const [targetRect, setTargetRect] = useState<Rect | null>(null);
  const [placement, setPlacement] = useState<Placement>("right");
  const [cardPos, setCardPos] = useState<{ left: number; top: number }>({
    left: 12,
    top: 12,
  });
  const cardRef = useRef<HTMLDivElement>(null);
  const closeLockRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const lastRectRef = useRef<Rect | null>(null);

  const step = steps[idx] ?? null;
  const total = steps.length;
  const isLast = idx === total - 1;

  const spotlightRect = useMemo(() => {
    if (!targetRect || !vw || !vh) return null;
    const pad = step?.target === "navbar" ? 4 : 14;
    return padded(targetRect, pad, vw, vh);
  }, [step?.target, targetRect, vw, vh]);

  useEffect(() => {
    if (!open) return;
    setIdx(0);
    closeLockRef.current = false;
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onResize() {
      setVw(window.innerWidth);
      setVh(window.innerHeight);
    }
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [open]);

  useEffect(() => {
    if (!open || !step) return;

    const sel = `[data-tour="${cssEscape(step.target)}"]`;
    const found = document.querySelector(sel);
    if (!(found instanceof HTMLElement)) {
      setTargetRect(null);
      return;
    }
    const el = found;

    el.scrollIntoView({
      block: "center",
      inline: "center",
      behavior: "smooth",
    });

    let raf = 0;
    let t = 0;

    function rectChanged(a: Rect | null, b: Rect): boolean {
      if (!a) return true;
      return (
        Math.abs(a.x - b.x) > 0.5 ||
        Math.abs(a.y - b.y) > 0.5 ||
        Math.abs(a.w - b.w) > 0.5 ||
        Math.abs(a.h - b.h) > 0.5
      );
    }

    function read() {
      const r = el.getBoundingClientRect();
      const next = toRect(r);
      const prev = lastRectRef.current;
      if (rectChanged(prev, next)) {
        lastRectRef.current = next;
        setTargetRect(next);
      }
    }

    function schedule() {
      if (rafRef.current != null) return;
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        read();
      });
    }

    raf = window.requestAnimationFrame(read);
    t = window.setTimeout(schedule, 240);

    const ro = new ResizeObserver(() => schedule());
    ro.observe(el);
    window.addEventListener("scroll", schedule, { capture: true, passive: true });
    window.addEventListener("resize", schedule, { passive: true });

    return () => {
      window.cancelAnimationFrame(raf);
      window.clearTimeout(t);
      ro.disconnect();
      window.removeEventListener("scroll", schedule, true);
      window.removeEventListener("resize", schedule);
      if (rafRef.current != null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [open, step]);

  useLayoutEffect(() => {
    if (!open || !spotlightRect || !vw || !vh) return;
    const card = cardRef.current?.getBoundingClientRect();
    if (!card) return;

    const gap = 14;
    const picked = pickPlacement(spotlightRect, { w: card.width, h: card.height }, vw, vh, gap);
    setPlacement(picked);
    setCardPos(computeCardPos(picked, spotlightRect, { w: card.width, h: card.height }, vw, vh, gap));
  }, [open, spotlightRect, vw, vh, idx]);

  function requestClose(reason: "skip" | "complete") {
    if (closeLockRef.current) return;
    closeLockRef.current = true;
    if (reason === "skip") onSkip();
    else onComplete();
    onOpenChange(false);
  }

  return (
    <DialogPrimitive.Root
      open={open}
      onOpenChange={(v) => {
        if (v) onOpenChange(true);
        else requestClose("skip");
      }}
      modal={true}
    >
      <DialogPrimitive.Portal>
        <AnimatePresence>
          {open && (
            <>
              <DialogPrimitive.Overlay asChild>
                <motion.div
                  className="fixed inset-0 z-50"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.18 }}
                >
                  <SpotlightOverlay rect={spotlightRect} vw={vw} vh={vh} maskId={maskId} />
                  {spotlightRect && (
                    <motion.div
                      className="fixed z-[51] pointer-events-none rounded-[22px] ring-1 ring-white/10"
                      initial={false}
                      animate={{
                        left: spotlightRect.x,
                        top: spotlightRect.y,
                        width: spotlightRect.w,
                        height: spotlightRect.h,
                        boxShadow:
                          "0 0 0 1px rgba(34, 211, 238, 0.12), 0 0 0 6px rgba(16, 185, 129, 0.08), 0 18px 90px rgba(0,0,0,0.55)",
                      }}
                      transition={{ type: "spring", stiffness: 260, damping: 30 }}
                    />
                  )}
                </motion.div>
              </DialogPrimitive.Overlay>

              <DialogPrimitive.Content
                onPointerDownOutside={(e) => e.preventDefault()}
                onInteractOutside={(e) => e.preventDefault()}
                className={cn(
                  "fixed z-[52] outline-none",
                  "w-[min(420px,calc(100vw-1.5rem))]",
                )}
                style={{
                  left: cardPos.left,
                  top: cardPos.top,
                }}
              >
                <motion.div
                  ref={cardRef}
                  initial={{ opacity: 0, scale: 0.985, y: 8 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.99, y: 10 }}
                  transition={{ duration: 0.22, ease: [0.165, 0.84, 0.44, 1] }}
                  className={cn(
                    "relative overflow-hidden rounded-[28px] corner-squircle",
                    "bg-white/95 text-foreground ring-1 ring-black/10",
                    "shadow-[0_30px_120px_rgba(0,0,0,0.35)]",
                  )}
                  style={{
                    fontFamily: "'Figtree Variable', ui-sans-serif, sans-serif",
                  }}
                >
                  <div
                    className={cn(
                      "absolute z-10 size-3 rotate-45 rounded-[3px] bg-white/95 ring-1 ring-black/10",
                      placement === "right" &&
                        "-left-1 top-1/2 -translate-y-1/2",
                      placement === "left" &&
                        "-right-1 top-1/2 -translate-y-1/2",
                      placement === "bottom" &&
                        "left-1/2 -top-1 -translate-x-1/2",
                      placement === "top" &&
                        "left-1/2 -bottom-1 -translate-x-1/2",
                    )}
                    aria-hidden={true}
                  />
                  <div className="absolute inset-x-0 top-0 h-20 bg-gradient-to-b from-emerald-400/18 via-cyan-300/6 to-transparent" />
                  <div className="absolute -left-14 -top-16 size-44 rounded-full bg-emerald-400/20 blur-2xl" />
                  <div className="absolute -right-14 -bottom-16 size-44 rounded-full bg-cyan-300/18 blur-2xl" />

                  <div className="relative p-5">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="inline-flex items-center gap-2 rounded-full bg-black/[0.04] px-2.5 py-1 text-[10px] font-mono text-foreground/60 ring-1 ring-black/10">
                          {idx + 1}/{total}
                          <span className="size-1 rounded-full bg-emerald-500/70" />
                          guided tour
                        </div>
                        <DialogPrimitive.Title
                          className="mt-2 text-[18px] leading-tight"
                          style={{ fontFamily: "var(--font-serif)" }}
                        >
                          {step?.title ?? "Quick tour"}
                        </DialogPrimitive.Title>
                        <DialogPrimitive.Description className="mt-1.5 text-sm leading-relaxed text-foreground/70">
                          {step?.body ?? "Let’s get you oriented."}
                        </DialogPrimitive.Description>
                      </div>

                      <Button
                        variant="ghost"
                        size="icon-sm"
                        className="text-foreground/60 hover:text-foreground hover:bg-black/[0.05]"
                        onClick={() => requestClose("skip")}
                        aria-label="Skip tour"
                      >
                        <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
                      </Button>
                    </div>

                    <div className="mt-5 flex items-center justify-between gap-3">
                      <Button
                        variant="ghost"
                        className="text-foreground/60 hover:text-foreground hover:bg-black/[0.05]"
                        onClick={() => requestClose("skip")}
                      >
                        Skip
                      </Button>

                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          className="border-black/10 bg-white/70 text-foreground hover:bg-white hover:text-foreground"
                          disabled={idx === 0}
                          onClick={() => setIdx((i) => Math.max(0, i - 1))}
                        >
                          <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                          Back
                        </Button>
                        {isLast ? (
                          <Button
                            variant="dark"
                            className="bg-gradient-to-r from-emerald-500 to-cyan-400 text-white hover:from-emerald-600 hover:to-cyan-500"
                            onClick={() => requestClose("complete")}
                          >
                            <HugeiconsIcon icon={CheckmarkCircle01Icon} className="size-4" />
                            Done
                          </Button>
                        ) : (
                          <Button
                            variant="dark"
                            className="bg-gradient-to-r from-emerald-500 to-cyan-400 text-white hover:from-emerald-600 hover:to-cyan-500"
                            onClick={() => setIdx((i) => Math.min(total - 1, i + 1))}
                          >
                            Next
                            <HugeiconsIcon icon={ArrowRight01Icon} className="size-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="h-px bg-gradient-to-r from-transparent via-black/10 to-transparent" />
                  <div className="px-5 py-3 text-[11px] text-foreground/55">
                    Tip: `Esc` skips. Tour blocks clicks so you can read.
                  </div>
                </motion.div>
              </DialogPrimitive.Content>
            </>
          )}
        </AnimatePresence>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
