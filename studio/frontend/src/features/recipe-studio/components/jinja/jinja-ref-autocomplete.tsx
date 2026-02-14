import {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ChangeEvent,
  type FocusEvent,
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { AvailableRefItem } from "../../utils/variables";

type CaretAnchor = { x: number; y: number; height: number };

const MAX_RESULTS = 50;

function isInViewport(el: HTMLElement): boolean {
  const boundsEl = el.closest(".react-flow") as HTMLElement | null;
  const bounds = boundsEl?.getBoundingClientRect() ?? {
    left: 0,
    top: 0,
    right: window.innerWidth,
    bottom: window.innerHeight,
  };
  const rect = el.getBoundingClientRect();
  return (
    rect.bottom >= bounds.top &&
    rect.top <= bounds.bottom &&
    rect.right >= bounds.left &&
    rect.left <= bounds.right
  );
}

function getJinjaContext(
  value: string,
  cursor: number,
): { start: number; replaceEnd: number; query: string } | null {
  if (cursor < 0) return null;

  const openIdx = value.lastIndexOf("{{", Math.max(0, cursor - 1));
  if (openIdx === -1) return null;

  const closeIdx = value.indexOf("}}", openIdx + 2);
  if (closeIdx !== -1 && closeIdx < cursor) return null;

  return {
    start: openIdx,
    replaceEnd: closeIdx === -1 ? cursor : closeIdx + 2,
    query: value.slice(openIdx + 2, cursor).trim(),
  };
}

function getItemIcon(item: AvailableRefItem) {
  if (item.kind === "expression") return FunctionIcon;
  if (item.kind === "seed") return Plant01Icon;
  if (item.kind === "llm") {
    if (item.subtype === "structured") return CodeIcon;
    if (item.subtype === "code") return CodeSimpleIcon;
    if (item.subtype === "judge") return BalanceScaleIcon;
    return PencilEdit02Icon;
  }
  if (item.subtype === "category") return Tag01Icon;
  if (item.subtype === "subcategory") return TagsIcon;
  if (item.subtype === "gaussian") return Parabola02Icon;
  if (item.subtype === "uniform" || item.subtype === "bernoulli") return EqualSignIcon;
  if (item.subtype === "datetime" || item.subtype === "timedelta") return Clock01Icon;
  if (item.subtype === "uuid") return FingerPrintIcon;
  if (item.subtype === "person" || item.subtype === "person_from_faker") return UserAccountIcon;
  return Tag01Icon;
}

function useJinjaRefAutocomplete<T extends HTMLInputElement | HTMLTextAreaElement>(
  value: string,
  onValueChange: (value: string) => void,
  items: AvailableRefItem[],
  suppress: boolean,
) {
  const fieldRef = useRef<T | null>(null);
  const [focused, setFocused] = useState(false);
  const [cursor, setCursor] = useState<number | null>(null);
  const [anchor, setAnchor] = useState<CaretAnchor | null>(null);
  const [inView, setInView] = useState(true);
  const ctx = useMemo(() => {
    if (!focused || cursor == null) return null;
    return getJinjaContext(value, cursor);
  }, [focused, cursor, value]);

  const filtered = useMemo(() => {
    if (!ctx) return [];
    const q = ctx.query.toLowerCase();
    const next = q
      ? items.filter((v) => v.ref.toLowerCase().includes(q))
      : items.slice();
    return next.slice(0, MAX_RESULTS);
  }, [ctx, items]);

  const open = !suppress && inView && Boolean(ctx && anchor) && items.length > 0;

  const getCaretAnchor = useCallback((el: T, pos: number) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const mirror = document.createElement("div");

    mirror.style.position = "fixed";
    mirror.style.left = `${rect.left}px`;
    mirror.style.top = `${rect.top}px`;
    mirror.style.visibility = "hidden";
    mirror.style.pointerEvents = "none";
    mirror.style.whiteSpace = el instanceof HTMLTextAreaElement ? "pre-wrap" : "pre";
    mirror.style.wordBreak = "break-word";
    mirror.style.boxSizing = style.boxSizing;
    mirror.style.width = `${rect.width}px`;
    mirror.style.height = `${rect.height}px`;
    mirror.style.overflow = "auto";
    mirror.style.fontFamily = style.fontFamily;
    mirror.style.fontSize = style.fontSize;
    mirror.style.fontWeight = style.fontWeight;
    mirror.style.letterSpacing = style.letterSpacing;
    mirror.style.lineHeight = style.lineHeight;
    mirror.style.padding = style.padding;
    mirror.style.border = style.border;
    mirror.style.textTransform = style.textTransform;
    mirror.style.textIndent = style.textIndent;

    const content = el.value ?? "";
    const before = content.slice(0, pos);
    const after = content.slice(pos) || ".";
    mirror.textContent = before;

    const span = document.createElement("span");
    span.textContent = after;
    mirror.appendChild(span);

    document.body.appendChild(mirror);
    mirror.scrollTop = (el as unknown as { scrollTop?: number }).scrollTop ?? 0;
    mirror.scrollLeft = (el as unknown as { scrollLeft?: number }).scrollLeft ?? 0;

    const spanRect = span.getBoundingClientRect();
    document.body.removeChild(mirror);

    let height = spanRect.height;
    if (!Number.isFinite(height) || height <= 0) {
      const lhRaw = style.lineHeight;
      if (lhRaw && lhRaw !== "normal") {
        height = Number.parseFloat(lhRaw);
      } else {
        height = Number.parseFloat(style.fontSize) * 1.2;
      }
    }

    return {
      x: spanRect.left - rect.left,
      y: spanRect.top - rect.top,
      height,
    };
  }, []);

  const captureCursor = useCallback((el: T | null) => {
    if (!el) return;
    const pos = el.selectionStart;
    if (typeof pos !== "number") {
      setCursor(null);
      setAnchor(null);
      setInView(true);
      return;
    }
    setCursor(pos);
    setAnchor(getCaretAnchor(el, pos));
    setInView(isInViewport(el));
  }, [getCaretAnchor]);

  useEffect(() => {
    if (suppress) return;
    if (!focused) return;
    requestAnimationFrame(() => {
      captureCursor(fieldRef.current);
    });
  }, [captureCursor, focused, suppress]);

  const insertRef = useCallback(
    (refName: string) => {
      if (!ctx) return;
      const replacement = `{{ ${refName} }}`;
      const next =
        value.slice(0, ctx.start) + replacement + value.slice(ctx.replaceEnd);
      onValueChange(next);

      const nextCursor = ctx.start + replacement.length;
      requestAnimationFrame(() => {
        const el = fieldRef.current;
        if (!el) return;
        el.focus();
        el.setSelectionRange(nextCursor, nextCursor);
        captureCursor(el);
      });
    },
    [captureCursor, ctx, onValueChange, value],
  );

  const onFocus = useCallback(
    (event: FocusEvent<T>) => {
      setFocused(true);
      captureCursor(event.currentTarget);
    },
    [captureCursor],
  );

  const onBlur = useCallback(() => {
    setFocused(false);
  }, []);

  const onSelect = useCallback(
    (event: React.SyntheticEvent<T>) => {
      captureCursor(event.currentTarget);
    },
    [captureCursor],
  );

  return {
    fieldRef,
    open,
    filtered,
    insertRef,
    onFocus,
    onBlur,
    onSelect,
    captureCursor,
    anchor,
  };
}

function RefList({
  items,
  onPick,
}: {
  items: AvailableRefItem[];
  onPick: (value: string) => void;
}): ReactElement {
  if (items.length === 0) {
    return (
      <div className="px-2 py-2 text-xs text-muted-foreground">
        No matches
      </div>
    );
  }

  return (
    <div className="max-h-64 overflow-auto p-1">
      {items.map((item) => (
        <button
          key={item.ref}
          type="button"
          className="corner-squircle flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm hover:bg-accent hover:text-accent-foreground"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => onPick(item.ref)}
        >
          <span className="corner-squircle flex size-8 shrink-0 items-center justify-center rounded-md border border-border/60 bg-muted/30">
            <HugeiconsIcon icon={getItemIcon(item)} strokeWidth={2} className="size-4" />
          </span>
          <span className="min-w-0 flex-1 font-mono text-[13px]">
            <span className="block truncate">{item.ref}</span>
          </span>
          <span className="corner-squircle shrink-0 rounded-md bg-muted/40 px-2 py-1 text-[11px] text-muted-foreground">
            {item.valueType ?? `${item.kind}:${item.subtype}`}
          </span>
        </button>
      ))}
    </div>
  );
}

export function JinjaRefInput({
  value,
  onValueChange,
  items,
  suppress = false,
  id,
  placeholder,
  className,
  disabled,
}: {
  value: string;
  onValueChange: (value: string) => void;
  items: AvailableRefItem[];
  suppress?: boolean;
  id?: string;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}): ReactElement {
  const {
    fieldRef,
    open,
    filtered,
    insertRef,
    onFocus,
    onBlur,
    onSelect,
    captureCursor,
    anchor,
  } = useJinjaRefAutocomplete<HTMLInputElement>(value, onValueChange, items, suppress);

  const onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      onValueChange(event.target.value);
      captureCursor(event.target);
    },
    [captureCursor, onValueChange],
  );

  return (
    <Popover open={open}>
      <div className="relative">
        <PopoverTrigger asChild={true}>
          <Input
            ref={fieldRef}
            id={id}
            disabled={disabled}
            className={cn(className)}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            onFocus={onFocus}
            onBlur={onBlur}
            onSelect={onSelect}
          />
        </PopoverTrigger>
        {anchor && (
          <PopoverAnchor asChild={true}>
            <span
              className="pointer-events-none absolute"
              style={{
                left: anchor.x,
                top: anchor.y + anchor.height,
                width: 1,
                height: 1,
              }}
            />
          </PopoverAnchor>
        )}
      </div>
      <PopoverContent
        align="start"
        side="bottom"
        sideOffset={8}
        className="corner-squircle nodrag nopan w-[360px] gap-0 rounded-xl p-1"
        onOpenAutoFocus={(event) => event.preventDefault()}
        onCloseAutoFocus={(event) => event.preventDefault()}
      >
        <RefList items={filtered} onPick={insertRef} />
      </PopoverContent>
    </Popover>
  );
}

export function JinjaRefTextarea({
  value,
  onValueChange,
  items,
  suppress = false,
  id,
  placeholder,
  className,
  disabled,
}: {
  value: string;
  onValueChange: (value: string) => void;
  items: AvailableRefItem[];
  suppress?: boolean;
  id?: string;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}): ReactElement {
  const {
    fieldRef,
    open,
    filtered,
    insertRef,
    onFocus,
    onBlur,
    onSelect,
    captureCursor,
    anchor,
  } = useJinjaRefAutocomplete<HTMLTextAreaElement>(value, onValueChange, items, suppress);

  const onChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => {
      onValueChange(event.target.value);
      captureCursor(event.target);
    },
    [captureCursor, onValueChange],
  );

  return (
    <Popover open={open}>
      <div className="relative">
        <PopoverTrigger asChild={true}>
          <Textarea
            ref={fieldRef}
            id={id}
            disabled={disabled}
            className={cn(className)}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            onFocus={onFocus}
            onBlur={onBlur}
            onSelect={onSelect}
          />
        </PopoverTrigger>
        {anchor && (
          <PopoverAnchor asChild={true}>
            <span
              className="pointer-events-none absolute"
              style={{
                left: anchor.x,
                top: anchor.y + anchor.height,
                width: 1,
                height: 1,
              }}
            />
          </PopoverAnchor>
        )}
      </div>
      <PopoverContent
        align="start"
        side="bottom"
        sideOffset={8}
        className="corner-squircle nodrag nopan w-[300px] gap-0 rounded-xl p-1"
        onOpenAutoFocus={(event) => event.preventDefault()}
        onCloseAutoFocus={(event) => event.preventDefault()}
      >
        <RefList items={filtered} onPick={insertRef} />
      </PopoverContent>
    </Popover>
  );
}
