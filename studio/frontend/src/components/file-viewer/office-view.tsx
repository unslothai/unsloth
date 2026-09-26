// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { repackDocxPreviewArchive } from "@/features/chat";
import { useT } from "@/i18n";
import { openLink } from "@/lib/open-link";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { cn } from "@/lib/utils";
import { useVirtualizer } from "@tanstack/react-virtual";
import { type CSSProperties, type MouseEvent, memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type DocumentKind, sheetDelimiter } from "./kind";
import { useWidth } from "./use-width";
import {
  type Deck,
  type Sheet,
  type SheetCell,
  type Slide,
  type SlideBox,
  columnName,
  readDelimited,
  readPptx,
  readXlsx,
} from "./office";

type Parsed =
  | { kind: "docx"; html: string; truncated: boolean }
  | { kind: "sheet"; sheets: Sheet[] }
  | { kind: "slides"; deck: Deck };

async function parse(file: Blob, kind: DocumentKind, name: string, contentType: string): Promise<Parsed> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (kind === "docx") {
    const { default: mammoth } = await import("mammoth");
    const repacked = repackDocxPreviewArchive(name, bytes, MAX_DOCX_PARAGRAPHS);
    const { value } = await mammoth.convertToHtml({ arrayBuffer: repacked.archive.buffer as ArrayBuffer });
    const { html, truncated } = sanitizeDocxHtml(value);
    return { kind, html, truncated: truncated || repacked.truncated };
  }
  if (kind === "slides") return { kind, deck: readPptx(bytes) };
  const delimiter = sheetDelimiter(name, contentType);
  if (delimiter) {
    // Excel's "Unicode Text" export is UTF-16 with a byte order mark.
    const encoding =
      bytes[0] === 0xff && bytes[1] === 0xfe
        ? "utf-16le"
        : bytes[0] === 0xfe && bytes[1] === 0xff
          ? "utf-16be"
          : "utf-8";
    const text = new TextDecoder(encoding).decode(bytes);
    return { kind: "sheet", sheets: [readDelimited(text, delimiter, name)] };
  }
  return { kind: "sheet", sheets: readXlsx(bytes) };
}

const DOCX_TAGS = new Set(
  "p h1 h2 h3 h4 h5 h6 strong b em i u s sup sub a img br ul ol li table thead tbody tr td th blockquote pre code span div hr".split(" "),
);
const DOCX_ATTRIBUTES = new Set(["href", "src", "alt", "id", "colspan", "rowspan"]);

// Paragraphs past this are cut before conversion, so mammoth and the DOM never see them.
const MAX_DOCX_PARAGRAPHS = 20_000;
// Elements past this are dropped: a paragraph can still convert to many.
const MAX_DOCX_ELEMENTS = 50_000;

/** mammoth writes a small vocabulary; anything else, and any link that is not a web, mail or
 *  in-document one, is dropped before the markup reaches the page. */
function sanitizeDocxHtml(html: string): { html: string; truncated: boolean } {
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html");
  const elements = doc.body.querySelectorAll("*");
  const cut = elements[MAX_DOCX_ELEMENTS];
  if (cut) {
    // Drop the cut and all that follows it at every level, so one long table keeps its first rows.
    for (let node: Element = cut; node !== doc.body; node = node.parentElement!) {
      while (node.nextSibling) node.nextSibling.remove();
    }
    cut.remove();
  }
  for (const element of Array.from(doc.body.querySelectorAll("*"))) {
    if (!DOCX_TAGS.has(element.localName)) {
      element.replaceWith(...Array.from(element.childNodes));
      continue;
    }
    for (const attr of Array.from(element.attributes)) {
      const value = attr.value.trim().toLowerCase();
      const unsafe =
        !DOCX_ATTRIBUTES.has(attr.name) ||
        (attr.name === "href" && !/^(https?:|mailto:|#)/.test(value)) ||
        (attr.name === "src" && !value.startsWith("data:image/"));
      if (unsafe) element.removeAttribute(attr.name);
    }
  }
  return { html: doc.body.innerHTML, truncated: Boolean(cut) };
}

// A Letter page's width at 96 dpi.
const DOCX_PAGE_WIDTH = 816;

function DocxView({ html, truncated, scale }: { html: string; truncated: boolean; scale: number }) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  // A pixel width, which zoom scales the same way in every engine; a percentage it does not.
  // 100% is the page, or the pane when that is narrower.
  const pageWidth = Math.max(
    200,
    Math.min(useWidth(container), DOCX_PAGE_WIDTH * useUiSpaceScale()),
  );
  const onClick = (event: MouseEvent<HTMLDivElement>) => {
    const link = (event.target as Element).closest("a");
    const href = link?.getAttribute("href");
    if (!href) return;
    event.preventDefault();
    if (href.startsWith("#")) {
      event.currentTarget.querySelector(`[id="${CSS.escape(href.slice(1))}"]`)?.scrollIntoView();
    } else {
      openLink(href);
    }
  };
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60 py-6">
      <article
        onClick={onClick}
        style={{ zoom: scale, width: pageWidth }}
        className="mx-auto min-h-full select-text bg-white px-[clamp(24px,8%,80px)] py-16 text-ui-15 leading-relaxed text-neutral-900 shadow-sm ring-1 ring-black/5 [&_a]:text-blue-700 [&_a]:underline [&_h1]:mb-3 [&_h1]:text-2xl [&_h1]:font-semibold [&_h2]:mt-5 [&_h2]:mb-2 [&_h2]:text-xl [&_h2]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2 [&_h3]:text-lg [&_h3]:font-semibold [&_img]:inline-block [&_img]:max-w-full [&_li]:my-1 [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-7 [&_p]:mb-2.5 [&_table]:my-3 [&_table]:border-collapse [&_td]:border [&_td]:border-neutral-300 [&_td]:px-2 [&_td]:py-1 [&_td]:align-top [&_th]:border [&_th]:border-neutral-300 [&_th]:px-2 [&_th]:py-1 [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-7"
        // Sanitised above: a fixed set of tags and attributes, with no scripts or handlers.
        dangerouslySetInnerHTML={{ __html: html }}
      />
      {truncated && (
        <p className="mt-4 text-center text-ui-12 text-muted-foreground">{t("library.preview.documentTruncated")}</p>
      )}
    </div>
  );
}

const ROW_HEIGHT = 28;
const DEFAULT_COLUMN_WIDTH = 100;
const ROW_HEADER_WIDTH = 48;

// Plain strings, not cn(): hundreds of cells render per scroll frame.
const CELL = "h-7 border-r border-b border-border px-2 whitespace-nowrap overflow-hidden text-ellipsis";
const HEADER = `${CELL} sticky bg-muted font-normal text-muted-foreground text-center`;

/** Columns `start` to `end` of a row, with spacer cells for those scrolled past. Memoized so rows
 *  still in view skip rendering. */
const SheetRow = memo(function SheetRow({
  r,
  row,
  columns,
  start,
  end,
  left,
  right,
}: {
  r: number;
  row: (SheetCell | undefined)[] | undefined;
  columns: number[];
  start: number;
  end: number;
  left: boolean;
  right: boolean;
}) {
  const cells = [];
  for (let i = start; i < end; i++) {
    const index = columns[i]!;
    const value = row?.[index];
    let className = CELL;
    if (value?.numeric) className += " text-right";
    if (value?.bold) className += " font-semibold";
    if (value?.italic) className += " italic";
    cells.push(
      <td key={index} title={value && value.text.length > 12 ? value.text : undefined} className={className}>
        {value?.text}
      </td>,
    );
  }
  return (
    <tr>
      <th className={`${HEADER} left-0 z-10`}>{r + 1}</th>
      {left && <td className="p-0" />}
      {cells}
      {right && <td className="p-0" />}
    </tr>
  );
});

function SheetGrid({
  sheet,
  scale,
  uiScale,
}: {
  sheet: Sheet;
  scale: number;
  /** The UI size setting, which h-7 rows and the text grow with before the zoom applies. */
  uiScale: number;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const rowHeight = ROW_HEIGHT * uiScale * scale;
  const headerWidth = ROW_HEADER_WIDTH * uiScale;
  // Per sheet, not per scroll frame.
  const { visibleColumns, visibleRows, widths, rowCount } = useMemo(() => {
    // reduce, not a spread: rows a sheet leaves out are holes, which a spread turns into NaN.
    const columns = Math.max(
      1,
      sheet.rows.reduce((max, row) => Math.max(max, row?.length ?? 0), sheet.widths.length),
    );
    const rowCount = Math.max(sheet.rows.length, 1);
    // Hidden rows and columns are skipped but keep their labels, as Excel shows them (A, C).
    const visibleColumns = Array.from({ length: columns }, (_, index) => index).filter(
      (index) => !sheet.hidden?.columns.has(index),
    );
    const visibleRows = sheet.hidden?.rows.size
      ? Array.from({ length: rowCount }, (_, index) => index).filter((index) => !sheet.hidden!.rows.has(index))
      : null;
    const widths = visibleColumns.map((index) => (sheet.widths[index] ?? DEFAULT_COLUMN_WIDTH) * uiScale);
    return { visibleColumns, visibleRows, widths, rowCount };
  }, [sheet, uiScale]);
  // A fixed layout only honours the column widths when the table has a width of its own.
  const tableWidth = useMemo(() => widths.reduce((sum, width) => sum + width, headerWidth), [widths, headerWidth]);
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: visibleRows?.length ?? rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => rowHeight,
    // The sticky column header sits above the first row.
    paddingStart: rowHeight,
    overscan: 6,
  });
  // Only the columns in view are mounted.
  const columnVirtualizer = useVirtualizer({
    horizontal: true,
    count: visibleColumns.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => widths[index]! * scale,
    // The sticky row header sits left of the first column.
    paddingStart: headerWidth * scale,
    overscan: 4,
  });
  const items = virtualizer.getVirtualItems();
  const before = ((items[0]?.start ?? 0) - rowHeight) / scale;
  const after = (virtualizer.getTotalSize() - (items.at(-1)?.end ?? 0)) / scale;
  const columnItems = columnVirtualizer.getVirtualItems();
  const left = ((columnItems[0]?.start ?? headerWidth * scale) - headerWidth * scale) / scale;
  const right = (columnVirtualizer.getTotalSize() - (columnItems.at(-1)?.end ?? headerWidth * scale)) / scale;
  const start = columnItems[0]?.index ?? 0;
  const end = (columnItems.at(-1)?.index ?? -1) + 1;
  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto">
      <table
        className="border-separate border-spacing-0 text-ui-13 tabular-nums"
        style={{ zoom: scale, tableLayout: "fixed", width: tableWidth } as CSSProperties}
      >
        <colgroup>
          <col style={{ width: headerWidth }} />
          {left > 0 && <col style={{ width: left }} />}
          {columnItems.map((item) => (
            <col key={item.index} style={{ width: widths[item.index] }} />
          ))}
          {right > 0 && <col style={{ width: right }} />}
        </colgroup>
        <thead>
          <tr>
            <th className={`${HEADER} top-0 left-0 z-20`} />
            {left > 0 && <th className="p-0" />}
            {columnItems.map((item) => {
              const index = visibleColumns[item.index]!;
              return (
                <th key={index} className={`${HEADER} top-0 z-10`}>
                  {columnName(index)}
                </th>
              );
            })}
            {right > 0 && <th className="p-0" />}
          </tr>
        </thead>
        <tbody>
          {before > 0 && <tr style={{ height: before }} />}
          {items.map((item) => {
            const r = visibleRows?.[item.index] ?? item.index;
            return (
              <SheetRow
                key={r}
                r={r}
                row={sheet.rows[r]}
                columns={visibleColumns}
                start={start}
                end={end}
                left={left > 0}
                right={right > 0}
              />
            );
          })}
          {after > 0 && <tr style={{ height: after }} />}
        </tbody>
      </table>
    </div>
  );
}

function SheetView({ sheets, tabs, scale }: { sheets: Sheet[]; tabs: boolean; scale: number }) {
  const t = useT();
  const uiScale = useUiSpaceScale();
  const [active, setActive] = useState(0);
  const sheet = sheets[active] ?? sheets[0];
  if (!sheet) return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.emptyDocument")}</p>;
  return (
    <div className="flex size-full min-h-0 flex-col overflow-hidden border-t border-border">
      {/* Keyed on everything the row height follows, so the virtualizer measures afresh. */}
      <SheetGrid key={`${active}:${scale}:${uiScale}`} sheet={sheet} scale={scale} uiScale={uiScale} />
      {(tabs || sheet.truncated) && (
        <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-t border-border px-2 py-1.5">
          {tabs &&
            sheets.map((item, index) => (
              <button
                key={index}
                type="button"
                onClick={() => setActive(index)}
                className={cn(
                  "shrink-0 rounded-md px-3 py-1 text-ui-13 transition-colors hover:bg-muted",
                  index === active ? "bg-muted font-medium text-foreground" : "text-muted-foreground",
                )}
              >
                {item.name}
              </button>
            ))}
          {sheet.truncated && (
            <span className="ml-auto shrink-0 px-2 text-ui-12 text-muted-foreground">
              {t("library.preview.sheetTruncated")}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

const TITLE_FRAME = { x: 0.06, y: 0.05, w: 0.88, h: 0.18 };
const BODY_FRAME = { x: 0.06, y: 0.26, w: 0.88, h: 0.68 };
// A title slide's placeholders, which take their place from the layout rather than the slide.
const CENTER_TITLE_FRAME = { x: 0.1, y: 0.26, w: 0.8, h: 0.24 };
const SUBTITLE_FRAME = { x: 0.15, y: 0.53, w: 0.7, h: 0.2 };
const TITLES = new Set(["title", "ctrTitle"]);

/** `mirror` for a picture, which a flip mirrors. Text is never mirrored: a vertical flip turns it
 *  upside down, as PowerPoint draws it, and a horizontal one leaves it be. */
function frameStyle(frame: NonNullable<SlideBox["frame"]>, mirror = false): CSSProperties {
  const turn = (frame.rot ?? 0) + (!mirror && frame.flipV ? 180 : 0);
  const flip = mirror && (frame.flipH || frame.flipV) ? `scale(${frame.flipH ? -1 : 1}, ${frame.flipV ? -1 : 1})` : "";
  return {
    left: `${frame.x * 100}%`,
    top: `${frame.y * 100}%`,
    width: `${frame.w * 100}%`,
    height: `${frame.h * 100}%`,
    transform: [turn ? `rotate(${turn}deg)` : "", flip].filter(Boolean).join(" ") || undefined,
  };
}

function SlideText({ box, widthPt }: { box: SlideBox; widthPt: number }) {
  const title = TITLES.has(box.placeholder ?? "");
  return (
    <>
      {box.paragraphs?.map((p, index) => (
        <p
          key={index}
          className={cn("leading-tight whitespace-pre-line", (title || p.bold) && "font-semibold", p.bullet && "pl-[1.1em] -indent-[1.1em]")}
          style={{
            fontSize: `${((p.size ?? (title ? 36 : 18)) / widthPt) * 100}cqw`,
            textAlign: p.align === "ctr" ? "center" : p.align === "r" ? "right" : undefined,
            marginBottom: "0.3em",
          }}
        >
          {p.bullet && "• "}
          {p.text}
        </p>
      ))}
    </>
  );
}

function SlideTable({ rows, caption, widthPt }: { rows: string[][]; caption?: string; widthPt: number }) {
  return (
    <table className="size-full table-fixed border-collapse" style={{ fontSize: `${(14 / widthPt) * 100}cqw` }}>
      {caption && <caption className="pb-[0.4em] text-left font-semibold">{caption}</caption>}
      <tbody>
        {rows.map((row, r) => (
          <tr key={r} className={cn(r === 0 && "font-semibold")}>
            {row.map((cell, c) => (
              <td key={c} className="border border-neutral-300 px-[0.4em] py-[0.2em] align-top whitespace-pre-wrap">
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** The object URL lives only while the slide is mounted. */
function SlideImage({ image }: { image: Blob }) {
  // Stable, or every scroll re-render would remake the URL.
  const attach = useCallback(
    (element: HTMLImageElement | null) => {
      if (!element) return;
      const url = URL.createObjectURL(image);
      element.src = url;
      return () => URL.revokeObjectURL(url);
    },
    [image],
  );
  return (
    <img
      ref={attach}
      alt=""
      decoding="async"
      className="size-full object-contain"
    />
  );
}

// Memoized: the list re-renders on every scroll frame.
const SlideFace = memo(function SlideFace({ slide, index, deck }: { slide: Slide; index: number; deck: Deck }) {
  const flow = slide.boxes.filter((box) => !box.frame && box.paragraphs);
  const titles = flow.filter((box) => TITLES.has(box.placeholder ?? ""));
  const body = flow.filter((box) => !TITLES.has(box.placeholder ?? ""));
  const titleSlide = titles.some((box) => box.placeholder === "ctrTitle");
  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="relative w-full overflow-hidden bg-white text-neutral-900 shadow-sm ring-1 ring-black/5 select-text"
        style={{ aspectRatio: `1 / ${deck.aspect}`, containerType: "inline-size" }}
      >
        {slide.boxes.map((box, boxIndex) =>
          box.frame ? (
            <div key={boxIndex} className="absolute overflow-hidden" style={frameStyle(box.frame, Boolean(box.image))}>
              {box.image ? (
                <SlideImage image={box.image} />
              ) : box.table ? (
                <SlideTable rows={box.table} caption={box.caption} widthPt={deck.widthPt} />
              ) : (
                <SlideText box={box} widthPt={deck.widthPt} />
              )}
            </div>
          ) : null,
        )}
        {titles.length > 0 && (
          <div
            className={cn("absolute flex flex-col justify-end overflow-hidden", titleSlide && "text-center")}
            style={frameStyle(titleSlide ? CENTER_TITLE_FRAME : TITLE_FRAME)}
          >
            {titles.map((box, i) => <SlideText key={i} box={box} widthPt={deck.widthPt} />)}
          </div>
        )}
        {body.length > 0 && (
          <div
            className={cn("absolute overflow-hidden", titleSlide && "text-center")}
            style={frameStyle(titleSlide ? SUBTITLE_FRAME : BODY_FRAME)}
          >
            {body.map((box, i) => <SlideText key={i} box={box} widthPt={deck.widthPt} />)}
          </div>
        )}
      </div>
      <span className="text-center text-ui-12 text-muted-foreground tabular-nums">{index + 1}</span>
    </div>
  );
});

const SLIDE_GAP = 24;
const SLIDE_LABEL = 26;

/** Only the slides near the viewport are mounted, so a deck of thousands opens as fast as a short one. */
function SlidesView({ deck, scale }: { deck: Deck; scale: number }) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  // 100% fits a slide to the pane's width; a larger zoom overflows it and scrolls sideways.
  const width = Math.max(200, useWidth(container)) * scale;
  if (!deck.slides.length) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.emptyDocument")}</p>;
  }
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60">
      {/* Keyed on the width, so the virtualizer measures afresh. */}
      <SlideList key={width} deck={deck} width={width} scrollElement={container} />
      {deck.truncated && (
        <p className="pb-6 text-center text-ui-12 text-muted-foreground">{t("library.preview.documentTruncated")}</p>
      )}
    </div>
  );
}

function SlideList({ deck, width, scrollElement }: { deck: Deck; width: number; scrollElement: HTMLElement | null }) {
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: deck.slides.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => width * deck.aspect + SLIDE_LABEL + SLIDE_GAP,
    paddingStart: SLIDE_GAP,
    overscan: 2,
  });
  return (
    <div className="relative mx-auto" style={{ width, height: virtualizer.getTotalSize() }}>
      {virtualizer.getVirtualItems().map((item) => (
        <div
          key={item.key}
          data-index={item.index}
          ref={virtualizer.measureElement}
          className="absolute top-0 left-0 w-full"
          style={{ transform: `translateY(${item.start}px)`, paddingBottom: SLIDE_GAP }}
        >
          <SlideFace slide={deck.slides[item.index]!} index={item.index} deck={deck} />
        </div>
      ))}
    </div>
  );
}

export default function OfficeView({
  file,
  kind,
  name,
  contentType,
  scale,
}: {
  file: Blob;
  kind: DocumentKind;
  name: string;
  contentType: string;
  scale: number;
}) {
  const t = useT();
  const [state, setState] = useState<{ file: Blob; parsed?: Parsed; error?: boolean } | null>(null);
  useEffect(() => {
    let cancelled = false;
    parse(file, kind, name, contentType).then(
      (parsed) => !cancelled && setState({ file, parsed }),
      () => !cancelled && setState({ file, error: true }),
    );
    return () => {
      cancelled = true;
    };
  }, [file, kind, name, contentType]);
  const current = state?.file === file ? state : null;
  if (current?.error) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  const parsed = current?.parsed;
  if (!parsed) return <Spinner className="m-auto size-6" />;
  if (parsed.kind === "docx") return <DocxView html={parsed.html} truncated={parsed.truncated} scale={scale} />;
  if (parsed.kind === "slides") return <SlidesView deck={parsed.deck} scale={scale} />;
  return <SheetView sheets={parsed.sheets} tabs={!sheetDelimiter(name, contentType)} scale={scale} />;
}
