// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { repackDocxAttachmentArchive } from "@/features/chat";
import { useT } from "@/i18n";
import { openLink } from "@/lib/open-link";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { cn } from "@/lib/utils";
import { useVirtualizer } from "@tanstack/react-virtual";
import { type CSSProperties, type MouseEvent, useEffect, useRef, useState } from "react";
import type { DocumentKind } from "./kind";
import { useWidth } from "./use-width";
import {
  type Deck,
  type Sheet,
  type SlideBox,
  columnName,
  readDelimited,
  readPptx,
  readXlsx,
} from "./office";

type Parsed =
  | { kind: "docx"; html: string }
  | { kind: "sheet"; sheets: Sheet[] }
  | { kind: "slides"; deck: Deck };

async function parse(file: Blob, kind: DocumentKind, name: string): Promise<Parsed> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const extension = name.split(".").pop()?.toLowerCase();
  if (kind === "docx") {
    const { default: mammoth } = await import("mammoth");
    const repacked = repackDocxAttachmentArchive(name, bytes);
    const { value } = await mammoth.convertToHtml({ arrayBuffer: repacked.buffer as ArrayBuffer });
    return { kind, html: sanitizeDocxHtml(value) };
  }
  if (kind === "slides") return { kind, deck: readPptx(bytes) };
  if (extension === "csv" || extension === "tsv") {
    // Excel's "Unicode Text" export is UTF-16 with a byte order mark.
    const encoding =
      bytes[0] === 0xff && bytes[1] === 0xfe
        ? "utf-16le"
        : bytes[0] === 0xfe && bytes[1] === 0xff
          ? "utf-16be"
          : "utf-8";
    const text = new TextDecoder(encoding).decode(bytes);
    return { kind: "sheet", sheets: [readDelimited(text, extension === "tsv" ? "\t" : ",", name)] };
  }
  return { kind: "sheet", sheets: readXlsx(bytes) };
}

const DOCX_TAGS = new Set(
  "p h1 h2 h3 h4 h5 h6 strong b em i u s sup sub a img br ul ol li table thead tbody tr td th blockquote pre code span div hr".split(" "),
);
const DOCX_ATTRIBUTES = new Set(["href", "src", "alt", "id", "colspan", "rowspan"]);

/** mammoth writes a small vocabulary; anything else, and any link that is not a web, mail or
 *  in-document one, is dropped before the markup reaches the page. */
function sanitizeDocxHtml(html: string): string {
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html");
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
  return doc.body.innerHTML;
}

// A Letter page's width at 96 dpi.
const DOCX_PAGE_WIDTH = 816;

function DocxView({ html, scale }: { html: string; scale: number }) {
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
    </div>
  );
}

const ROW_HEIGHT = 28;
const DEFAULT_COLUMN_WIDTH = 100;
const ROW_HEADER_WIDTH = 48;

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
  // reduce, not a spread: rows a sheet leaves out are holes, which a spread turns into NaN.
  const columns = Math.max(
    1,
    sheet.rows.reduce((max, row) => Math.max(max, row?.length ?? 0), sheet.widths.length),
  );
  const rowCount = Math.max(sheet.rows.length, 1);
  const widths = Array.from(
    { length: columns },
    (_, index) => (sheet.widths[index] ?? DEFAULT_COLUMN_WIDTH) * uiScale,
  );
  const headerWidth = ROW_HEADER_WIDTH * uiScale;
  // A fixed layout only honours the column widths when the table has a width of its own.
  const tableWidth = widths.reduce((sum, width) => sum + width, headerWidth);
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => rowHeight,
    // The sticky column header sits above the first row.
    paddingStart: rowHeight,
    overscan: 20,
  });
  const items = virtualizer.getVirtualItems();
  const before = ((items[0]?.start ?? 0) - rowHeight) / scale;
  const after = (virtualizer.getTotalSize() - (items.at(-1)?.end ?? 0)) / scale;
  const cell = "h-7 border-r border-b border-border px-2 whitespace-nowrap overflow-hidden text-ellipsis";
  const header = "sticky bg-muted font-normal text-muted-foreground text-center";
  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto">
      <table
        className="border-separate border-spacing-0 text-ui-13 tabular-nums"
        style={{ zoom: scale, tableLayout: "fixed", width: tableWidth } as CSSProperties}
      >
        <colgroup>
          <col style={{ width: headerWidth }} />
          {widths.map((width, index) => (
            <col key={index} style={{ width }} />
          ))}
        </colgroup>
        <thead>
          <tr>
            <th className={cn(cell, header, "top-0 left-0 z-20")} />
            {Array.from({ length: columns }, (_, index) => (
              <th key={index} className={cn(cell, header, "top-0 z-10")}>
                {columnName(index)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {before > 0 && <tr style={{ height: before }} />}
          {items.map((item) => {
            const row = sheet.rows[item.index];
            return (
              <tr key={item.index}>
                <th className={cn(cell, header, "left-0 z-10")}>{item.index + 1}</th>
                {Array.from({ length: columns }, (_, index) => {
                  const value = row?.[index];
                  return (
                    <td
                      key={index}
                      title={value && value.text.length > 12 ? value.text : undefined}
                      className={cn(
                        cell,
                        value?.numeric && "text-right",
                        value?.bold && "font-semibold",
                        value?.italic && "italic",
                      )}
                    >
                      {value?.text}
                    </td>
                  );
                })}
              </tr>
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

function frameStyle(frame: NonNullable<SlideBox["frame"]>): CSSProperties {
  return {
    left: `${frame.x * 100}%`,
    top: `${frame.y * 100}%`,
    width: `${frame.w * 100}%`,
    height: `${frame.h * 100}%`,
  };
}

function SlideText({ box, widthPt }: { box: SlideBox; widthPt: number }) {
  const title = TITLES.has(box.placeholder ?? "");
  return (
    <>
      {box.paragraphs?.map((p, index) => (
        <p
          key={index}
          className={cn("leading-tight", (title || p.bold) && "font-semibold", p.bullet && "pl-[1.1em] -indent-[1.1em]")}
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

function SlideTable({ rows, widthPt }: { rows: string[][]; widthPt: number }) {
  return (
    <table className="size-full table-fixed border-collapse" style={{ fontSize: `${(14 / widthPt) * 100}cqw` }}>
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

function SlidesView({ deck, scale }: { deck: Deck; scale: number }) {
  const t = useT();
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  // 100% fits a slide to the pane's width; a larger zoom overflows it and scrolls sideways.
  const width = Math.max(200, useWidth(container)) * scale;
  if (!deck.slides.length) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.emptyDocument")}</p>;
  }
  return (
    <div ref={setContainer} className="size-full overflow-auto bg-muted/60 py-6">
      <div className="mx-auto flex flex-col gap-6" style={{ width }}>
        {deck.slides.map((slide, index) => {
          const flow = slide.boxes.filter((box) => !box.frame && box.paragraphs);
          const titles = flow.filter((box) => TITLES.has(box.placeholder ?? ""));
          const body = flow.filter((box) => !TITLES.has(box.placeholder ?? ""));
          const titleSlide = titles.some((box) => box.placeholder === "ctrTitle");
          return (
            <div key={index} className="flex flex-col gap-1.5">
              <div
                className="relative w-full overflow-hidden bg-white text-neutral-900 shadow-sm ring-1 ring-black/5 select-text"
                style={{ aspectRatio: `1 / ${deck.aspect}`, containerType: "inline-size" }}
              >
                {slide.boxes.map((box, boxIndex) =>
                  box.frame ? (
                    <div key={boxIndex} className="absolute overflow-hidden" style={frameStyle(box.frame)}>
                      {box.image ? (
                        <img src={box.image} alt="" className="size-full object-contain" />
                      ) : box.table ? (
                        <SlideTable rows={box.table} widthPt={deck.widthPt} />
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
        })}
      </div>
    </div>
  );
}

export default function OfficeView({
  file,
  kind,
  name,
  scale,
}: {
  file: Blob;
  kind: DocumentKind;
  name: string;
  scale: number;
}) {
  const t = useT();
  const [state, setState] = useState<{ file: Blob; parsed?: Parsed; error?: boolean } | null>(null);
  useEffect(() => {
    let cancelled = false;
    parse(file, kind, name).then(
      (parsed) => !cancelled && setState({ file, parsed }),
      () => !cancelled && setState({ file, error: true }),
    );
    return () => {
      cancelled = true;
    };
  }, [file, kind, name]);
  const current = state?.file === file ? state : null;
  if (current?.error) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  const parsed = current?.parsed;
  if (!parsed) return <Spinner className="m-auto size-6" />;
  if (parsed.kind === "docx") return <DocxView html={parsed.html} scale={scale} />;
  if (parsed.kind === "slides") return <SlidesView deck={parsed.deck} scale={scale} />;
  return <SheetView sheets={parsed.sheets} tabs={!/\.(csv|tsv)$/i.test(name)} scale={scale} />;
}
