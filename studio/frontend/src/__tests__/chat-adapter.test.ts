/**
 * Tests for chat-adapter XML parsing — contracts §3 / §4, T3.
 *
 * Coverage:
 * - New XML (document_id + chunk_id) → citationId, documentId, backendChunkId set.
 * - Legacy XML (no durable IDs) → citationId set, documentId/backendChunkId absent.
 * - Same visible [N] across turns does NOT imply same backendChunkId.
 * - Same filename in two chunks → distinct documentId values kept.
 * - Missing attributes degrade gracefully — no throw.
 * - citationId is always the visible "N" counter, never the UUID.
 */

import {
  type ParsedChunk,
  parseChunks,
} from "@/components/assistant-ui/tool-ui-search-knowledge-base";
import { describe, expect, it } from "vitest";

// ── Tests ─────────────────────────────────────────────────────────────

describe("parseChunks — durable IDs (contracts §3/§4)", () => {
  it("new XML with document_id + chunk_id populates all three identity fields", () => {
    const xml = `
<chunk id="1" source="report.pdf" page="7" document_id="doc-abc" chunk_id="chunk-xyz">
The margin rose to 18%.
</chunk>
    `.trim();

    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts).toHaveLength(1);
    expect(parts[0].id).toBe("1");
    expect(parts[0].documentId).toBe("doc-abc");
    expect(parts[0].backendChunkId).toBe("chunk-xyz");
    expect(parts[0].source).toBe("report.pdf");
    expect(parts[0].page).toBe("7");
  });

  it("legacy XML without durable IDs leaves documentId and backendChunkId absent", () => {
    // Old XML: no document_id/chunk_id — hover-only, not preview-clickable (Q3).
    const xml = `
<chunk id="2" source="old-doc.pdf" page="3">
Legacy chunk text.
</chunk>
    `.trim();

    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts).toHaveLength(1);
    expect(parts[0].id).toBe("2");
    expect(parts[0].documentId).toBeUndefined();
    expect(parts[0].backendChunkId).toBeUndefined();
  });

  it("citationId (id) is the visible counter string, never the backend UUID", () => {
    const docId = "550e8400-e29b-41d4-a716-446655440000";
    const chunkId = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
    const xml = `<chunk id="5" source="paper.pdf" document_id="${docId}" chunk_id="${chunkId}">text</chunk>`;
    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts[0].id).toBe("5");
    expect(parts[0].id).not.toBe(docId);
    expect(parts[0].id).not.toBe(chunkId);
  });

  it("same filename in two chunks preserves distinct documentId values", () => {
    const xml = `
<chunk id="1" source="annual.pdf" document_id="doc-001" chunk_id="chunk-001">First excerpt.</chunk>
<chunk id="2" source="annual.pdf" document_id="doc-002" chunk_id="chunk-002">Second excerpt.</chunk>
    `.trim();

    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts).toHaveLength(2);
    expect(parts[0].documentId).toBe("doc-001");
    expect(parts[1].documentId).toBe("doc-002");
    expect(parts[0].documentId).not.toBe(parts[1].documentId);
  });

  it("same visible id in different turns does not imply same backendChunkId", () => {
    // Turn 1 and turn 2 both have id="1" but different backend identities.
    const turn1 = `<chunk id="1" source="a.pdf" document_id="doc-A" chunk_id="chunk-A">Turn 1.</chunk>`;
    const turn2 = `<chunk id="1" source="b.pdf" document_id="doc-B" chunk_id="chunk-B">Turn 2.</chunk>`;

    const p1: ParsedChunk[] = parseChunks(turn1);
    const p2: ParsedChunk[] = parseChunks(turn2);
    expect(p1[0].id).toBe(p2[0].id); // both "1"
    expect(p1[0].backendChunkId).not.toBe(p2[0].backendChunkId);
    expect(p1[0].documentId).not.toBe(p2[0].documentId);
  });

  it("missing chunk_id only (partial durable attrs) → backendChunkId absent", () => {
    const xml = `<chunk id="3" source="x.pdf" document_id="doc-XYZ">text</chunk>`;
    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts[0].documentId).toBe("doc-XYZ");
    expect(parts[0].backendChunkId).toBeUndefined();
  });

  it("multiple new-format chunks all carry independent IDs", () => {
    const xml = `
<chunk id="1" source="a.pdf" document_id="doc-1" chunk_id="ck-1">A</chunk>
<chunk id="2" source="b.pdf" document_id="doc-2" chunk_id="ck-2">B</chunk>
<chunk id="3" source="c.pdf" document_id="doc-3" chunk_id="ck-3">C</chunk>
    `.trim();

    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts).toHaveLength(3);
    const docIds = new Set(parts.map((p) => p.documentId));
    const backendChunkIds = new Set(parts.map((p) => p.backendChunkId));
    const citationIds = new Set(parts.map((p) => p.id));
    expect(docIds.size).toBe(3);
    expect(backendChunkIds.size).toBe(3);
    expect(citationIds.size).toBe(3);
  });

  it("empty XML returns empty array without throwing", () => {
    expect(parseChunks("")).toHaveLength(0);
    expect(parseChunks("No chunks here.")).toHaveLength(0);
  });

  it("XML entity encoding in source attribute is decoded", () => {
    // &amp; should decode to & in source (decodeXml in parseChunks)
    const xml = `<chunk id="1" source="report &amp; summary.pdf" document_id="doc-1" chunk_id="ck-1">text</chunk>`;
    const parts: ParsedChunk[] = parseChunks(xml);
    expect(parts).toHaveLength(1);
    expect(parts[0].id).toBe("1");
    expect(parts[0].source).toBe("report & summary.pdf");
  });
});
