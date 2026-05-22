// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  HtmlSvgRenderer,
  isHtmlFence,
  isSvgFence,
  parseCodeFence,
  sanitizeSvgSource,
} from "../html-svg-renderer";

describe("HtmlSvgRenderer", () => {
  it("renders an HTML preview inside a sandboxed iframe by default", () => {
    const html = "<html><body><h1>hello</h1></body></html>";
    render(<HtmlSvgRenderer language="html" source={html} />);

    const root = screen.getByTestId("html-svg-renderer");
    expect(root.getAttribute("data-active-tab")).toBe("preview");
    expect(root.getAttribute("data-language")).toBe("html");

    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    expect(iframe.tagName).toBe("IFRAME");
    // SECURITY: allow-scripts only; never allow-same-origin or
    // allow-top-navigation. If this ever changes, the preview can read
    // parent.document and exfiltrate session data.
    expect(iframe.getAttribute("sandbox")).toBe("allow-scripts");
    expect(iframe.getAttribute("sandbox")).not.toContain("allow-same-origin");
    expect(iframe.getAttribute("sandbox")).not.toContain(
      "allow-top-navigation",
    );
    expect(iframe.getAttribute("srcdoc") ?? iframe.srcdoc).toContain("hello");
  });

  it("renders an SVG preview and strips malicious <script> + on* payloads", () => {
    const malicious = `<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
  <circle cx="25" cy="25" r="20" fill="blue" onclick="alert('pwn')" />
  <script>window.parent.alert("pwn")</script>
</svg>`;

    render(<HtmlSvgRenderer language="svg" source={malicious} />);

    const previewHost = screen.getByTestId("html-svg-renderer-svg-preview");
    // The circle survives, but every script tag and on* handler must be
    // stripped by DOMPurify before the SVG is inserted into the DOM.
    const circle = previewHost.querySelector("circle");
    expect(circle).not.toBeNull();
    expect(circle?.getAttribute("onclick")).toBeNull();
    expect(previewHost.querySelector("script")).toBeNull();
    expect(previewHost.innerHTML.toLowerCase()).not.toContain("onclick");
    expect(previewHost.innerHTML.toLowerCase()).not.toContain("alert");
  });

  it("toggles between Preview and Code tabs", () => {
    const html = "<html><body>hi</body></html>";
    render(
      <HtmlSvgRenderer
        language="html"
        source={html}
        codeView={<pre data-testid="custom-code-view">{html}</pre>}
      />,
    );

    // Default tab is preview.
    expect(screen.getByTestId("html-svg-renderer-iframe")).toBeTruthy();
    expect(screen.queryByTestId("custom-code-view")).toBeNull();

    const codeTab = screen.getByRole("tab", { name: /code/i });
    act(() => {
      fireEvent.click(codeTab);
    });

    // Iframe is unmounted, custom code view appears.
    expect(screen.queryByTestId("html-svg-renderer-iframe")).toBeNull();
    expect(screen.getByTestId("custom-code-view")).toBeTruthy();

    const previewTab = screen.getByRole("tab", { name: /preview/i });
    act(() => {
      fireEvent.click(previewTab);
    });

    expect(screen.getByTestId("html-svg-renderer-iframe")).toBeTruthy();
    expect(screen.queryByTestId("custom-code-view")).toBeNull();
  });

  it("locks to the Code tab while the fence is still streaming in", () => {
    const html = "<html><body>partial";
    render(
      <HtmlSvgRenderer language="html" source={html} isIncomplete={true} />,
    );

    const root = screen.getByTestId("html-svg-renderer");
    expect(root.getAttribute("data-active-tab")).toBe("code");

    // Preview tab is rendered but disabled while incomplete so the user can
    // see the streaming tokens without flicker.
    const previewTab = screen.getByRole("tab", { name: /preview/i });
    expect(previewTab.hasAttribute("disabled")).toBe(true);
  });
});

describe("Fence helpers", () => {
  it("parses a typical markdown code fence", () => {
    const block = "```python\nprint('hi')\n```";
    const fence = parseCodeFence(block);
    expect(fence).not.toBeNull();
    expect(fence?.language).toBe("python");
    expect(fence?.source).toBe("print('hi')");
  });

  it("isSvgFence picks up explicit svg fences and html/xml fences that begin with <svg", () => {
    expect(isSvgFence({ language: "svg", source: "<svg/>" })).toBe(true);
    expect(
      isSvgFence({ language: "html", source: "<svg xmlns='...'></svg>" }),
    ).toBe(true);
    expect(
      isSvgFence({
        language: "xml",
        source: "<?xml version='1.0'?><svg></svg>",
      }),
    ).toBe(true);
    expect(isSvgFence({ language: "html", source: "<div></div>" })).toBe(false);
  });

  it("isHtmlFence is true only for non-SVG html fences", () => {
    expect(isHtmlFence({ language: "html", source: "<div></div>" })).toBe(true);
    expect(isHtmlFence({ language: "html", source: "<svg></svg>" })).toBe(
      false,
    );
    expect(isHtmlFence({ language: "python", source: "print()" })).toBe(false);
  });

  it("non-HTML / non-SVG fences are not handled by the renderer", () => {
    // The markdown pipeline only invokes HtmlSvgRenderer when these helpers
    // agree the fence is html/svg. A python fence must fall through.
    const fence = parseCodeFence("```python\nprint('hi')\n```");
    expect(fence).not.toBeNull();
    if (!fence) return;
    expect(isSvgFence(fence)).toBe(false);
    expect(isHtmlFence(fence)).toBe(false);
  });
});

describe("sanitizeSvgSource", () => {
  it("removes <script>, on* handlers, and javascript: URLs", () => {
    const malicious = `<svg xmlns="http://www.w3.org/2000/svg">
  <a href="javascript:alert(1)"><circle cx="5" cy="5" r="4" onload="alert(1)"/></a>
  <script>alert("pwn")</script>
</svg>`;

    const clean = sanitizeSvgSource(malicious).toLowerCase();
    expect(clean).not.toContain("<script");
    expect(clean).not.toContain("onload");
    expect(clean).not.toContain("javascript:");
    expect(clean).toContain("<circle");
  });

  it("drops <?xml ... ?> processing instructions", () => {
    const svg = `<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>`;
    const clean = sanitizeSvgSource(svg);
    expect(clean.startsWith("<?xml")).toBe(false);
    expect(clean).toContain("<rect");
  });
});
