// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createMathPlugin } from "@streamdown/math";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const math = createMathPlugin({ singleDollarTextMath: true });

test("escaped inline math in a generated list reaches KaTeX", () => {
  const markdown = [
    "$$",
    "ds^2 = -c^2dt^2 + (dx-v_sf(r_s)dt)^2 + dy^2 + dz^2",
    "$$",
    "",
    "where:",
    "",
    "- \\$v_s\\$ is the velocity of the bubble,",
    "- \\$f(r_s)\\$ is a shape function, with \\$f \\to 0\\$ far away and \\$f \\to 1\\$ inside,",
    "- \\$r_s\\$ is the radial coordinate.",
  ].join("\n");

  const html = renderToStaticMarkup(
    React.createElement(
      Streamdown,
      { mode: "static", plugins: { math } },
      preprocessLaTeX(markdown),
    ),
  );

  for (const source of ["v_s", "f(r_s)", "f \\to 0", "f \\to 1", "r_s"]) {
    assert.ok(
      html.includes(
        `<annotation encoding="application/x-tex">${source}</annotation>`,
      ),
      source,
    );
  }
  assert.ok(!html.includes("$v_s$"));
});

test("escaped math in a loose-list continuation reaches KaTeX", () => {
  const html = renderToStaticMarkup(
    React.createElement(
      Streamdown,
      { mode: "static", plugins: { math } },
      preprocessLaTeX("- item\n\n    \\$x\\$"),
    ),
  );

  assert.ok(
    html.includes('<annotation encoding="application/x-tex">x</annotation>'),
  );
  assert.ok(html.indexOf("<li") < html.indexOf("<annotation"));
  assert.ok(html.indexOf("<annotation") < html.indexOf("</li>"));
  assert.ok(!html.includes("$x$"));
});
