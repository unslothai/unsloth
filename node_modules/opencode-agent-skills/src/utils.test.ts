import { describe, test, expect } from "bun:test";
import { levenshtein, findClosestMatch } from "./utils";

describe("levenshtein", () => {
  test("identical strings have distance 0", () => {
    expect(levenshtein("hello", "hello")).toBe(0);
  });

  test("completely different strings have high distance", () => {
    expect(levenshtein("abc", "xyz")).toBe(3);
  });

  test("single character difference", () => {
    expect(levenshtein("cat", "bat")).toBe(1);
  });

  test("insertion", () => {
    expect(levenshtein("cat", "cats")).toBe(1);
  });

  test("deletion", () => {
    expect(levenshtein("cats", "cat")).toBe(1);
  });

  test("substitution", () => {
    expect(levenshtein("cat", "cut")).toBe(1);
  });

  test("case sensitive", () => {
    expect(levenshtein("Cat", "cat")).toBe(1);
  });
});

describe("findClosestMatch", () => {
  test("returns null for empty candidate list", () => {
    expect(findClosestMatch("test", [])).toBe(null);
  });

  test("exact match returns the match", () => {
    const candidates = ["brainstorming", "git-helper", "pdf"];
    expect(findClosestMatch("pdf", candidates)).toBe("pdf");
  });

  test("prefix match - user types partial skill name", () => {
    const candidates = ["brainstorming", "git-helper", "pdf"];
    expect(findClosestMatch("git", candidates)).toBe("git-helper");
  });

  test("prefix match - longer match", () => {
    const candidates = ["brainstorming", "git-helper", "pdf"];
    expect(findClosestMatch("brainstorm", candidates)).toBe("brainstorming");
  });

  test("typo correction via Levenshtein", () => {
    const candidates = ["pattern", "git-helper", "pdf"];
    expect(findClosestMatch("patern", candidates)).toBe("pattern");
  });

  test("case insensitive matching", () => {
    const candidates = ["Brainstorming", "Git-Helper", "PDF"];
    expect(findClosestMatch("brainstorm", candidates)).toBe("Brainstorming");
  });

  test("case insensitive exact match", () => {
    const candidates = ["Brainstorming", "Git-Helper", "PDF"];
    expect(findClosestMatch("PDF", candidates)).toBe("PDF");
  });

  test("substring match", () => {
    const candidates = ["document-processor", "git-helper", "pdf-reader"];
    expect(findClosestMatch("pdf", candidates)).toBe("pdf-reader");
  });

  test("no close matches below threshold returns null", () => {
    const candidates = ["brainstorming", "git-helper", "pdf"];
    expect(findClosestMatch("xyzabc", candidates)).toBe(null);
  });

  test("multiple similar candidates returns best match", () => {
    const candidates = ["test", "testing", "tests"];
    expect(findClosestMatch("test", candidates)).toBe("test");
  });

  test("prefix matching beats substring matching", () => {
    const candidates = ["pdf-reader", "reader-pdf"];
    expect(findClosestMatch("pdf", candidates)).toBe("pdf-reader");
  });

  test("handles hyphenated names", () => {
    const candidates = ["git-helper", "github-actions", "gitlab-ci"];
    expect(findClosestMatch("git", candidates)).toBe("git-helper");
  });

  test("script path matching", () => {
    const candidates = ["build.sh", "scripts/deploy.sh", "tools/build.sh"];
    expect(findClosestMatch("deploy", candidates)).toBe("scripts/deploy.sh");
  });

  test("typo in script name", () => {
    const candidates = ["build.sh", "deploy.sh", "test.sh"];
    expect(findClosestMatch("biuld.sh", candidates)).toBe("build.sh");
  });
});
