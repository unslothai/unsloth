import { describe, expect, test } from "bun:test";
import { getEmbedding, cosineSimilarity, matchSkills } from "./embeddings";
import type { SkillSummary } from "./skills";

describe("embeddings", () => {
  describe("getEmbedding", () => {
    test("generates 384-dimensional embedding", async () => {
      const embedding = await getEmbedding("A test description");
      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    test("generates normalized embeddings", async () => {
      const embedding = await getEmbedding("normalized vector");
      let magnitude = 0;
      for (let i = 0; i < embedding.length; i++) {
        const val = embedding[i];
        if (val !== undefined) {
          magnitude += val * val;
        }
      }
      expect(Math.sqrt(magnitude)).toBeCloseTo(1.0, 5);
    });

    test("caches results", async () => {
      const text = "Test caching behavior";
      const embedding1 = await getEmbedding(text);
      const embedding2 = await getEmbedding(text);

      // Should be identical (from cache)
      expect(embedding2.length).toBe(embedding1.length);
      for (let i = 0; i < embedding1.length; i++) {
        expect(embedding2[i]).toBe(embedding1[i]);
      }
    });

    test("generates different embeddings for different inputs", async () => {
      const embedding1 = await getEmbedding("First description");
      const embedding2 = await getEmbedding("Different description");

      let areSame = true;
      for (let i = 0; i < embedding1.length; i++) {
        if (embedding1[i] !== embedding2[i]) {
          areSame = false;
          break;
        }
      }
      expect(areSame).toBe(false);
    });
  });

  describe("cosineSimilarity", () => {
    test("returns 1.0 for identical vectors", () => {
      const vec = new Float32Array([1, 2, 3, 4, 5]);
      expect(cosineSimilarity(vec, vec)).toBeCloseTo(1.0, 5);
    });

    test("returns 0.0 for orthogonal vectors", () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([0, 1, 0]);
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(0.0, 5);
    });

    test("returns -1.0 for opposite vectors", () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([-1, 0, 0]);
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(-1.0, 5);
    });

    test("calculates correct similarity for arbitrary vectors", () => {
      const vec1 = new Float32Array([1, 2, 3]);
      const vec2 = new Float32Array([4, 5, 6]);
      // (1*4 + 2*5 + 3*6) / (sqrt(14) * sqrt(77)) ≈ 0.9746
      expect(cosineSimilarity(vec1, vec2)).toBeCloseTo(0.9746, 3);
    });

    test("throws error for mismatched vector lengths", () => {
      const vec1 = new Float32Array([1, 2, 3]);
      const vec2 = new Float32Array([1, 2]);
      expect(() => cosineSimilarity(vec1, vec2)).toThrow("same length");
    });

    test("returns 0 for zero vectors", () => {
      const vec1 = new Float32Array([0, 0, 0]);
      const vec2 = new Float32Array([1, 2, 3]);
      expect(cosineSimilarity(vec1, vec2)).toBe(0);
    });

    test("works with real embeddings", async () => {
      const embedding1 = await getEmbedding("The cat sat on the mat");
      const embedding2 = await getEmbedding("A cat was sitting on a mat");
      const similarity = cosineSimilarity(embedding1, embedding2);

      // Similar sentences should have high similarity
      expect(similarity).toBeGreaterThan(0.7);
      expect(similarity).toBeLessThanOrEqual(1.0);
    });
  });

  describe("matchSkills", () => {
    const sampleSkills: SkillSummary[] = [
      {
        name: "git-helper",
        description: "Provides git workflow assistance, branch management, and commit message optimization",
      },
      {
        name: "pdf",
        description: "Comprehensive PDF manipulation toolkit for extracting text and tables",
      },
      {
        name: "docx",
        description: "Document creation, editing, and analysis with support for tracked changes",
      },
      {
        name: "brainstorming",
        description: "Refines rough ideas into fully-formed designs through collaborative questioning",
      },
      {
        name: "frontend-design",
        description: "Create distinctive, production-grade frontend interfaces with high design quality",
      },
    ];


    describe("task request matching", () => {
      test("matches git-related tasks", async () => {
        const matches = await matchSkills("Help me create a new branch and commit my changes", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "git-helper")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("matches PDF tasks", async () => {
        const matches = await matchSkills("Extract tables from this PDF document", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "pdf")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("matches document editing tasks", async () => {
        const matches = await matchSkills("Edit this Word document and track changes", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "docx")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("matches brainstorming tasks", async () => {
        const matches = await matchSkills("Help me refine this rough idea into a design", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "brainstorming")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("matches frontend design tasks", async () => {
        const matches = await matchSkills("Create a production-grade user interface", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "frontend-design")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });
    });

    describe("multiple skill matching", () => {
      test("can match multiple skills for complex tasks", async () => {
        const matches = await matchSkills("Design a frontend interface and help me brainstorm ideas", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "frontend-design" || m.name === "brainstorming")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("returns at most 5 skills (respects topK limit)", async () => {
        const manySkills: SkillSummary[] = Array.from({ length: 20 }, (_, i) => ({
          name: `skill-${i}`,
          description: "Test skill for matching testing purposes",
        }));

        const matches = await matchSkills("testing", manySkills);
        expect(matches.length).toBeLessThanOrEqual(5);
        expect(matches.every(m => m.name && m.description)).toBe(true);
      });
    });

    describe("edge cases", () => {
      test("returns empty array when skill list is empty", async () => {
        const matches = await matchSkills("Help me with git", []);
        expect(matches).toEqual([]);
      });

      test("returns empty array for unrelated topics", async () => {
        const matches = await matchSkills("xyzabc123qwerty456", sampleSkills);
        expect(matches).toEqual([]);
      });

      test("handles very long messages", async () => {
        const longMessage = "Create a frontend interface ".repeat(100);
        const matches = await matchSkills(longMessage, sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
      });

      test("handles messages with special characters", async () => {
        const matches = await matchSkills("Create git branch for feature work! @#$%^&*()", sampleSkills);

        expect(matches.length).toBeGreaterThan(0);
        expect(matches.some(m => m.name === "git-helper")).toBe(true);
        expect(matches.every(m => m.description)).toBe(true);
      });

      test("returns SkillSummary array with name and description", async () => {
        const matches = await matchSkills("Help with git", sampleSkills);

        expect(Array.isArray(matches)).toBe(true);
        if (matches.length > 0) {
          matches.forEach(match => {
            expect(match).toHaveProperty("name");
            expect(match).toHaveProperty("description");
            expect(typeof match.name).toBe("string");
            expect(typeof match.description).toBe("string");
          });
        }
      });
    });


    describe("consistency with original behavior", () => {
      test("returns empty array when no match", async () => {
        const matches = await matchSkills("completely unrelated query xyz123", sampleSkills);
        expect(matches).toEqual([]);
      });

      test("returns skill names as strings", async () => {
        const matches = await matchSkills("Help with git", sampleSkills);

        if (matches.length > 0) {
          matches.forEach(match => {
            expect(typeof match.name).toBe("string");
          });
        }
      });
    });
  });
});
