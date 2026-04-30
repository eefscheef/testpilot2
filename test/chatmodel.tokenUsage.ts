import fs from "fs";
import os from "os";
import path from "path";
import { expect } from "chai";
import { PerformanceMeasurer } from "../benchmark/performanceMeasurer";
import { TestResultCollector } from "../benchmark/testResultCollector";
import { ChatModel } from "../src/chatmodel";
import { APIFunction } from "../src/exploreAPI";
import { defaultPromptOptions, Prompt } from "../src/promptCrafting";

function extractUsage(payload: object) {
  return (ChatModel as any).extractUsage(payload);
}

describe("ChatModel extractUsage cache fields", () => {
  it("reads OpenAI prompt_tokens_details.cached_tokens as cacheReadInputTokens", () => {
    const usage = extractUsage({
      usage: {
        prompt_tokens: 2048,
        completion_tokens: 32,
        total_tokens: 2080,
        prompt_tokens_details: { cached_tokens: 1700 },
      },
    });
    expect(usage?.inputTokens).to.equal(2048);
    expect(usage?.cacheReadInputTokens).to.equal(1700);
    expect(usage?.cacheCreationInputTokens).to.be.undefined;
  });

  it("reads Anthropic cache_read_input_tokens and cache_creation_input_tokens", () => {
    const usage = extractUsage({
      usage: {
        input_tokens: 1024,
        output_tokens: 64,
        cache_read_input_tokens: 768,
        cache_creation_input_tokens: 256,
      },
    });
    expect(usage?.cacheReadInputTokens).to.equal(768);
    expect(usage?.cacheCreationInputTokens).to.equal(256);
  });

  it("prefers LangChain-style input_token_details over raw shapes", () => {
    const usage = extractUsage({
      usage: {
        prompt_tokens: 2048,
        completion_tokens: 32,
        total_tokens: 2080,
        prompt_tokens_details: { cached_tokens: 1700 },
        input_token_details: { cache_read: 1500, cache_creation: 100 },
      },
    });
    expect(usage?.cacheReadInputTokens).to.equal(1500);
    expect(usage?.cacheCreationInputTokens).to.equal(100);
  });

  it("reads Gemini cachedContentTokenCount as cacheReadInputTokens", () => {
    const usage = extractUsage({
      usageMetadata: {
        promptTokenCount: 4096,
        candidatesTokenCount: 80,
        totalTokenCount: 4176,
        cachedContentTokenCount: 3500,
      },
    });
    expect(usage?.inputTokens).to.equal(4096);
    expect(usage?.cacheReadInputTokens).to.equal(3500);
    expect(usage?.cacheCreationInputTokens).to.be.undefined;
  });

  it("reads Gemini fallback cachedPromptTokenCount when cachedContentTokenCount is absent", () => {
    const usage = extractUsage({
      usageMetadata: {
        promptTokenCount: 1000,
        candidatesTokenCount: 20,
        totalTokenCount: 1020,
        cachedPromptTokenCount: 800,
      },
    });
    expect(usage?.cacheReadInputTokens).to.equal(800);
  });

  it("returns undefined when no usage shape is present", () => {
    expect(extractUsage({ unrelated: true })).to.be.undefined;
  });

  it("preserves zero cache reads (does not coalesce 0 to undefined)", () => {
    const usage = extractUsage({
      usage: {
        prompt_tokens: 100,
        completion_tokens: 1,
        total_tokens: 101,
        prompt_tokens_details: { cached_tokens: 0 },
      },
    });
    expect(usage?.cacheReadInputTokens).to.equal(0);
  });

  it("preserves zero cache creations (does not coalesce 0 to undefined)", () => {
    const usage = extractUsage({
      usage: {
        input_tokens: 100,
        output_tokens: 1,
        cache_read_input_tokens: 50,
        cache_creation_input_tokens: 0,
      },
    });
    expect(usage?.cacheCreationInputTokens).to.equal(0);
  });
});

describe("getTokenUsageSummary cache aggregation", () => {
  function makeCollector(packageDir: string) {
    const fun = APIFunction.fromSignature("string-utils.titleCase(string)");
    const promptOptions = {
      ...defaultPromptOptions(),
      templateFileName: "templates/template.hb",
      retryTemplateFileName: "templates/retry-template.hb",
    };
    const outputDir = path.join(packageDir, "results");
    const collector = new TestResultCollector(
      "string-utils",
      packageDir,
      outputDir,
      [fun],
      new Map([[fun.functionName, ["titleCase('x')"]]]),
      new PerformanceMeasurer(),
      "doc",
      1,
      20,
      1
    );
    return {
      collector,
      outputDir,
      makePrompt: (snippets: string[] = []) =>
        new Prompt(fun, snippets, {
          ...promptOptions,
          includeSnippets: snippets.length > 0,
        }),
    };
  }

  it("sums cache fields across calls and surfaces diagnostics", () => {
    const packageDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "testpilot-cache-agg-")
    );
    try {
      const { collector, outputDir, makePrompt } = makeCollector(packageDir);

      const promptA = makePrompt([]);
      const promptB = makePrompt(["titleCase('x')"]);

      // Anthropic-shaped usage: cache reads + cache creations.
      collector.recordPromptInfo(promptA, 0, new Set(["a"]), {
        inputTokens: 1024,
        outputTokens: 64,
        cacheReadInputTokens: 768,
        cacheCreationInputTokens: 256,
      });
      // OpenAI-shaped usage: cache reads only (no creation field).
      collector.recordPromptInfo(promptB, 0, new Set(["b"]), {
        inputTokens: 2048,
        outputTokens: 32,
        cacheReadInputTokens: 1700,
      });

      collector.report();
      const reportJson = JSON.parse(
        fs.readFileSync(path.join(outputDir, "report.json"), "utf8")
      );
      const usage = reportJson.tokenUsage;
      expect(usage.cacheReadInputTokens).to.equal(768 + 1700);
      expect(usage.cacheCreationInputTokens).to.equal(256);
      expect(usage.cacheDiagnostics).to.deep.equal({
        tokenFieldsObserved: true,
        cacheReadInputTokensObserved: true,
        cacheCreationInputTokensObserved: true,
      });
    } finally {
      fs.rmSync(packageDir, { recursive: true, force: true });
    }
  });

  it("leaves cache fields undefined when no provider reported any", () => {
    const packageDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "testpilot-cache-noop-")
    );
    try {
      const { collector, outputDir, makePrompt } = makeCollector(packageDir);

      const promptA = makePrompt([]);
      const promptB = makePrompt(["titleCase('x')"]);

      collector.recordPromptInfo(promptA, 0, new Set(["a"]), {
        inputTokens: 100,
        outputTokens: 10,
      });
      collector.recordPromptInfo(promptB, 0, new Set(["b"]), {
        inputTokens: 200,
        outputTokens: 20,
      });

      collector.report();
      const reportJson = JSON.parse(
        fs.readFileSync(path.join(outputDir, "report.json"), "utf8")
      );
      const usage = reportJson.tokenUsage;
      expect(usage).to.not.have.property("cacheReadInputTokens");
      expect(usage).to.not.have.property("cacheCreationInputTokens");
      expect(usage.cacheDiagnostics).to.deep.equal({
        tokenFieldsObserved: false,
        cacheReadInputTokensObserved: false,
        cacheCreationInputTokensObserved: false,
      });
    } finally {
      fs.rmSync(packageDir, { recursive: true, force: true });
    }
  });

  it("flags only cache-read observation when no creation was reported", () => {
    const packageDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "testpilot-cache-readonly-")
    );
    try {
      const { collector, outputDir, makePrompt } = makeCollector(packageDir);

      const prompt = makePrompt([]);
      collector.recordPromptInfo(prompt, 0, new Set(["a"]), {
        inputTokens: 100,
        outputTokens: 10,
        cacheReadInputTokens: 0,
      });

      collector.report();
      const reportJson = JSON.parse(
        fs.readFileSync(path.join(outputDir, "report.json"), "utf8")
      );
      const usage = reportJson.tokenUsage;
      expect(usage.cacheReadInputTokens).to.equal(0);
      expect(usage).to.not.have.property("cacheCreationInputTokens");
      expect(usage.cacheDiagnostics).to.deep.equal({
        tokenFieldsObserved: true,
        cacheReadInputTokensObserved: true,
        cacheCreationInputTokensObserved: false,
      });
    } finally {
      fs.rmSync(packageDir, { recursive: true, force: true });
    }
  });
});
