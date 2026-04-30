import fs from "fs";
import os from "os";
import path from "path";
import { expect } from "chai";
import { PerformanceMeasurer } from "../benchmark/performanceMeasurer";
import { TestResultCollector } from "../benchmark/testResultCollector";
import { APIFunction } from "../src/exploreAPI";
import { defaultPromptOptions, Prompt } from "../src/promptCrafting";
import { TestOutcome } from "../src/report";

describe("provider usage artifacts", () => {
  it("should write per-request usage to prompts.json and aggregate it in report.json", () => {
    const packageDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "testpilot-usage-pkg-")
    );
    const outputDir = path.join(packageDir, "results");

    try {
      const fun = APIFunction.fromSignature("string-utils.titleCase(string)");
      const promptOptions = {
        ...defaultPromptOptions(),
        templateFileName: "templates/template.hb",
        retryTemplateFileName: "templates/retry-template.hb",
      };
      const prompt = new Prompt(fun, [], promptOptions);
      const promptWithSnippets = new Prompt(fun, ["titleCase('x')"], {
        ...promptOptions,
        includeSnippets: true,
      });

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

      collector.recordPromptInfo(
        prompt,
        0,
        new Set(["completion one"]),
        {
          inputTokens: 100,
          outputTokens: 25,
          visibleOutputTokens: 20,
          totalTokens: 125,
          reasoningTokens: 5,
        },
        3,
        ["stop", "length", "stop"]
      );
      collector.recordPromptInfo(
        promptWithSnippets,
        0,
        new Set(["completion two"])
      );

      const testInfo = collector.recordTestInfo(
        "describe('suite', function() { it('works', function() {}); });",
        prompt,
        fun.accessPath
      );
      collector.recordTestResult(
        testInfo,
        0,
        TestOutcome.FAILED({ message: "test failed" })
      );
      collector.report();

      const promptsJson = JSON.parse(
        fs.readFileSync(path.join(outputDir, "prompts.json"), "utf8")
      );
      expect(promptsJson.prompts).to.have.length(2);
      expect(promptsJson.prompts[0].usage).to.deep.equal({
        inputTokens: 100,
        outputTokens: 25,
        visibleOutputTokens: 20,
        totalTokens: 125,
        reasoningTokens: 5,
      });
      expect(promptsJson.prompts[0].rawChoiceCount).to.equal(3);
      expect(promptsJson.prompts[0].finishReasons).to.deep.equal([
        "stop",
        "length",
        "stop",
      ]);
      expect(promptsJson.prompts[1]).to.not.have.property("usage");

      const reportJson = JSON.parse(
        fs.readFileSync(path.join(outputDir, "report.json"), "utf8")
      );
      expect(reportJson.tokenUsage).to.deep.equal({
        inputTokens: 100,
        outputTokens: 25,
        visibleOutputTokens: 20,
        totalTokens: 125,
        reasoningTokens: 5,
        promptsWithUsage: 1,
        promptsWithoutUsage: 1,
        cacheDiagnostics: {
          tokenFieldsObserved: false,
          cacheReadInputTokensObserved: false,
          cacheCreationInputTokensObserved: false,
        },
      });
    } finally {
      fs.rmSync(packageDir, { recursive: true, force: true });
    }
  });
});
