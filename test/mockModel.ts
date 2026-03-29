import fs from "fs";
import os from "os";
import path from "path";
import { expect } from "chai";
import { MockCompletionModel } from "../src/mockModel";

describe("test MockCompletionModel", () => {
  it("should be able to add and get completions", async () => {
    const model = new MockCompletionModel(true);
    model.addCompletions("foo", 0.5, ["bar", "baz"]);
    expect((await model.completions("foo", 0.5)).completions).to.deep.equal(
      new Set(["bar", "baz"])
    );
  });

  it("should throw an error if completions are not found", async () => {
    const model = new MockCompletionModel(true);
    try {
      await model.completions("foo", 0.5);
      expect.fail();
    } catch (e: any) {
      expect(e.message).to.equal("Prompt not found at temperature 0.5: foo");
    }
  });

  it("should not throw an error if completions are not found and strictResponses is false", async () => {
    const model = new MockCompletionModel(false);
    expect((await model.completions("foo", 0.5)).completions).to.deep.equal(
      new Set()
    );
  });

  it("should preserve usage when loading responses from prompts.json", async () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "testpilot-mock-"));
    const promptsDir = path.join(tempDir, "prompts");
    fs.mkdirSync(promptsDir);

    try {
      const prompt = "write a test";
      fs.writeFileSync(path.join(promptsDir, "prompt_0.js"), prompt);
      fs.writeFileSync(
        path.join(tempDir, "prompts.json"),
        JSON.stringify(
          {
            prompts: [
              {
                file: "prompt_0.js",
                temperature: 0.5,
                completions: ["bar"],
                usage: {
                  inputTokens: 11,
                  outputTokens: 7,
                  visibleOutputTokens: 4,
                  totalTokens: 18,
                  reasoningTokens: 3,
                },
                rawChoiceCount: 2,
                finishReasons: ["stop", "length"],
              },
            ],
          },
          null,
          2
        )
      );

      const model = MockCompletionModel.fromFile(
        path.join(tempDir, "prompts.json"),
        true
      );
      expect(await model.completions(prompt, 0.5)).to.deep.equal({
        completions: new Set(["bar"]),
        usage: {
          inputTokens: 11,
          outputTokens: 7,
          visibleOutputTokens: 4,
          totalTokens: 18,
          reasoningTokens: 3,
        },
        rawChoiceCount: 2,
        finishReasons: ["stop", "length"],
      });
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
