import { expect } from "chai";
import { ChatModel } from "../src/chatmodel";

describe("ChatModel provider option aliases", () => {
  function applyProviderOptionAliases(model: string, options: object) {
    return (ChatModel as any).applyProviderOptionAliases(model, options);
  }

  it("should keep max_tokens for Gemini OpenAI-compatible models", () => {
    expect(
      applyProviderOptionAliases("gemini-3-pro-preview", {
        max_tokens: 4096,
        temperature: 0,
      })
    ).to.deep.equal({
      max_tokens: 4096,
      temperature: 0,
    });
  });

  it("should translate max_tokens to max_completion_tokens for GPT-5 models", () => {
    expect(
      applyProviderOptionAliases("gpt-5.4", {
        max_tokens: 4096,
        temperature: 0,
      })
    ).to.deep.equal({
      max_completion_tokens: 4096,
      temperature: 0,
    });
  });
});
