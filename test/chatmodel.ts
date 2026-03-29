import { expect } from "chai";
import { ChatModel } from "../src/chatmodel";

describe("ChatModel provider option aliases", () => {
  function applyProviderOptionAliases(model: string, options: object) {
    return (ChatModel as any).applyProviderOptionAliases(model, options);
  }

  function extractUsage(payload: object) {
    return (ChatModel as any).extractUsage(payload);
  }

  function extractFinishReason(choice: object) {
    return (ChatModel as any).extractFinishReason(choice);
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

  it("should preserve reasoning_effort when applying provider aliases", () => {
    expect(
      applyProviderOptionAliases("gemini-3-pro-preview", {
        max_tokens: 4096,
        reasoning_effort: "minimal",
      })
    ).to.deep.equal({
      max_tokens: 4096,
      reasoning_effort: "minimal",
    });
  });

  it("should derive visible output tokens when reasoning tokens are reported", () => {
    expect(
      extractUsage({
        usage: {
          prompt_tokens: 100,
          completion_tokens: 40,
          total_tokens: 140,
          completion_tokens_details: {
            reasoning_tokens: 15,
          },
        },
      })
    ).to.deep.equal({
      inputTokens: 100,
      outputTokens: 40,
      visibleOutputTokens: 25,
      totalTokens: 140,
      reasoningTokens: 15,
      cachedInputTokens: undefined,
    });
  });

  it("should mark missing finish reasons as unknown", () => {
    expect(extractFinishReason({})).to.equal("unknown");
    expect(extractFinishReason({ finish_reason: "length" })).to.equal("length");
  });
});
