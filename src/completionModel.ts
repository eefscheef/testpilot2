/**
 * An abstract representation of a model such as Codex that can provide
 * completions for a prompt.
 */
export interface ITokenUsage {
  /** Tokens sent in the request prompt or message history. */
  inputTokens?: number;
  /** Tokens generated in the provider response, including hidden reasoning when the provider counts it there. */
  outputTokens?: number;
  /** Visible response tokens returned to the caller, excluding hidden reasoning when the provider exposes the split. */
  visibleOutputTokens?: number;
  /** Total tokens reported for the request, if available. */
  totalTokens?: number;
  /** Tokens spent on internal reasoning or thinking, if the provider exposes them. */
  reasoningTokens?: number;
  /**
   * Input tokens served from a prompt-cache hit, if the provider exposes them.
   *
   * Provider semantics differ — the value is the count the provider reports,
   * not a billing-normalized figure:
   *
   * - OpenAI / Azure OpenAI: `usage.prompt_tokens_details.cached_tokens` IS
   *   already counted inside `usage.prompt_tokens` (subset).
   * - Anthropic: `usage.cache_read_input_tokens` is reported SEPARATELY from
   *   `usage.input_tokens` (additive, not a subset).
   * - Google Gemini: `usageMetadata.cachedContentTokenCount` IS already
   *   counted inside `usageMetadata.promptTokenCount` (subset).
   *
   * Consumers that compute "billable input tokens" must branch on provider;
   * do not blindly subtract this from `inputTokens`.
   */
  cacheReadInputTokens?: number;
  /**
   * Input tokens newly written into the prompt cache on a miss, if the
   * provider exposes them. Anthropic-only as of writing —
   * `usage.cache_creation_input_tokens`. OpenAI and Gemini do not currently
   * surface a creation counter.
   */
  cacheCreationInputTokens?: number;
}

export interface ICompletionResult {
  completions: Set<string>;
  usage?: ITokenUsage;
  /** Number of raw choices returned by the provider before deduping or skipping non-text choices. */
  rawChoiceCount?: number;
  /** Finish reasons for the raw provider choices, in order. */
  finishReasons?: string[];
}

export interface ICompletionModel {
  /**
   * Get a set of completions for the given prompt with the given sampling temperature.
   */
  completions(prompt: string, temperature: number): Promise<ICompletionResult>;
}
