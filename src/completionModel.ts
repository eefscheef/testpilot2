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
  /** Cached input tokens, if the provider exposes them separately. */
  cachedInputTokens?: number;
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
