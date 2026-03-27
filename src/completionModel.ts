/**
 * An abstract representation of a model such as Codex that can provide
 * completions for a prompt.
 */
export interface ITokenUsage {
  /** Tokens sent in the request prompt or message history. */
  inputTokens?: number;
  /** Tokens generated in the provider response. */
  outputTokens?: number;
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
}

export interface ICompletionModel {
  /**
   * Get a set of completions for the given prompt with the given sampling temperature.
   */
  completions(prompt: string, temperature: number): Promise<ICompletionResult>;
}
