import axios from "axios";
import { performance } from "perf_hooks";
import {
  ICompletionModel,
  ICompletionResult,
  ITokenUsage,
} from "./completionModel";
import { retry, IRateLimiter } from "./promise-utils";

const defaultPostOptions = {
  max_tokens: 1000, // maximum number of tokens to return
  temperature: 0, // sampling temperature; higher values increase diversity
  top_p: 1, // no need to change this
};
export interface PostOptions {
  max_tokens?: number;
  max_completion_tokens?: number;
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  [key: string]: any;
}

type ChatCompletionsRequest = {
  model: string;
  stream?: boolean;
  messages: { role: string; content: string }[];
  max_tokens?: number;
  max_completion_tokens?: number;
  // Gemini OpenAI-compat endpoint tends to honor this name.
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  [key: string]: any;
};

function getEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    console.error(`Please set the ${name} environment variable.`);
    process.exit(1);
  }
  return value;
}

/**
 * A model that uses the ChatModel API to provide completions.
 */
export class ChatModel implements ICompletionModel {
  private readonly apiEndpoint: string;
  private readonly authHeaders: string;
  private static readonly SYSTEM_PROMPT = "You are a programming assistant.";
  private static readonly RETRYABLE_STATUS_CODES = new Set([
    408, 409, 429, 500, 502, 503, 504, 529,
  ]);
  private static readonly RETRYABLE_ERROR_CODES = new Set([
    "ECONNABORTED",
    "ECONNRESET",
    "EAI_AGAIN",
    "ENETDOWN",
    "ENETUNREACH",
    "ETIMEDOUT",
  ]);

  private static extractChoiceText(choice: any): string | undefined {
    const content =
      choice?.message?.content ??
      choice?.delta?.content ??
      choice?.text ??
      choice?.content;

    if (typeof content === "string") {
      return content;
    }

    // Some APIs represent content as an array of parts, e.g. [{type:'text', text:'...'}]
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const item of content) {
        if (typeof item === "string") {
          parts.push(item);
        } else if (typeof item?.text === "string") {
          parts.push(item.text);
        } else if (typeof item?.content === "string") {
          parts.push(item.content);
        }
      }
      const joined = parts.join("");
      return joined.length > 0 ? joined : undefined;
    }

    if (content && typeof content === "object") {
      if (typeof (content as any).text === "string") {
        return (content as any).text;
      }
    }

    return undefined;
  }

  private static extractUsage(json: any): ITokenUsage | undefined {
    const usage = json?.usage;
    const usageMetadata = json?.usageMetadata;
    if (!usage && !usageMetadata) {
      return undefined;
    }

    const normalized: ITokenUsage = {
      inputTokens:
        usage?.prompt_tokens ??
        usageMetadata?.promptTokenCount ??
        usageMetadata?.prompt_token_count,
      outputTokens:
        usage?.completion_tokens ??
        usageMetadata?.candidatesTokenCount ??
        usageMetadata?.completionTokenCount ??
        usageMetadata?.completion_token_count,
      totalTokens:
        usage?.total_tokens ??
        usageMetadata?.totalTokenCount ??
        usageMetadata?.total_token_count,
      reasoningTokens:
        usage?.completion_tokens_details?.reasoning_tokens ??
        usage?.output_tokens_details?.reasoning_tokens ??
        usageMetadata?.thoughtsTokenCount ??
        usageMetadata?.reasoningTokenCount,
      cachedInputTokens:
        usage?.prompt_tokens_details?.cached_tokens ??
        usageMetadata?.cachedContentTokenCount ??
        usageMetadata?.cachedPromptTokenCount,
    };

    if (Object.values(normalized).every((value) => value === undefined)) {
      return undefined;
    }
    return normalized;
  }

  private static shouldUseMaxCompletionTokens(model: string) {
    return /^(gpt-5|o1|o3|o4)/i.test(model);
  }

  private static applyProviderOptionAliases(
    model: string,
    options: PostOptions
  ): PostOptions {
    const aliasedOptions = { ...options };
    const maxTokenBudget = aliasedOptions.max_tokens;
    if (
      maxTokenBudget !== undefined &&
      aliasedOptions.max_completion_tokens === undefined &&
      ChatModel.shouldUseMaxCompletionTokens(model)
    ) {
      aliasedOptions.max_completion_tokens = maxTokenBudget;
      delete aliasedOptions.max_tokens;
    }
    if (
      maxTokenBudget !== undefined &&
      aliasedOptions.max_output_tokens === undefined &&
      /gemini/i.test(model)
    ) {
      aliasedOptions.max_output_tokens = maxTokenBudget;
    }
    return aliasedOptions;
  }

  private static estimateTokenBudget(prompt: string, options: PostOptions) {
    const outputBudgetPerChoice =
      options.max_completion_tokens ??
      options.max_output_tokens ??
      options.max_tokens ??
      defaultPostOptions.max_tokens;
    const outputBudget = outputBudgetPerChoice * Math.max(1, options.n ?? 1);
    // Use a conservative character heuristic; this is only used for pacing.
    const inputTokens = Math.ceil(
      (prompt.length + ChatModel.SYSTEM_PROMPT.length + 32) / 3
    );
    return inputTokens + outputBudget;
  }

  private static getRetryAfterDelayMs(error: unknown): number | undefined {
    if (!axios.isAxiosError(error)) {
      return undefined;
    }

    const retryAfterHeader =
      error.response?.headers?.["retry-after"] ??
      error.response?.headers?.["Retry-After"];
    if (!retryAfterHeader) {
      return undefined;
    }

    const retryAfter = Array.isArray(retryAfterHeader)
      ? retryAfterHeader[0]
      : retryAfterHeader;
    if (typeof retryAfter !== "string" && typeof retryAfter !== "number") {
      return undefined;
    }

    const retrySeconds = Number(retryAfter);
    if (Number.isFinite(retrySeconds)) {
      return Math.max(0, retrySeconds * 1000);
    }

    const retryDate = Date.parse(String(retryAfter));
    if (Number.isNaN(retryDate)) {
      return undefined;
    }
    return Math.max(0, retryDate - Date.now());
  }

  private static shouldRetryRequest(error: unknown): boolean {
    if (!axios.isAxiosError(error)) {
      return true;
    }
    if (!error.response) {
      return (
        !error.code || ChatModel.RETRYABLE_ERROR_CODES.has(String(error.code))
      );
    }
    return ChatModel.RETRYABLE_STATUS_CODES.has(error.response.status);
  }

  private static getRetryDelayMs(error: unknown, attempt: number): number {
    const retryAfterMs = ChatModel.getRetryAfterDelayMs(error);
    if (retryAfterMs !== undefined) {
      return retryAfterMs;
    }

    const baseDelayMs = Math.min(30000, 1000 * Math.pow(2, attempt - 1));
    const jitterMs = Math.floor(Math.random() * 250);
    return baseDelayMs + jitterMs;
  }

  private static formatError(error: unknown): string {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      const statusText = error.response?.statusText;
      const data = error.response?.data;
      const dataSnippet =
        typeof data === "string"
          ? data.slice(0, 2000)
          : JSON.stringify(data)?.slice(0, 2000);
      return `HTTP ${status ?? "?"} ${statusText ?? ""}${
        dataSnippet ? `; body: ${dataSnippet}` : ""
      }`;
    }
    return (error as any)?.message ?? String(error);
  }

  constructor(
    private readonly model: string,
    private readonly nrAttempts: number,
    private readonly rateLimiter: IRateLimiter,
    private readonly instanceOptions: PostOptions = {},
    private readonly failOnError = false
  ) {
    this.apiEndpoint = getEnv("TESTPILOT_LLM_API_ENDPOINT");
    this.authHeaders = getEnv("TESTPILOT_LLM_AUTH_HEADERS");

    console.log(
      `Using ${this.model} at ${this.apiEndpoint} with ${
        this.nrAttempts
      } attempts and ${this.rateLimiter.getDescription()}`
    );
  }

  /**
   * Query the ChatModel for completions with a given prompt.
   *
   * @param prompt The prompt to use for the completion.
   * @param requestPostOptions The options to use for the request.
   * @returns A promise that resolves to a set of completions.
   */
  public async query(
    prompt: string,
    requestPostOptions: PostOptions = {}
  ): Promise<ICompletionResult> {
    const headers = {
      "Content-Type": "application/json",
      ...JSON.parse(this.authHeaders),
    };

    const options = ChatModel.applyProviderOptionAliases(this.model, {
      ...defaultPostOptions,
      // options provided to constructor override default options
      ...this.instanceOptions,
      // options provided to this function override default and instance options
      ...requestPostOptions,
    });

    performance.mark("llm-query-start");

    const postOptions: ChatCompletionsRequest = {
      model: this.model,
      // Some OpenAI-compatible endpoints (including Gemini proxies) may default
      // to streaming; we only support non-streaming responses.
      stream: false,
      messages: [
        {
          role: "system",
          content: ChatModel.SYSTEM_PROMPT,
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      ...options,
    };

    const res = await retry(
      () =>
        this.rateLimiter.next(
          () => axios.post(this.apiEndpoint, postOptions, { headers }),
          {
            estimatedTokens: ChatModel.estimateTokenBudget(prompt, options),
          }
        ),
      this.nrAttempts,
      {
        shouldRetry: ChatModel.shouldRetryRequest,
        getDelayMs: ChatModel.getRetryDelayMs,
      }
    );

    performance.measure(
      `llm-query:${JSON.stringify({
        ...options,
        promptLength: prompt.length,
      })}`,
      "llm-query-start"
    );
    if (res.status !== 200) {
      throw new Error(
        `Request failed with status ${res.status} and message ${res.statusText}`
      );
    }
    if (!res.data) {
      throw new Error("Response data is empty");
    }

    const json = res.data;
    if (json.error) {
      throw new Error(
        typeof json.error === "string" ? json.error : JSON.stringify(json.error)
      );
    }

    if (!Array.isArray(json.choices)) {
      throw new Error(
        `Unexpected LLM response format: expected choices array, got ${typeof json.choices}`
      );
    }

    const completions = new Set<string>();
    let skipped = 0;
    for (const choice of json.choices) {
      const text = ChatModel.extractChoiceText(choice);
      if (typeof text !== "string") {
        skipped++;
        continue;
      }
      completions.add(text);
    }

    if (skipped > 0) {
      const first = json.choices.find(
        (c: any) => typeof ChatModel.extractChoiceText(c) !== "string"
      );
      const snippet = JSON.stringify(first)?.slice(0, 500);
      console.warn(
        `Warning: skipped ${skipped} LLM choice(s) with no text content. Example: ${snippet}`
      );
    }
    return {
      completions,
      usage: ChatModel.extractUsage(json),
    };
  }

  /**
   * Get completions from the LLM; issue a warning if it did not produce any
   *
   * @param prompt the prompt to use
   */
  public async completions(
    prompt: string,
    temperature: number
  ): Promise<ICompletionResult> {
    try {
      const queryResult = await this.query(prompt, { temperature });
      let result = new Set<string>();
      for (const completion of queryResult.completions) {
        result.add(completion);
      }
      return {
        completions: result,
        usage: queryResult.usage,
      };
    } catch (err: any) {
      const formattedError = ChatModel.formatError(err);
      if (this.failOnError) {
        throw new Error(`Failed to get completions: ${formattedError}`);
      }
      console.warn(`Failed to get completions: ${formattedError}`);
      return { completions: new Set<string>() };
    }
  }
}
