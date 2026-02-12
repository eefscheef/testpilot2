import axios from "axios";
import { performance } from "perf_hooks";
import { ICompletionModel } from "./completionModel";
import {
  retry,
  RateLimiter,
  BenchmarkRateLimiter,
  FixedRateLimiter,
  IRateLimiter,
} from "./promise-utils";

const defaultPostOptions = {
  max_tokens: 1000, // maximum number of tokens to return
  temperature: 0, // sampling temperature; higher values increase diversity
  top_p: 1, // no need to change this
};
export type PostOptions = Partial<typeof defaultPostOptions>;

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

  constructor(
    private readonly model: string,
    private readonly nrAttempts: number,
    private readonly rateLimiter: IRateLimiter,
    private readonly instanceOptions: PostOptions = {}
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
  ): Promise<Set<string>> {
    const headers = {
      "Content-Type": "application/json",
      ...JSON.parse(this.authHeaders),
    };

    const options = {
      ...defaultPostOptions,
      // options provided to constructor override default options
      ...this.instanceOptions,
      // options provided to this function override default and instance options
      ...requestPostOptions,
    };

    performance.mark("llm-query-start");

    const postOptions = {
      model: this.model,
      messages: [
        {
          role: "system",
          content: "You are a programming assistant.",
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
        this.rateLimiter!.next(() =>
          axios.post(this.apiEndpoint, postOptions, { headers })
        ),
      this.nrAttempts
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
      throw new Error(json.error);
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
    return completions;
  }

  /**
   * Get completions from the LLM; issue a warning if it did not produce any
   *
   * @param prompt the prompt to use
   */
  public async completions(
    prompt: string,
    temperature: number
  ): Promise<Set<string>> {
    try {
      let result = new Set<string>();
      for (const completion of await this.query(prompt, { temperature })) {
        result.add(completion);
      }
      return result;
    } catch (err: any) {
      console.warn(`Failed to get completions: ${err.message}`);
      return new Set<string>();
    }
  }
}
