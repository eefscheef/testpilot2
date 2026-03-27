import path from "path";
import {
  ICompletionModel,
  ICompletionResult,
  ITokenUsage,
} from "./completionModel";
import { readFileSync } from "fs";

export class MockCompletionModel implements ICompletionModel {
  private completionMap: Map<string, ICompletionResult> = new Map();

  constructor(private strictResponses: boolean) {}

  static fromFile(file: string, strictResponses: boolean) {
    const data = JSON.parse(readFileSync(file, "utf8"));
    console.log("Loading completions from file");
    const model = new MockCompletionModel(strictResponses);
    for (const {
      file: promptFile,
      temperature,
      completions,
      usage,
    } of data.prompts) {
      const prompt = readFileSync(
        path.join(path.dirname(file), "prompts", promptFile),
        "utf8"
      );
      model.addCompletions(prompt, temperature, completions, usage);
    }
    return model;
  }

  private key(prompt: string, temperature: number) {
    return JSON.stringify([prompt, temperature]);
  }

  public addCompletions(
    prompt: string,
    temperature: number,
    completions: string[],
    usage?: ITokenUsage
  ) {
    this.completionMap.set(this.key(prompt, temperature), {
      completions: new Set(completions),
      usage,
    });
  }

  public async completions(
    prompt: string,
    temperature: number
  ): Promise<ICompletionResult> {
    const completionResult = this.completionMap.get(
      this.key(prompt, temperature)
    );
    if (!completionResult) {
      const err = `Prompt not found at temperature ${temperature}: ${prompt}`;
      if (this.strictResponses) {
        throw new Error(err);
      } else {
        console.warn(err);
      }
      return { completions: new Set() };
    }
    return {
      completions: new Set(completionResult.completions),
      usage: completionResult.usage,
    };
  }
}
