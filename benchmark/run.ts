import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import {
  APIFunction,
  exploreAPI,
  FunctionDescriptor,
  getDocSnippets,
  getSnippets,
  ICompletionModel,
  MochaValidator,
  MockCompletionModel,
  TestGenerator,
  TestValidator,
} from "..";
import { ChatModel } from "../src/chatmodel";
import yargs, { parse } from "yargs";
import { hideBin } from "yargs/helpers";
import { PerformanceMeasurer } from "./performanceMeasurer";
import { TestResultCollector } from "./testResultCollector";
import {
  BenchmarkRateLimiter,
  CompositeRateLimiter,
  FixedRateLimiter,
  IRateLimiter,
  NoRateLimiter,
  TokenRateLimiter,
} from "../src/promise-utils";
/**
 * Run an end-to-end experiment.
 * Given a package generate tests for its methods, run them, and generate a report.
 * @param model The completion model to use.
 * @param packageName The name of the package to use.
 * @param packagePath The path to the package to use.
 * @param functions The list of functions in the API.
 * @param snippetMap The snippets for package methods.
 * @param timeLimit The maximum time (in milliseconds) to run the experiment.
 */
export async function runExperiment(
  functions: APIFunction[],
  temperatures: number[],
  snippetMap: Map<string, string[]>,
  model: ICompletionModel,
  templateFileName: string,
  retryTemplateFileName: string,
  validator: TestValidator,
  collector: TestResultCollector,
  timeLimit: number
): Promise<void> {
  const deadline = performance.now() + timeLimit;
  const generator = new TestGenerator(
    temperatures,
    (fn) => snippetMap.get(fn),
    model,
    templateFileName,
    retryTemplateFileName,
    validator,
    collector
  );

  // initialize the workList with all functions
  let workList = functions.map((f) => ({ fun: f, nrTimesExtended: 0 }));

  while (workList.length > 0) {
    if (performance.now() > deadline) {
      console.log(
        `Time limit reached, ${workList.length} worklist items ignored.`
      );
      break;
    }

    const { fun } = workList.shift()!;
    await generator.generateAndValidateTests(fun);
  }

  collector.recordCoverageInfo(validator.computeCoverageSummary());
}

if (require.main === module) {
  (async () => {
    const parser = yargs(hideBin(process.argv))
      .strict()
      .options({
        outputDir: {
          type: "string",
          demandOption: true,
          description: "directory where output files will be placed",
        },
        package: {
          type: "string",
          demandOption: true,
          description: "package source",
        },
        api: {
          type: "string",
          description:
            "JSON file with API to generate tests for (usually api.json from a previous run)",
        },
        snippets: {
          type: "string",
          choices: ["code", "doc", "both", "none"],
          default: "doc",
          description: "where to collect usage snippets from",
        },
        database: {
          type: "string",
          description:
            "CodeQL database; only required if collecting snippets from code",
        },
        responses: {
          type: "string",
          description:
            "file with simulated model responses (usually prompts.json from a previous run)",
        },
        timeLimit: {
          type: "number",
          default: 5 * 60 * 60,
          description: "time limit in seconds (default is five hours)",
        },
        numSnippets: {
          default: "all",
          description:
            'number of snippets to include in the prompt, or "all" to include all snippets',
        },
        snippetLength: {
          type: "number",
          default: 20,
          description: "maximum length of each snippet in lines",
        },
        maxTokens: {
          type: "number",
          default: 1000,
          demandOption: false,
          description: "maximum number of tokens in a completion",
        },
        temperatures: {
          type: "string",
          default: "0.0",
          description:
            "whitespace-separated list of sampling temperatures to try when obtaining completions",
        },
        numCompletions: {
          type: "number",
          default: 5,
          description: "number of completions to generate for each prompt",
        },
        tokensPerMinute: {
          type: "number",
          demandOption: false,
          description:
            "maximum estimated provider tokens per minute across prompts",
        },
        strictResponses: {
          type: "boolean",
          default: true,
          description:
            "whether to require that all prompts are found when running with --responses; does not have any effect otherwise",
        },
        model: {
          type: "string",
          demandOption: true,
          default: "llama-3-70b-instruct",
          description: "LLM to use",
        },
        template: {
          type: "string",
          demandOption: true,
          default: "./templates/template.hb",
          description: "Handlebars template file to use",
        },
        retryTemplate: {
          type: "string",
          demandOption: true,
          default: "./templates/retry-template.hb",
          description: "Handlebars template file to use",
        },
        nrAttempts: {
          type: "number",
          default: 3,
          description: "number of attempts to make for each request",
        },
        failOnProviderError: {
          type: "boolean",
          default: true,
          description:
            "whether to abort the run if the provider keeps failing after retries",
        },
        rateLimit: {
          type: "string",
          default: "",
          demandOption: false,
          description: 'number of milliseconds between prompts or "benchmark"',
        },
      });
    const argv = await parser.argv;

    var model: ICompletionModel;
    if (!argv.responses) {
      if (argv.strictResponses) {
        console.warn(
          "Warning: --strictResponses has no effect when not using --responses"
        );
      }

      const rateLimiters: IRateLimiter[] = [];
      if (argv.rateLimit === "benchmark") {
        rateLimiters.push(new BenchmarkRateLimiter());
      } else if (argv.rateLimit) {
        const rateLimit: number = parseInt(argv.rateLimit, 10);
        if (!Number.isNaN(rateLimit)) {
          rateLimiters.push(new FixedRateLimiter(+argv.rateLimit));
        } else {
          throw new Error(`Invalid rate limit: ${argv.rateLimit}`);
        }
      }
      if (argv.tokensPerMinute !== undefined) {
        if (!(argv.tokensPerMinute > 0)) {
          throw new Error(
            `Invalid value for --tokensPerMinute: ${argv.tokensPerMinute}`
          );
        }
        rateLimiters.push(new TokenRateLimiter(argv.tokensPerMinute));
      }

      let rateLimiter: IRateLimiter;
      if (rateLimiters.length === 0) {
        rateLimiter = new NoRateLimiter();
      } else if (rateLimiters.length === 1) {
        rateLimiter = rateLimiters[0];
      } else {
        rateLimiter = new CompositeRateLimiter(rateLimiters);
      }

      model = new ChatModel(
        argv.model,
        argv.nrAttempts,
        rateLimiter,
        {
          max_tokens: argv.maxTokens,
          n: argv.numCompletions,
        },
        argv.failOnProviderError
      );
    } else {
      model = MockCompletionModel.fromFile(
        argv.responses,
        argv.strictResponses
      );
    }

    const packagePath = path.resolve(argv.package);
    const packageName = JSON.parse(
      fs.readFileSync(path.join(packagePath, "package.json"), "utf8")
    ).name;
    const perf = new PerformanceMeasurer();
    console.log(`Running experiment for ${packageName}`);

    let api: APIFunction[];
    if (argv.api) {
      console.log(`Loading API from ${argv.api}`);
      const rawApi: {
        accessPath: string;
        descriptor: FunctionDescriptor;
      }[] = JSON.parse(fs.readFileSync(argv.api, "utf8"));
      api = rawApi.map(
        ({ accessPath, descriptor }) =>
          new APIFunction(accessPath, descriptor, packageName)
      );
    } else {
      console.log("Exploring API");
      api = Array.from(exploreAPI(packagePath).getFunctions(packageName));
    }

    let numSnippets: number | "all" =
      argv.numSnippets === "all" ? argv.numSnippets : +argv.numSnippets;
    if (numSnippets !== "all" && !(numSnippets >= 0)) {
      throw new Error(`Invalid value for --numSnippets: ${argv.numSnippets}`);
    }

    performance.mark("snippet-extraction-start");
    let allSnippets = new Map<string, string[]>();
    if (numSnippets !== 0) {
      console.log("Extracting snippets");
      const functionNames = api.map((f) => f.functionName);
      if (argv.snippets == "code") {
        if (!argv.database) {
          throw new Error("--database is required if --snippets is code");
        }
        if (numSnippets === "all") {
          throw new Error(
            "--numSnippets=all is not supported when collecting snippets from code"
          );
        }
        allSnippets = getSnippets(
          argv.database,
          numSnippets,
          functionNames,
          argv.snippetLength
        );
      } else if (argv.snippets == "doc") {
        if (argv.database) {
          console.warn("--database is ignored if --snippets is doc");
        }
        allSnippets = getDocSnippets(
          packagePath,
          numSnippets,
          functionNames,
          argv.snippetLength
        );
      } else if (argv.snippets == "both") {
        if (!argv.database) {
          throw new Error("--database is required if --snippets is code");
        }
        if (numSnippets === "all") {
          throw new Error(
            "--numSnippets=all is not supported when collecting snippets from code"
          );
        }
        const snippets = getSnippets(
          argv.database,
          numSnippets,
          functionNames,
          argv.snippetLength
        );
        const docSnippets = getDocSnippets(
          packagePath,
          numSnippets,
          functionNames,
          argv.snippetLength
        );
        for (const [key, value] of snippets.entries()) {
          allSnippets.set(key, [...value, ...(docSnippets.get(key) || [])]);
        }
      } else {
        if (argv.database) {
          console.warn("--database is ignored if --snippets is none");
        }
      }
    }
    performance.measure("snippet-extraction", "snippet-extraction-start");

    console.log("Generating tests");
    const collector = new TestResultCollector(
      packageName,
      packagePath,
      argv.outputDir,
      api,
      allSnippets,
      perf,
      argv.snippets,
      numSnippets,
      argv.snippetLength,
      argv.numCompletions
    );
    const validator = new MochaValidator(packageName, packagePath);
    try {
      await runExperiment(
        api,
        argv.temperatures.split(/\s+/).map(parseFloat),
        allSnippets,
        model,
        argv.template,
        argv.retryTemplate,
        validator,
        collector,
        argv.timeLimit * 1000
      );
      collector.report();
      const report = collector.getReport();
      const coverage = report.coverage?.total.statements.pct ?? 0;
      console.log(`${coverage}% statement coverage`);
    } finally {
      validator.cleanup();
    }
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
