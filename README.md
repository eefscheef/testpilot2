# TestPilot 2

TestPilot 2 is a tool for automatically generating unit tests for npm packages
written in JavaScript/TypeScript using a large language model (LLM). It is an
adaptation of [TestPilot](https://github.com/githubnext/testpilot/) that works with chat models instead of completion models.
A research paper describing TestPilot in detail is available from [IEEExplore](https://ieeexplore.ieee.org/document/10329992).

## Background

TestPilot 2 generates tests for a given function `f` by prompting the LLM with a
skeleton of a test for `f`, including information about `f` embedded in code
comments, such as its signature, the body of `f`, and examples usages of `f`
automatically mined from project documentation. The model's response is then
parsed and translated into a runnable unit test. Optionally, the test is run and
if it fails the model is prompted again with additional information about the
failed test, giving it a chance to refine the test.

Unlike other systems for LLM-based test generation, TestPilot 2 does not require
any additional training or reinforcement learning, and no examples of functions
and their associated tests are needed.

## Requirements

In general, to be able to run TestPilot you need access to an LLM
with a chat API. Set the `TESTPILOT_LLM_API_ENDPOINT` environment variable to
the URL of the LLM API endpoint you want to use, and
`TESTPILOT_LLM_AUTH_HEADERS` to a JSON object containing the headers you need to
authenticate with the API.

Typical values for these variables might be:

- `TESTPILOT_LLM_API_ENDPOINT='https://api.perplexity.ai/chat/completions'`
- `TESTPILOT_LLM_AUTH_HEADERS='{"Authorization": "Bearer <your API key>" }'`

For Azure OpenAI, TestPilot expects the full chat-completions URL, not just the
API base URL. If your Azure resource base URL is
`https://<resource>.openai.azure.com/openai/v1`, set:

- `TESTPILOT_LLM_API_ENDPOINT='https://<resource>.openai.azure.com/openai/v1/chat/completions'`
- `TESTPILOT_LLM_AUTH_HEADERS='{"api-key": "<your Azure API key>" }'`

Then use the Azure deployment name as the `--model` value or workflow `model`
input. For example, if your deployment is named `gpt-5.4`, run with
`--model gpt-5.4`.

For Gemini's OpenAI-compatible endpoint, a typical configuration is:

- `TESTPILOT_LLM_API_ENDPOINT='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'`
- `TESTPILOT_LLM_AUTH_HEADERS='{"Authorization": "Bearer <your Gemini API key>" }'`

As of November 18, 2025, Google's published model code for Gemini 3 Pro Preview
is `gemini-3-pro-preview`. Some Google documentation refers to "Gemini 3.1
Pro", but the model identifier to pass to the API is `gemini-3-pro-preview`.

Note, however, that you can run TestPilot 2 in reproduction mode without access to
the LLM API where model responses are taken from the output of a previous run;
see below for details.

## Installation

The `src/` directory contains the source code for TestPilot 2, which is written in
TypeScript and gets compiled into the `dist/` directory. Tests are in `test/`;
the `benchmark/` directory contains a benchmarking harness for running TestPilot
on multiple npm packages.

In the root directory of a checkout of this repository, run `npm build` to
install dependencies and build the package.

You can also use `npm run build:watch` to automatically build anytime you make
changes to the code. Note, however, that this will not automatically install
dependencies, and also will not build the benchmarking harness.

Use `npm run test` to run the tests. For convenience, this will also install
dependencies and run a build.

## Benchmarking

If you install TestPilot 2 from source, you can use the benchmarking harness to
run TestPilot 2 on multiple packages and analyze the results.

### Running locally

Basic usage is as follows:

```sh
node benchmark/run.js --outputDir <report_dir> --package <package_dir>
```

This generates tests for all functions exported by the package in
`<package_dir>`, validates them, and writes the results to `<report_dir>`.

Note that this assumes that package dependencies are installed and any build
steps have been run (e.g., using `npm i` and `npm run build`). TestPilot also
relies on `mocha`, so if the package under test does not already depend on it,
you must install it separately, for example using the command `npm i --no-save mocha`.

### Running on Actions

The `run-experiment.yml` workflow runs an experiment on GitHub Actions,
producing the final report as an artifact you can download. The `results-all`
artifact contains the results of all packages, while the other artifacts contain
the individual results of each package.

The workflow accepts optional provider-specific inputs:

- `llmApiEndpoint`: override the endpoint for a single run without changing the
  repository secret.
- `llmAuthHeadersSecretName`: choose which secret contains the JSON auth headers
  for the selected provider.
- `reasoningEffort`: pass provider reasoning effort through to the chat API.
  For Gemini's OpenAI-compatible endpoint, useful values are `minimal`, `low`,
  `medium`, and `high`.
- `maxRequestsPerMinute`: fixed request pacing.
- `maxTokensPerMinute`: estimated token pacing based on prompt size and
  completion budget. This is useful for token-based quotas such as Azure TPM
  limits.
- `failOnProviderError`: fail the package benchmark if the provider still
  errors after retries.

For an Azure deployment with a 98k TPM limit, start with
`maxTokensPerMinute=98000`. Because TestPilot runs prompts sequentially, you do
not need a custom request-per-minute limiter just to integrate Azure, but a
token limiter is the better fit for token-based quotas.

### Reproducing results

The results of TestPilot 2 are non-deterministic, so even if you run it from the
same package on the same machine multiple times, you will get different results.
However, the benchmarking harness records enough data to be able to replay a
benchmark run in many cases.

To do this, use the `--api` and `--responses` options to reuse the API listings
and responses from a previous run:

```sh
node benchmark/run.js --outputDir <report_dir> --package <package_dir> --api <api.json> --responses <prompts.json>
```

Note that by default replay will fail if any of the prompts are not found in the
responses file. This typically happens if TestPilot 2 is refining failing tests,
since in this case the prompt to the model depends on the exact failure message,
which can be system-specific (e.g., containing local file-system paths), or
depend on the Node.js version or other factors.

To work around these limitations, you can pass the `--strictResponses false`
flag handle treat missing prompts by treating them as getting no response from
the model. This will not, in general, produce the same results as the initial
run, but suffices in many cases.

## License

This project is licensed under the terms of the MIT open source license. Please refer to [MIT](./LICENSE.txt) for the full terms.

## Maintainers

- Frank Tip (@franktip)
- Max Schaefer (@max-schaefer)
- Sarah Nadi (@snadi)

## Support

TestPilot 2 is a research prototype and is not officially supported. However, if
you have questions or feedback, please file an issue and we will do our best to
respond.

## Acknowledgement

We thank Aryaz Eghbali (@aryaze) for his work on the initial version of
TestPilot.
