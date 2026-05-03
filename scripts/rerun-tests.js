#!/usr/bin/env node
// Re-validate a directory of on-disk test_*.js files against an installed
// package, using the same MochaValidator the harness uses. Prints a
// machine-readable JSON summary on stdout (per-test outcomes + aggregate
// coverage), and a human-readable summary on stderr.
//
// Usage:
//   node scripts/rerun-tests.js <packagePath> <packageName> <testsDir>
//
//   packagePath  installed package source (must already have node_modules/)
//   packageName  the require('<name>') the tests use
//   testsDir     directory containing test_*.js files
//
// Used to measure the impact of the closeBrackets fix on prior runs without
// going back through the LLM-prompt pipeline. The on-disk test sources are
// what the harness *would have* validated had closeBrackets accepted them.

const fs = require("fs");
const path = require("path");
const { MochaValidator } = require(path.join(
  __dirname,
  "..",
  "dist",
  "mochaValidator"
));

if (process.argv.length !== 5) {
  console.error(
    "usage: node scripts/rerun-tests.js <packagePath> <packageName> <testsDir>"
  );
  process.exit(2);
}

let [, , packagePath, packageName, testsDir] = process.argv;
packagePath = path.resolve(packagePath);
testsDir = path.resolve(testsDir);

if (!fs.existsSync(packagePath)) {
  console.error(`ERROR: packagePath ${packagePath} does not exist`);
  process.exit(2);
}
if (!fs.existsSync(testsDir)) {
  console.error(`ERROR: testsDir ${testsDir} does not exist`);
  process.exit(2);
}

const validator = new MochaValidator(packageName, packagePath);

const testFiles = fs
  .readdirSync(testsDir)
  .filter((f) => /^test_\d+\.js$/.test(f))
  .sort((a, b) => {
    const ai = parseInt(a.match(/\d+/)[0]);
    const bi = parseInt(b.match(/\d+/)[0]);
    return ai - bi;
  });

const results = [];
const tally = { PASSED: 0, FAILED: 0, OTHER: 0, PENDING: 0 };

for (const fname of testFiles) {
  const source = fs.readFileSync(path.join(testsDir, fname), "utf8");
  let outcome;
  try {
    outcome = validator.validateTest(fname, source);
  } catch (e) {
    outcome = { status: "OTHER", err: { message: String(e) } };
  }
  const status = outcome.status || "OTHER";
  tally[status] = (tally[status] || 0) + 1;
  const errMsg = (outcome.err && outcome.err.message) || "";
  results.push({
    test: fname,
    status,
    err: errMsg ? errMsg.slice(0, 240) : undefined,
  });
  process.stderr.write(
    `${fname}: ${status}${errMsg ? "  " + errMsg.slice(0, 80) : ""}\n`
  );
}

let coverage;
try {
  coverage = validator.computeCoverageSummary();
} catch (e) {
  process.stderr.write(`coverage summary failed: ${e}\n`);
  coverage = null;
}

const summary = {
  packagePath,
  packageName,
  testsDir,
  tally,
  total: testFiles.length,
  results,
  coverage: coverage && coverage.total ? coverage.total : coverage,
};

process.stderr.write(
  `\nSUMMARY ${packageName}: ${tally.PASSED}/${testFiles.length} pass, ` +
    `coverage=${
      coverage && coverage.total && coverage.total.statements
        ? coverage.total.statements.pct + "%"
        : "n/a"
    }\n`
);

process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
