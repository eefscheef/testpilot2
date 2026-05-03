#!/usr/bin/env bash
# Replay a previous TestPilot 2 benchmark run locally with the current harness.
#
# Reads cached prompts/completions from a previous run's output directory and
# feeds them back through `benchmark/run.js`. No provider calls are made
# (--strictResponses false treats cache misses as no-response).
#
# Usage:
#   scripts/replay.sh <packages_file> <orig_results_dir> <output_dir>
#
#   packages_file     path to a .github/*.txt list (URLs of packages, with SHAs)
#   orig_results_dir  dir containing <pkg>/api.json and <pkg>/prompts.json
#   output_dir        where the replayed reports will be written
#
# Example:
#   scripts/replay.sh \
#     .github/image-downloader-crawler-url-parser.txt \
#     results/image-downloader-crawler-url-parser/results \
#     results/image-downloader-crawler-url-parser/replay
#
# The model/maxTokens/snippets/etc. flags below mirror what
# .github/workflows/run-experiment.yml uses for the gpt-5.4 default. Replay
# does not contact the provider, but run.js still requires --model.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <packages_file> <orig_results_dir> <output_dir>" >&2
  exit 1
fi

PACKAGES_FILE="$(realpath "$1")"
ORIG_DIR="$(realpath "$2")"
OUT_DIR="$(realpath -m "$3")"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH_ROOT="$OUT_DIR/_packages"

mkdir -p "$OUT_DIR" "$SCRATCH_ROOT"

# Parse packages_file via the same script used by the workflow.
PACKAGES_JSON=$(node "$REPO_ROOT/.github/parse_packages.js" "+$(basename "$PACKAGES_FILE")")
# parse_packages.js looks under .github/, so the +<name>.txt arg form requires
# the file to live there. Fall back to absolute path if it doesn't.
if [ ! -f "$REPO_ROOT/.github/$(basename "$PACKAGES_FILE")" ]; then
  echo "ERROR: $PACKAGES_FILE must live in .github/ (parse_packages.js limitation)" >&2
  exit 1
fi

echo "$PACKAGES_JSON" | node -e '
const pkgs = JSON.parse(require("fs").readFileSync(0, "utf8"));
for (const p of pkgs) {
  console.log([p.host, p.owner, p.repo, p.sha, p.path || ""].join("\t"));
}
' | while IFS=$'\t' read -r HOST OWNER REPO SHA SUBPATH; do
  echo
  echo "=========================================================="
  echo "Replaying $HOST/$OWNER/$REPO@$SHA  path='$SUBPATH'"
  echo "=========================================================="

  # Clone the package source.
  CLONE_DIR="$SCRATCH_ROOT/${OWNER//\//_}__${REPO}__${SHA:0:8}"
  if [ ! -d "$CLONE_DIR/.git" ]; then
    rm -rf "$CLONE_DIR"
    git clone --quiet "https://$HOST/$OWNER/$REPO.git" "$CLONE_DIR"
  fi
  (cd "$CLONE_DIR" && git checkout --quiet "$SHA")

  # The package may live in a subpath of the repo (e.g. monorepo).
  PKG_SRC_DIR="$CLONE_DIR"
  if [ -n "$SUBPATH" ]; then PKG_SRC_DIR="$CLONE_DIR/$SUBPATH"; fi

  PKG_NAME=$(node -e "console.log(require('$PKG_SRC_DIR/package.json').name)")
  PKG_NAME_SAFE="${PKG_NAME//\//_}"
  echo "Package: $PKG_NAME"

  # The harness expects a checkout dir named after the package (some packages
  # introspect their own dir name). Copy the source into a renamed dir.
  RUN_PKG_DIR="$SCRATCH_ROOT/$PKG_NAME_SAFE"
  if [ ! -d "$RUN_PKG_DIR" ]; then
    mkdir -p "$(dirname "$RUN_PKG_DIR")"
    cp -r "$PKG_SRC_DIR" "$RUN_PKG_DIR"
  fi

  # Install the package's dependencies and mocha (matching the workflow).
  if [ ! -d "$RUN_PKG_DIR/node_modules" ]; then
    (cd "$RUN_PKG_DIR" \
      && (npm i --silent \
          || npm i --silent --legacy-peer-deps \
          || npm i --silent --ignore-scripts \
          || npm i --silent --legacy-peer-deps --ignore-scripts) \
      && (npm run build --silent || npm run prepack --silent || true) \
      && npm i --no-save --silent mocha)
  fi

  # Locate the original cached prompts/api for this package.
  ORIG_PKG_DIR="$ORIG_DIR/$PKG_NAME_SAFE"
  if [ ! -f "$ORIG_PKG_DIR/prompts.json" ] || [ ! -f "$ORIG_PKG_DIR/api.json" ]; then
    echo "WARN: missing $ORIG_PKG_DIR/{prompts,api}.json — skipping" >&2
    continue
  fi

  # Replay output goes here.
  PKG_OUT_DIR="$OUT_DIR/$PKG_NAME_SAFE"
  mkdir -p "$PKG_OUT_DIR"

  # Replay. --strictResponses false tolerates refinement-prompt cache misses
  # caused by absolute-path / Node-version differences between the original
  # CI runner and this machine.
  (cd "$REPO_ROOT" && node benchmark/run.js \
    --outputDir "$PKG_OUT_DIR" \
    --package "$RUN_PKG_DIR" \
    --api "$ORIG_PKG_DIR/api.json" \
    --responses "$ORIG_PKG_DIR/prompts.json" \
    --strictResponses false \
    --model gpt-5.4 \
    --temperatures 0.0 \
    --maxTokens 1024 \
    --snippets doc \
    --numSnippets all \
    --snippetLength 20 \
    --numCompletions 5 \
    --failOnProviderError false)

  echo "Wrote replay results to $PKG_OUT_DIR"
done

echo
echo "All packages replayed. Output: $OUT_DIR"
