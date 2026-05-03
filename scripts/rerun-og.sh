#!/usr/bin/env bash
# Re-validate on-disk test files from previous TestPilot runs (e.g. the
# original ICSE 2024 paper artifact) against the installed package using
# the *current* harness (i.e. with the closeBrackets fix applied).
#
# This is "Path B": we don't replay through the LLM-prompt pipeline (the
# OG runs used the old completion-model template, so prompt-keyed cache
# lookups would all miss). Instead we just call MochaValidator on the
# tests the OG run already wrote to disk. The closeBrackets fix only
# affects whether tests reach the validator, so this is a faithful
# answer to "what would those runs have looked like with the fix?".
#
# Usage:
#   scripts/rerun-og.sh <packages_file> <runs_root> <output_dir>
#
#   packages_file  path to a .github/<name>.txt list (URLs+SHAs)
#   runs_root      root containing <model>/<runid>/<pkg>/tests/...
#                  (e.g. ~/final-project/artifact-tse-minorrev/artifact/data)
#   output_dir     where to write per-(model,run,pkg) summary JSON files
#
# Iterates over MODEL=cushman,gptturbo,starcoder; for each model, processes
# every run directory found under <runs_root>/<model>/. Re-validates both
# crawler-url-parser and image-downloader against freshly-cloned packages.
#
# Edit MODELS below to extend, or set MAX_RUNS_PER_MODEL=N to cap the count
# per model (useful for smoke runs).

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <packages_file> <runs_root> <output_dir>" >&2
  exit 1
fi

PACKAGES_FILE="$1"
RUNS_ROOT="$(realpath "$2")"
OUT_DIR="$(realpath -m "$3")"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH_ROOT="$OUT_DIR/_packages"

mkdir -p "$OUT_DIR" "$SCRATCH_ROOT"

# Build packages once (re-uses scratch checkouts across all models).
# We invoke the helper from replay.sh inline by parsing the same .txt list.
if [ ! -f "$REPO_ROOT/.github/$(basename "$PACKAGES_FILE")" ]; then
  echo "ERROR: $PACKAGES_FILE must live in .github/" >&2
  exit 1
fi

PACKAGES_JSON=$(node "$REPO_ROOT/.github/parse_packages.js" \
  "+$(basename "$PACKAGES_FILE")")

declare -A PKG_DIR # package-name -> install dir

while IFS=$'\t' read -r HOST OWNER REPO SHA SUBPATH; do
  CLONE_DIR="$SCRATCH_ROOT/${OWNER//\//_}__${REPO}__${SHA:0:8}"
  if [ ! -d "$CLONE_DIR/.git" ]; then
    rm -rf "$CLONE_DIR"
    git clone --quiet "https://$HOST/$OWNER/$REPO.git" "$CLONE_DIR"
  fi
  (cd "$CLONE_DIR" && git checkout --quiet "$SHA")

  PKG_SRC_DIR="$CLONE_DIR"
  if [ -n "$SUBPATH" ]; then PKG_SRC_DIR="$CLONE_DIR/$SUBPATH"; fi

  PKG_NAME=$(node -e "console.log(require('$PKG_SRC_DIR/package.json').name)")
  PKG_NAME_SAFE="${PKG_NAME//\//_}"

  RUN_PKG_DIR="$SCRATCH_ROOT/$PKG_NAME_SAFE"
  if [ ! -d "$RUN_PKG_DIR" ]; then
    cp -r "$PKG_SRC_DIR" "$RUN_PKG_DIR"
  fi
  if [ ! -d "$RUN_PKG_DIR/node_modules" ]; then
    (cd "$RUN_PKG_DIR" \
      && (npm i --silent \
          || npm i --silent --legacy-peer-deps \
          || npm i --silent --ignore-scripts \
          || npm i --silent --legacy-peer-deps --ignore-scripts) \
      && (npm run build --silent || npm run prepack --silent || true) \
      && npm i --no-save --silent mocha)
  fi

  PKG_DIR["$PKG_NAME_SAFE"]="$RUN_PKG_DIR"
  echo "Prepared $PKG_NAME at $RUN_PKG_DIR"
done < <(echo "$PACKAGES_JSON" | node -e '
const pkgs = JSON.parse(require("fs").readFileSync(0, "utf8"));
for (const p of pkgs) console.log([p.host, p.owner, p.repo, p.sha, p.path||""].join("\t"));
')

# Now iterate models × packages.
MODELS=("cushman" "gptturbo" "starcoder")

for MODEL in "${MODELS[@]}"; do
  MODEL_ROOT="$RUNS_ROOT/$MODEL"
  if [ ! -d "$MODEL_ROOT" ]; then
    echo "WARN: $MODEL_ROOT not found, skipping" >&2
    continue
  fi
  RUN_IDS=$(ls "$MODEL_ROOT" | sort)
  if [ -n "${MAX_RUNS_PER_MODEL:-}" ]; then
    RUN_IDS=$(echo "$RUN_IDS" | head -n "$MAX_RUNS_PER_MODEL")
  fi

  for RUN_ID in $RUN_IDS; do
    RUN_DIR="$MODEL_ROOT/$RUN_ID"

    for PKG_NAME_SAFE in "${!PKG_DIR[@]}"; do
      PKG_TESTS_DIR="$RUN_DIR/$PKG_NAME_SAFE/tests"
      if [ ! -d "$PKG_TESTS_DIR" ]; then
        echo "WARN: $PKG_TESTS_DIR not found, skipping" >&2
        continue
      fi
      PKG_PATH="${PKG_DIR[$PKG_NAME_SAFE]}"
      OUT_FILE="$OUT_DIR/${MODEL}__${RUN_ID}__${PKG_NAME_SAFE}.json"

      if [ -f "$OUT_FILE" ]; then
        echo "skip (already have) $OUT_FILE"
        continue
      fi

      echo
      echo "=== $MODEL/$RUN_ID/$PKG_NAME_SAFE ==="
      node "$REPO_ROOT/scripts/rerun-tests.js" \
        "$PKG_PATH" "$PKG_NAME_SAFE" "$PKG_TESTS_DIR" \
        > "$OUT_FILE" 2>/dev/null
      echo "wrote $OUT_FILE"
    done
  done
done

echo
echo "All summaries in $OUT_DIR"
