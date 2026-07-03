#!/usr/bin/env bash
# https://sharats.me/posts/shell-script-best-practices/

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
  set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
  echo 'Usage: ./build.sh [--verbose|-v]'
  echo
  echo 'Build the SHLLM library (Xcode/Metal aware).'
  echo
  echo 'Options:'
  echo '  --verbose, -v   Bypass xcbeautify and show raw xcodebuild output.'
  exit
fi

verbose=false
for arg in "$@"; do
  case "$arg" in
    --verbose|-v) verbose=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

DIR=$(dirname "$0")
pushd "$DIR/.." &>/dev/null

beautify=""
if ! $verbose && command -v xcbeautify &>/dev/null; then
  beautify="2>&1 | xcbeautify"
fi

eval "exec xcodebuild \
  -scheme SHLLM \
  -destination 'platform=OS X' \
  build ${beautify}"

popd &>/dev/null
