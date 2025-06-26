#!/usr/bin/env bash
# https://sharats.me/posts/shell-script-best-practices/

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
  set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
  echo 'Usage: ./test.sh [MODULE/FILE]'
  exit
fi

DIR=$(dirname "$0")
pushd "$DIR/.." &>/dev/null

testSpecifiers=()
for arg in "$@"; do
  testSpecifiers+=("-only-testing:$arg")
done

tests=$(echo "${testSpecifiers[@]:-}" | sed 's/ *$//')

beautify=""
if command -v xcbeautify &>/dev/null; then
  beautify="| xcbeautify"
fi

# eval "exec xcodebuild \
#   -scheme SHLLM \
#   -destination 'platform=OS X' \
#   ${tests} \
#   test ${beautify}"

eval "exec xcodebuild \
  -scheme SHLLM \
  -destination 'platform=OS X' \
  ${tests} \
  test"

popd &>/dev/null
