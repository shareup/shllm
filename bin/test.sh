#!/usr/bin/env bash

set -eo pipefail
cd "$(dirname $0)/.."

verbose=false
testSpecifiers=()
for arg in "$@"; do
  case "$arg" in
    --verbose|-v) verbose=true ;;
    -h|--help)
      echo 'Usage: ./test.sh [--verbose|-v] [TEST_SPECIFIER ...]'
      echo
      echo 'Run SHLLM tests (Xcode/Metal aware).'
      echo
      echo 'Options:'
      echo '  --verbose, -v   Bypass xcbeautify and show raw xcodebuild output.'
      echo
      echo 'Test specifiers are passed to xcodebuild as -only-testing: arguments.'
      echo 'Examples:'
      echo '  ./test.sh'
      echo '  ./test.sh SHLLMTests/ResponseParserTests'
      echo '  ./test.sh -v SHLLMTests/Gemma4_E2BTests/canStreamResult()'
      exit
      ;;
    *) testSpecifiers+=("-only-testing:$arg") ;;
  esac
done

xcodeVersion=$(xcodebuild -version | sed -n 's/Xcode \([0-9]*\).*/\1/p')
if [ "$xcodeVersion" -ge 26 ]; then
  if ! xcodebuild -showComponent metalToolchain >/dev/null 2>&1; then
    echo "❌ Metal toolchain is not installed"

    echo "⬇️ Downloading Metal toolchain..."
    xcodebuild \
      -downloadComponent metalToolchain \
      -exportPath /tmp/metalToolchainDownload/

    echo "🧰 Installing Metal toolchain..."
    xcodebuild \
      -importComponent metalToolchain \
      -importPath /tmp/metalToolchainDownload/*.exportedBundle
  fi
fi

signingFlags="CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO"

if ! $verbose && command -v xcbeautify &>/dev/null; then
  xcodebuild \
    -scheme SHLLM \
    -destination 'platform=OS X' \
    ${signingFlags} \
    "${testSpecifiers[@]}" \
    test 2>&1 | xcbeautify
else
  xcodebuild \
    -scheme SHLLM \
    -destination 'platform=OS X' \
    ${signingFlags} \
    "${testSpecifiers[@]}" \
    test
fi
