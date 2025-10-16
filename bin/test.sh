#!/usr/bin/env bash

set -eo pipefail
cd "$(dirname $0)/.."

testSpecifiers=()
for arg in "$@"; do
  testSpecifiers+=("-only-testing:$arg")
done

xcodeVersion=$(xcodebuild -version | sed -n 's/Xcode \([0-9]*\).*/\1/p')
if [ "$xcodeVersion" -ge 26 ]; then
  if ! xcodebuild -showComponent metalToolchain >/dev/null 2>&1; then
    echo "❌ Metal toolchain is not installed"

    echo "⬇️ Downloading Metal toolchain..."
    eval "exec xcodebuild \
      -downloadComponent metalToolchain
      -exportPath /tmp/metalToolchainDownload/ ${beautify}"

    echo "🧰 Installing Metal toolchain..."
    eval "exec xcodebuild
      -importComponent metalToolchain
      -importPath /tmp/metalToolchainDownload/*.exportedBundle ${beautify}"
  fi
fi

signingFlags="CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO"

if command -v xcbeautify &>/dev/null; then
  xcodebuild \
    -scheme SHLLM \
    -destination 'platform=OS X' \
    ${signingFlags} \
    "${testSpecifiers[@]}" \
    test | xcbeautify
else
  xcodebuild \
    -scheme SHLLM \
    -destination 'platform=OS X' \
    ${signingFlags} \
    "${testSpecifiers[@]}" \
    test
fi
