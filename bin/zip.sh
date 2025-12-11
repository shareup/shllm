#!/usr/bin/env bash
# https://sharats.me/posts/shell-script-best-practices/

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
  set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
  echo 'Usage: ./zip.sh MODEL_NAME [MODEL_NAME ...]'
  echo
  echo 'Zips model directories in Sources/SHLLM/Resources/'
  echo
  echo 'Example: ./zip.sh Qwen3-VL-2B-Thinking-4bit Qwen3-VL-4B-Instruct-4bit'
  exit
fi

if [[ $# -eq 0 ]]; then
  echo 'Error: At least one model name is required'
  echo 'Usage: ./zip.sh MODEL_NAME [MODEL_NAME ...]'
  exit 1
fi

DIR=$(dirname "$0")
pushd "$DIR/../Sources/SHLLM/Resources" &>/dev/null

for model in "$@"; do
  if [[ ! -d "${model}" ]]; then
    echo "Error: Directory '${model}' not found"
    exit 1
  fi

  echo "Zipping ${model}..."
  zip -r "${model}.zip" "${model}" -x "*.DS_Store" -x "__MACOSX/*" -x "*/._*"
done

popd &>/dev/null
