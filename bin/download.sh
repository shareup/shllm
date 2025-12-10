#!/usr/bin/env bash

set -eo pipefail

ids=(
  "CodeLlama-13b-Instruct-hf-4bit-MLX"
  "DeepSeek-R1-Distill-Qwen-7B-4bit"
  "gemma-2-2b-it-4bit"
  "gemma-2-9b-it-4bit"
  "gemma-3-12b-it-qat-3bit"
  "gemma-3-12b-it-qat-4bit"
  "gemma-3-1b-it-qat-4bit"
  "gemma-3-27b-it-qat-3bit"
  "gemma-3-27b-it-qat-4bit"
  "gemma-3-4b-it-qat-3bit"
  "gemma-3-4b-it-qat-4bit"
  "LFM2-8B-A1B-4bit"
  "Llama-3.2-1B-Instruct-4bit"
  "Llama-3.2-3B-Instruct-4bit"
  "gpt-oss-20b-MXFP4-Q4"
  "lmstudio-community/gpt-oss-20b-MLX-8bit"
  "Meta-Llama-3-8B-Instruct-4bit"
  "Meta-Llama-3.1-8B-Instruct-4bit"
  "Mistral-7B-Instruct-v0.3-4bit"
  "Mistral-Nemo-Instruct-2407-4bit"
  "OpenELM-270M-Instruct"
  "Orchestrator-8B-4bit"
  "phi-2-hf-4bit-mlx"
  "Phi-3.5-mini-instruct-4bit"
  "Phi-3.5-MoE-instruct-4bit"
  "quantized-gemma-2b-it"
  "Qwen1.5-0.5B-Chat-4bit"
  "Qwen2.5-1.5B-Instruct-4bit"
  "Qwen2.5-7B-Instruct-4bit"
  "Qwen3-0.6B-4bit"
  "Qwen3-1.7B-4bit"
  "Qwen3-30B-A3B-4bit"
  "Qwen3-4B-4bit"
  "Qwen3-8B-4bit"
  "Qwen3-VL-2B-Instruct-4bit"
  "Qwen3-VL-2B-Thinking-4bit"
  "Qwen3-VL-4B-Instruct-4bit"
  "Qwen3-VL-4B-Thinking-4bit"
  "SmolLM-135M-Instruct-4bit"
)

id="${1}"

if [[ -z "${id}" ]]; then
  echo "Choose a model from:"

  for model in ${ids[@]}; do
    echo "* ${model}"
  done

  echo

  exit 2
fi

project=$(dirname "$0")
pushd "$project/.." &>/dev/null

if [[ "${id}" == *"/"* ]]; then
  repo_path="${id}"
else
  repo_path="mlx-community/${id}"
fi

model_name=$(basename "${id}")

metaurl="https://huggingface.co/api/models/${repo_path}"
echo "${metaurl}"

files=$(curl -s "${metaurl}" | jq -r '.siblings[].rfilename')
echo "${files}"

pushd Sources/SHLLM/Resources &>/dev/null

mkdir -p "${model_name}"

pushd "${model_name}" &>/dev/null

for file in ${files[@]}; do
  curl -L -# -o "${file}" "https://huggingface.co/${repo_path}/resolve/main/${file}"
done

popd
popd
