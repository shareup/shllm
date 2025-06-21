# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHLLM is a Swift library that provides a high-level interface for running Large Language Models (LLMs) on Apple devices using MLX. It wraps various LLM models with a unified async/streaming API and supports quantized models for efficient on-device inference.

## Development Commands

It's **very important** to use the following scripts (`bin/build.sh` or `bin/test.sh`) to build and test the library because `swift build` or `swift-test` do not build Metal, which SHLLM requires. 

```bash
# Build the library
bin/build.sh

# Run all tests
bin/test.sh

# Run tests for a specific model
bin/test.sh SHLLMTests/MODEL_NAMETests

# Format the code
bin/format.sh
```

## Architecture Overview

### Core Components

1. **LLM.swift** - Main LLM implementation that:
   - Implements `AsyncSequence` for streaming token generation
   - Wraps MLXLLM models with unified interface
   - Handles model loading from bundled resources
   - Manages token limits and generation parameters

2. **Tools.swift** - Function calling implementation:
   - `Tool` struct for defining callable functions with JSON schemas
   - `Tools` struct for managing multiple tools
   - Integrates with LLM for structured outputs

3. **Resources/** - Pre-quantized model files organized by model family:
   - Each model directory contains weights, config, and tokenizer files
   - Models are quantized to 4-bit or 8-bit for efficiency

### Key Patterns

- **Async/Await**: All LLM operations use Swift's async/await for non-blocking execution
- **AsyncSequence**: Token generation returns an AsyncSequence for streaming responses
- **Bundle Resources**: Models are embedded in the bundle and loaded at runtime
- **Metal Requirements**: Code checks for Metal support before model initialization

### Model Support

The library supports multiple model families, each with specific test files:

- Qwen (multiple versions including 3-2B)
- Gemma (2B, 9B variants)
- Llama (3.2-1B, 3.2-3B)
- Mistral (7B)
- OpenELM (270M, 450M, 1.1B, 3B)
- Phi (3, 3.5, 4)
- SmolLM (135M, 360M, 1.7B)
- DeepSeek R1 (1.5B)

### Dependencies

The project depends on:

- `mlx-swift-examples` (local dependency at ../mlx-swift-examples)
  - `mlx-swift-examples` is the developer’s fork of the upstream mlx-swift-examples. For development purposes, the mlx-swift-examples dependency can be loaded locally at ../mlx-swift-examples. However, typically, it points to the remote version at https://github.com/shareup/mlx-swift-examples
- `swift-transformers` for tokenization
- `swift-async-algorithms` for async utilities
- MLX framework for model execution

When upgrading dependencies, ensure compatibility with the MLXLLM interface. Because running local LLMs is slow and resource-intensive, typically, we only test the smallest modern model, which, in this case is Qwen3-0.6B-4bit. Before committing changes, the human will conduct more expansive tests, but Claude Code should selectively test only Qwen3-0.6B-4bit to ensure a fast development loop.
