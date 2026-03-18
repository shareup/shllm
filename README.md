# SHLLM

Run LLMs on Apple devices with Swift and MLX.

SHLLM provides a high-level async/streaming API for running large language models on-device. It wraps quantized models with a unified `AsyncSequence` interface, supporting text generation, reasoning, vision, and tool calling.

## Requirements

- Swift 5.12+
- macOS 14+, iOS 17+, or Mac Catalyst 17+
- Metal-capable device

## Installation

Add SHLLM to your project via Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/shareup/shllm", from: "0.13.0"),
]
```

Then add `"SHLLM"` as a dependency of your target.

## Quick Start

```swift
import SHLLM

let input = UserInput(chat: [
    .system("You are a helpful assistant."),
    .user("What is the meaning of life?"),
])

let llm = try LLM.qwen3(
    directory: modelDirectory,
    input: input
)

for try await response in llm {
    switch response {
    case .text(let text):
        print(text, terminator: "")
    case .reasoning(let thought):
        print("[thinking] \(thought)", terminator: "")
    case .toolCall(let call):
        print("Tool call: \(call.function.name)")
    }
}
```

## Usage

### Streaming Responses

`LLM` conforms to `AsyncSequence`, yielding `Response` values:

```swift
public enum Response {
    case reasoning(String)
    case text(String)
    case toolCall(ToolCall)
}
```

Iterate with `for try await`:

```swift
for try await response in llm {
    switch response {
    case .text(let text):
        print(text, terminator: "")
    case .reasoning(let thought):
        // Handle reasoning/thinking tokens
        break
    case .toolCall(let call):
        // Handle tool calls
        break
    }
}
```

### Text-Only Streaming

The `.text` property returns a `TextAsyncSequence` that filters to only text tokens:

```swift
for try await text in llm.text {
    print(text, terminator: "")
}
```

### Awaiting Complete Results

Use `.result` to collect the full response:

```swift
let (reasoning, text, toolCalls) = try await llm.result
// reasoning: String? — thinking/reasoning content
// text: String? — generated text
// toolCalls: [ToolCall]? — any tool calls made
```

Or for text only:

```swift
let text = try await llm.text.result
```

### Reasoning Models

Models like Qwen3 support a thinking/reasoning mode. The `qwen3` factory method automatically configures the response parser to separate reasoning from text output:

```swift
let llm = try LLM.qwen3(
    directory: modelDirectory,
    input: input
)

for try await response in llm {
    switch response {
    case .reasoning(let thought):
        // Internal reasoning tokens
        break
    case .text(let text):
        // Final response text
        print(text, terminator: "")
    case .toolCall:
        break
    }
}
```

### Vision Models

Vision-language models accept image input via URL or `Data`. The `Qwen3VL` type requires an additional import:

```swift
import MLXVLM

let llm = try LLM.qwen3VL(
    directory: modelDirectory,
    input: UserInput(chat: [
        .system("You are a helpful assistant."),
        .user("Describe this image.", images: [.url(imageURL)]),
    ]),
    responseParser: LLM<Qwen3VL>.qwen3VLInstructParser
)
```

### Tool Calling

Define tools with `Tool<Input, Output>` and pass them to the LLM:

```swift
struct WeatherInput: Codable {
    let location: String
}

struct WeatherOutput: Codable {
    let temperature: Double
    let condition: String
}

let weatherTool = Tool<WeatherInput, WeatherOutput>(
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: [
        .required("location", type: .string, description: "The city name"),
    ],
    handler: { input in
        WeatherOutput(temperature: 72.0, condition: "sunny")
    }
)

let llm = try LLM.qwen3(
    directory: modelDirectory,
    input: input,
    tools: [weatherTool]
)

for try await response in llm {
    switch response {
    case .toolCall(let call):
        print("Function: \(call.function.name)")
        print("Arguments: \(call.function.arguments)")
    case .text(let text):
        print(text, terminator: "")
    case .reasoning:
        break
    }
}
```

## Supported Models

| Family | Model Type | Factory Method |
|---|---|---|
| DeepSeek R1 | `Qwen2Model` | `deepSeekR1` |
| **Devstral** | `Mistral3VLM` | `devstral2` |
| Gemma 2 | `Gemma2Model` | `gemma2` |
| Gemma 3 | `Gemma3TextModel` | `gemma3`, `gemma3_1B` |
| **GPT-OSS** | `GPTOSSModel` | `gptOSS_20B` |
| LFM-2 | `LFM2MoEModel` | `lfm2` |
| Llama 3 | `LlamaModel` | `llama3` |
| **Ministral** | `Mistral3VLM` | `ministral` |
| Mistral | `LlamaModel` | `mistral` |
| **Nemotron** | `NemotronHModel` | `nemotron3Nano` |
| OpenELM | `OpenELMModel` | `openELM` |
| Orchestrator | `Qwen3Model` | `orchestrator` |
| Phi 2 | `PhiModel` | `phi2` |
| Phi 3.5 | `Phi3Model` | `phi3` |
| Phi MoE | `PhiMoEModel` | `phiMoE` |
| Qwen 1.5 | `Qwen2Model` | `qwen1_5` |
| Qwen 2.5 | `Qwen2Model` | `qwen2_5` |
| Qwen 3 | `Qwen3Model` | `qwen3` |
| Qwen 3 MoE | `Qwen3MoEModel` | `qwen3MoE` |
| Qwen 3 VL | `Qwen3VL` | `qwen3VL` |
| **Qwen 3.5** | `Qwen35` | `qwen3_5` |
| **Qwen 3.5 MoE** | `Qwen35MoE` | `qwen3_5MoE` |
| SmolLM | `LlamaModel` | `smolLM` |

Each factory method takes `directory`, `input`, and optional parameters for `tools`, `maxInputTokenCount`, and `maxOutputTokenCount`.

## Configuration

### Generation Parameters

Customize generation with `GenerateParameters`:

```swift
let params = GenerateParameters(
    temperature: 0.7,
    topP: 0.9
)

let llm = LLM<Qwen3Model>(
    directory: modelDirectory,
    input: input,
    generateParameters: params
)
```

Each factory method provides sensible defaults for its model family.

### Token Limits

Control input and output token counts:

```swift
let llm = try LLM.qwen3(
    directory: modelDirectory,
    input: input,
    maxInputTokenCount: 4096,
    maxOutputTokenCount: 2048
)
```

### Model Caching

SHLLM caches loaded models in memory for reuse:

```swift
SHLLM.isModelCacheEnabled = true   // enabled by default
SHLLM.cacheLimit = 1_000_000_000   // cache size limit in bytes
SHLLM.clearCache()                 // clear the model cache
```

### Device Support

Check for Metal support before loading models:

```swift
guard SHLLM.isSupportedDevice else {
    fatalError("This device does not support Metal")
}
```

## License

[MIT](LICENSE)
