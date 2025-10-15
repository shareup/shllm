# Repository Guidelines

## Project Structure & Module Organization
- Source: `Sources/SHLLM/` (core types like `LLM.swift`, `ResponseParser.swift`, tools, and bundled model `Resources/`).
- Tests: `Tests/SHLLMTests/` (Swift Testing with `@Test` plus `Resources/` for fixtures).
- Scripts: `bin/` (`build.sh`, `test.sh`, `format.sh`, `download.sh`).
- Config: `Package.swift`, `.swiftformat`, `.swift-version`, CI in `.github/workflows/ci.yml`.

## Build, Test, and Development Commands
- Build (Xcode/Metal aware): `bin/build.sh`
- Test all: `bin/test.sh`
- Test a file: `bin/test.sh SHLLMTests/ResponseParserTests`
- Test one case: `bin/test.sh 'SHLLMTests/SomeTests/testBehavior()'`
- Format: `bin/format.sh`
- Download a model (writes to `Sources/SHLLM/Resources/<MODEL>/`): `bin/download.sh <repo-or-id>`
Note: Prefer these scripts over `swift build/test` so Metal toolchain and Xcode settings are honored.

## Coding Style & Naming Conventions
- Use `swiftformat` with repo config: run `bin/format.sh` before committing.
- Line width 96; wrap args/params before first; no hoist of `await`/`try`.
- Indentation: 4 spaces; types `UpperCamelCase`, funcs/vars `lowerCamelCase`.
- Avoid inline comments; prefer clear names and small functions.

## Testing Guidelines
- Framework: Swift Testing (`import Testing`, `@Test`).
- File naming: end with `Tests.swift` (e.g., `Tool+SHLLMTests.swift`).
- Keep tests deterministic and small; prefer the smallest on-device model for fast runs.
- GPU is preferred; CPU inference may be disabled or slow. Run via `bin/test.sh`.

## Commit & Pull Request Guidelines
- Commits: short, imperative mood (e.g., "Fix crash in parser", "Update Qwen models").
- PRs must include: clear description, linked issue, test plan (commands run), and hardware context (CPU/GPU, model used). Attach logs for failures.
- CI must pass (`.github/workflows/ci.yml` runs `bin/test.sh` on macOS 15 / Xcode 16).

## Security & Configuration Tips
- Do not commit secrets or large model files not required for tests. Use `bin/download.sh` locally; `.gitignore` already excludes common artifacts.
- Requires macOS 14+, Xcode 16, and Metal-capable GPU for full functionality; `bin/test.sh` will prompt to install the Metal toolchain if missing.

