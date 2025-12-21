# Contributing to CogniXpert v1.0

Thanks for your interest in improving CogniXpert. Contributions of code, docs, evaluations, and safety improvements are welcome.

## Ways to Contribute

- Report bugs or issues
- Improve documentation and examples
- Propose new evaluation scripts or prompts
- Contribute training or alignment recipes
- Optimize inference and memory usage

## Development Setup

- Python 3.10+
- `pip install -U transformers unsloth peft bitsandbytes`
- Optional GPU acceleration: CUDA 12.x with a recent NVIDIA driver

## Pull Request Guidelines

- Fork the repo and create a topic branch
- Keep changes focused and incremental
- Update docs and examples when behavior changes
- Add usage notes for new configs or flags
- Ensure code is free of secrets or proprietary data

## Coding and Docs Style

- Prefer clear, simple Python samples
- Use `device_map="auto"` for examples unless reasoned otherwise
- Keep README snippets runnable
- Write concise commit messages in imperative mood

## Safety and Scope

- Do not claim medical capability; include help‑seeking guidance
- Avoid training data that identifies individuals
- Note limitations and potential biases in evaluations

## Issue Triage

- `bug`: malfunction or incorrect behavior
- `docs`: documentation improvements
- `perf`: performance or memory optimization
- `safety`: alignment or guardrails
- `feature`: new recipes or capabilities

## Release and Weights

- LoRA adapters should include `adapter_config.json` and weights files
- Reference base model IDs and versions used during training
- Document data sources and filtering where possible

## License

By contributing, you agree your contributions are licensed under AGPL‑3.0.

