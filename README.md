### CogniXpert-AI Model v1.0:

- Safety‑aware, non‑clinical conversational AI for supportive mental health and wellbeing use‑cases, built on Meta Llama 3.1 8B and fine‑tuned with LoRA. This repository contains the model configuration, tokenizer, generation defaults, and adapter metadata for efficient deployment.

- **Disclaimer:** This project is intended for supportive, non‑clinical use. It does not replace care from licensed mental health professionals and does not provide diagnosis or treatment. If you are in crisis or may harm yourself or others, contact emergency services or your local suicide prevention hotline immediately.

![License](https://img.shields.io/badge/License-AGPL--3.0-blue)
![Model](https://img.shields.io/badge/Model-Llama_3.1_8B-green)
![Quantization](https://img.shields.io/badge/Quantization-4bit_(NF4)-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.47.1-purple)
![Unsloth](https://img.shields.io/badge/Unsloth-2024.9-teal)

**Note:** For Hugging Face model card rendering, use `README.hf.md` (includes YAML metadata).

### Research & Open Innovation for Mental Health:

- We’re giving developers, researchers, academia, non‑profits, and mental health advocates the foundation tools to build better care through open‑source contribution.
Foundational fine‑tuned model developed by CogniX LTD.

### Highlights:

- 8B Llama 3.1 backbone with long‑context (131k) support
- 4‑bit quantization via BitsAndBytes for single‑GPU inference
- LoRA fine‑tuning targeting core attention/MLP projections
- Conversational alignment with supportive, coaching‑style responses
- Ready for Python `transformers` + `unsloth` workflows

### Model Description:

- CogniXpert v1.0 is a LoRA‑tuned variant of Llama 3.1 8B optimized for safe, empathetic conversation. The adapter focuses on key transformer projection modules, and default generation settings favor stable, coherent replies. Tokenizer configuration preserves the Llama 3 special tokens and right‑padding for batched inference.

### Dataset Sources:

- Native datasets: open mental health dialogue corpora curated for supportive conversation and coaching contexts.
- Synthetic datasets: additional coaching‑style dialogues generated using OpenAI models to augment coverage and style diversity.
- Fine‑tuning combined both native and synthetic sources with safety‑oriented filtering and prompt design.

 
### Model Details:

- Base model: `Meta Llama 3.1 8B`
- Context length: `131,072`
- Quantization: `4‑bit` (`bitsandbytes`, NF4, bfloat16 compute)
- LoRA params: `r=16`, `alpha=16`, `dropout=0`
- Target modules: `v_proj, gate_proj, up_proj, o_proj, q_proj, down_proj, k_proj`
- Dtype: `float16`
- Libraries: `transformers==4.47.1`, `unsloth==2024.9`, `peft`

### Generation Configuration:

- Temperature: `0.6`
- Top‑p: `0.9`
- Sampling: enabled
- Max length: inherits long‑context defaults

### Quick Start:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Base-only inference (loads Unsloth 4-bit backbone)
MODEL_ID = "unsloth/meta-llama-3.1-8b-bnb-4bit"
LOCAL_DIR = "c:/Users/Public/CogniXpert-Model-v1.0"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

# Streamed generation
inputs = tokenizer("Hello, how can I help you today?", return_tensors="pt").to(model.device)
streamer = TextStreamer(tokenizer, skip_prompt=True)
outputs = model.generate(**inputs, max_new_tokens=200, streamer=streamer)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Streamlit Demo:

- Install dependencies: `pip install -U streamlit transformers peft bitsandbytes unsloth accelerate`
- Run the app: `streamlit run app.py`
- The app loads the Unsloth 4‑bit base and attaches the LoRA adapter if `adapter_config.json` is present in the repo.

Alternatively: `pip install -r requirements.txt`

### Using the LoRA Adapter:

- If you have the LoRA adapter weights (e.g., `adapter_model.bin`) for CogniXpert v1.0, you can attach them to the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_ID = "unsloth/meta-llama-3.1-8b-bnb-4bit"
LOCAL_DIR = "c:/Users/Public/CogniXpert-Model-v1.0"  # contains adapter_config.json
ADAPTER_DIR = LOCAL_DIR  # place adapter weights here (adapter_model.bin)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)  # loads LoRA

prompt = "Share three supportive coping strategies for mild stress."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.6, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Restoring Adapter Weights:

- Place `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin` in the project root (`c:/Users/Public/CogniXpert-Model-v1.0`).
- The Streamlit demo auto‑attaches the adapter only when both config and weights are present; otherwise it runs base‑only and shows a warning.

### Chat Prompting:

- Llama 3 models expect structured headers and end‑of‑turn markers:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are CogniXpert, a supportive, safety-aware assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
I feel overwhelmed at work. Any suggestions?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

Terminate the assistant’s reply with `<|eot_id|>` when streaming multiple turns.

### Hardware & Performance:

- Runs on a single modern GPU with 4‑bit quantization; 16GB+ VRAM recommended
- Up to 131k tokens context; long prompts require sufficient memory and batching care
- Use `device_map="auto"` to distribute layers when available

### Safety, Scope, and Limitations:

- Intended for supportive conversation and coaching. It is not a medical device and does not provide diagnosis or treatment.
- Encourages help‑seeking, evidence‑based coping strategies, and urgent resource guidance where appropriate.
- May reflect biases present in training sources; review outputs before production use.
- Avoid high‑risk clinical decision support without human oversight.

### Reproducibility (Unsloth + PEFT sketch):

```python
# Pseudocode sketch of LoRA fine-tuning configuration
from unsloth import FastLanguageModel
from peft import LoraConfig

base = FastLanguageModel.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")
config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["v_proj","gate_proj","up_proj","o_proj","q_proj","down_proj","k_proj"],
    task_type="CAUSAL_LM",
)
# ...dataset preparation & training loop...
# Save with: model.save_pretrained(ADAPTER_DIR)  # produces adapter_model.bin + adapter_config.json
```

### Contribute:

- Improve safety alignment, prompting, and guardrails
- Add evaluation scripts, probes, and benchmarks
- Optimize inference throughput and memory footprint
- Curate or synthesize datasets with transparent provenance
- Enhance documentation, tutorials, and translations
- Report issues and propose features

Read `CONTRIBUTING.md` to get started. Open issues for discussion and submit focused pull requests. Community standards are defined in `CODE_OF_CONDUCT.md`. For responsible disclosure, see `SECURITY.md`.

### Roadmap:

- Release reproducible training scripts and evaluation suite
- Publish LoRA adapter weights with versioned artifacts
- Expand datasets with multilingual supportive dialogues
- Improve safety alignment and escalation guidance
- Optimize inference for CPU and low‑VRAM devices
- Add tutorials and notebooks for deployment and monitoring

### Optional Next Steps:

- Add CI with GitHub Actions for lint/type import and Streamlit smoke tests
- Publish versioned releases and attach LoRA adapter artifacts
- Move Hugging Face model card YAML to the top for HF rendering
- Provide `CITATION.cff` for academic citation and indexing
- Create a docs site (MkDocs/Docusaurus) with tutorials and API references
- Add example notebooks for inference, alignment, and evaluation

### Hugging Face Sync:

- Set environment variables: `HF_TOKEN=<your_token>` and `HF_REPO_ID=<org/model>`
- Install: `pip install -r requirements.txt`
- Sync model card: `python tools/sync_hf_readme.py` or `python tools/sync_hf_readme.py <org/model>`
- This replaces the HF repo `README.md` with `README.hf.md` so metadata renders correctly.

### Acknowledgements:

- Foundational fine‑tuned model dev engineered by CogniX LTD.
- Meta Llama 3.1 for the base architecture
- OpenAI for synthetic dataset generation used to augment training data
- Open‑source contributors in safety‑aligned conversational AI
- Libraries: `transformers`, `unsloth`, `peft`, `bitsandbytes`

### License:

- This repository is licensed under `AGPL‑3.0`. The underlying base model (Meta Llama 3.1) is subject to Meta’s license; ensure compliance when distributing derivatives or weights.
