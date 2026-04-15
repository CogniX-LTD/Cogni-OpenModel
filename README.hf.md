---
license: mit
base_model: unsloth/meta-llama-3.1-8b-bnb-4bit
tags:
- llama
- llama-3.1
- conversational-ai
- mental-health
- lora
- quantization
- text-generation-inference
- bitsandbytes
- 4-bit
model_name: Cogni-OpenModel
pipeline_tag: text-generation
language:
- en
---

### Cogni-OpenModel:

- Safety‑aware, non‑clinical conversational AI for supportive mental health and wellbeing use‑cases, built on Meta Llama 3.1 8B and fine‑tuned with LoRA. This repository contains the model configuration, tokenizer, generation defaults, and adapter metadata for efficient deployment.

- **Disclaimer:** This project is intended for supportive, non‑clinical use. It does not replace care from licensed mental health professionals and does not provide diagnosis or treatment. If you are in crisis or may harm yourself or others, contact emergency services or your local suicide prevention hotline immediately.

![License](https://img.shields.io/badge/License-MIT-blue)
![Model](https://img.shields.io/badge/Model-Llama_3.1_8B-green)
![Quantization](https://img.shields.io/badge/Quantization-4bit_(NF4)-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.47.1-purple)
![Unsloth](https://img.shields.io/badge/Unsloth-2024.9-teal)

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

- Cogni-OpenModel is a LoRA‑tuned variant of Llama 3.1 8B optimized for safe, empathetic conversation. The adapter focuses on key transformer projection modules, and default generation settings favor stable, coherent replies. Tokenizer configuration preserves the Llama 3 special tokens and right‑padding for batched inference.

### Dataset Sources:

- **Native datasets**: Open mental health dialogue corpora curated for supportive conversation and coaching contexts. This includes publicly available datasets such as **Counsel Chat** and **Psych8k**.
- **Synthetic datasets**: Additional coaching‑style dialogues generated using OpenAI models (**GPT‑4o**) to augment coverage and style diversity.
- **Release & Licensing**: Fine‑tuning combined both native and synthetic sources with safety‑oriented filtering and prompt design. Full dataset provenance will be released with our open dataset under a **CC‑BY 4.0** license.

 
### Model Details:

- Base model: `Meta Llama 3.1 8B`
- Context length: `131,072`
- Quantization: `4‑bit` (`bitsandbytes`, NF4, bfloat16 compute)
- LoRA params: `r=16`, `alpha=16`, `dropout=0`
- Target modules: `v_proj, gate_proj, up_proj, o_proj, q_proj, down_proj, k_proj`
- Dtype: `float16`
- Libraries: `transformers==4.47.1`, `unsloth==2024.9`, `peft`

### Evaluation Results:

We evaluate Cogni-OpenModel using our **DeepEval/GEval framework** with custom rubrics tailored for African mental health contexts. All metrics score from 0 to 1 (higher is better).

#### Therapeutic Quality Metrics:

| Metric | Score | Description |
|:---|:---|:---|
| **Empathy** | 0.89 | Emotional attunement, validation, warmth, cultural sensitivity |
| **Active Listening** | 0.85 | Accurate reflection, open-ended questions, emotional pacing |
| **Motivational Interviewing Alignment** | 0.82 | Adherence to OARS framework (Open questions, Affirmations, Reflections, Summaries) |
| **Therapeutic Boundaries** | 0.94 | Refrains from diagnosis/prescription, maintains supportive role |
| **Overall Therapeutic Quality** | **0.88** | Composite score across all therapeutic metrics |

#### Safety & Robustness Metrics:

| Metric | Score | Description |
|:---|:---|:---|
| **Harmful Content Detection** | 0.97 | Blocks self-harm methods, substance abuse encouragement, dangerous remedies |
| **Crisis Handling** | 0.92 | Recognizes acute distress, prioritizes safety, provides crisis resources |
| **Bias Detection** | 0.88 | Penalizes gender/ethnic/religious stereotypes, respects cultural diversity |
| **Hallucination Prevention** | 0.91 | Avoids fabricating studies, inventing treatments, or making false claims |
| **Overall Safety Score** | **0.92** | Composite score across all safety metrics |

#### Comparison to Industry Benchmarks:

| Model | Therapeutic Quality | Safety | Context Window |
|:---|:---|:---|:---|
| **Cogni-OpenModel** | **0.88** | **0.92** | 131k |
| Llama 3.1 8B (base) | 0.71 | 0.78 | 128k |
| GPT-3.5 Turbo | 0.79 | 0.85 | 16k |
| Claude 3 Haiku | 0.82 | 0.87 | 200k |

*Note: Benchmark scores are approximations based on internal evaluations using identical rubrics.*

#### Responsible AI Alignment:

Our evaluation framework operationalizes Google's Responsible AI Principles:
- **Safety:** Crisis handling and harmful content metrics ensure user protection
- **Fairness:** Bias detection rubrics prevent stereotyping
- **Transparency:** Clear disclaimers and documentation
- **Human oversight:** Tiered escalation for high-risk cases

*Full evaluation suite and rubrics available at [https://github.com/CogniX-LTD/Cogni-OpenModel].*


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
LOCAL_DIR = "./"

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

- If you have the LoRA adapter weights (e.g., `adapter_model.bin`) for Cogni-OpenModel, you can attach them to the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_ID = "unsloth/meta-llama-3.1-8b-bnb-4bit"
LOCAL_DIR = "./"  # contains adapter_config.json
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

- Place `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin` in the project root.
- The Streamlit demo auto‑attaches the adapter only when both config and weights are present; otherwise it runs base‑only and shows a warning.

### Chat Prompting:

- Llama 3 models expect structured headers and end‑of‑turn markers:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Cogni-OpenModel, a supportive, safety-aware assistant.<|eot_id|>
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

### Acknowledgements:

- Foundational fine‑tuned model dev engineered by CogniX LTD.
- Meta Llama 3.1 for the base architecture
- OpenAI for synthetic dataset generation used to augment training data
- Open‑source contributors in safety‑aligned conversational AI
- Libraries: `transformers`, `unsloth`, `peft`, `bitsandbytes`

### License:

- This repository (LoRA adapter weights, configuration files, and code) is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.

- **Note on licensing:** The underlying base model (**Meta Llama 3.1 8B**) is **not** covered by the MIT License and remains subject to [Meta's Llama 3.1 Community License Agreement](https://llama.meta.com/llama3_1/license/). You must comply with Meta's license terms when using, distributing, or building upon the combined model.
