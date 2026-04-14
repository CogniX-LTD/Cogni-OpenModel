import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LOCAL_DIR = "c:/Users/Public/Cogni-OpenModel"
BASE_ID = "unsloth/meta-llama-3.1-8b-bnb-4bit"

@st.cache_resource
def load_model(use_adapter: bool):
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, device_map="auto")
    cfg_path = os.path.join(LOCAL_DIR, "adapter_config.json")
    safetensors_path = os.path.join(LOCAL_DIR, "adapter_model.safetensors")
    bin_path = os.path.join(LOCAL_DIR, "adapter_model.bin")
    has_config = os.path.exists(cfg_path)
    has_weights = os.path.exists(safetensors_path) or os.path.exists(bin_path)
    if use_adapter and has_config and has_weights:
        base = PeftModel.from_pretrained(base, LOCAL_DIR)
    elif use_adapter and has_config and not has_weights:
        st.warning("LoRA adapter config found but weights missing. Proceeding without adapter.")
    return tok, base

def format_prompt(system_text: str, messages: list[str]):
    s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system_text + "<|eot_id|>\n"
    content = s
    for i in range(len(messages)):
        if i % 2 == 0:
            content += "<|start_header_id|>user<|end_header_id|>\n" + messages[i] + "<|eot_id|>\n"
        else:
            content += "<|start_header_id|>assistant<|end_header_id|>\n" + messages[i] + "<|eot_id|>\n"
    content += "<|start_header_id|>assistant<|end_header_id|>\n"
    return content

st.set_page_config(page_title="Cogni-OpenModel Chat", page_icon="🧠", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Cogni-OpenModel Chat")
st.caption("Supportive, safety‑aware conversational AI. Not medical advice.")

use_adapter = st.sidebar.checkbox("Use LoRA adapter if available", value=True)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.6, 0.05)
top_p = st.sidebar.slider("Top‑p", 0.1, 1.0, 0.9, 0.05)
max_new_tokens = st.sidebar.slider("Max new tokens", 32, 1024, 256, 32)

system_default = "You are Cogni-OpenModel, a supportive, safety‑aware assistant. Encourage help‑seeking and evidence‑based coping strategies. Avoid clinical diagnosis or prescriptive treatment."
system_text = st.text_area("System prompt", value=system_default, height=100)

tok, model = load_model(use_adapter)

for i, msg in enumerate(st.session_state.messages):
    role = "assistant" if i % 2 == 1 else "user"
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Type your message")

if user_input:
    st.session_state.messages.append(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    prompt = format_prompt(system_text, st.session_state.messages)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    text = tok.decode(out[0], skip_special_tokens=False)
    key = "<|start_header_id|>assistant<|end_header_id|>"
    idx = text.rfind(key)
    resp = text[idx + len(key):]
    eot = resp.find("<|eot_id|>")
    if eot != -1:
        resp = resp[:eot]
    resp = resp.strip()
    st.session_state.messages.append(resp)
    with st.chat_message("assistant"):
        st.markdown(resp)
