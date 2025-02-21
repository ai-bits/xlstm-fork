#repo_path = "/home/gy/dl/xlstm-fork/xlstm/xlstm_large"  # Adjust this path
#MODEL_PATH = "/home/gy/dl/xLSTM-7"

import sys
import os

# Add the xlstm_large directory to Python’s path
repo_path = os.path.abspath("/home/gy/dl/xlstm-fork/xlstm/xlstm_large")
sys.path.append(repo_path)

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from from_pretrained import load_from_pretrained  # Now it should work

# Load model and tokenizer
MODEL_PATH = "//home/gy/dl/xLSTM-7"
xlstm = load_from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_response(prompt, max_length=256):
    """Generates text using xLSTM"""
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(xlstm.device)
    generated_tokens, _ = xlstm.generate(prefill_tokens=tokens, max_length=max_length)
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(label="Input Prompt"), gr.Slider(50, 512, value=256, step=10, label="Max Length")],
    outputs=gr.Textbox(label="Generated Response"),
    title="xLSTM Chat",
    description="Chat with an xLSTM-powered AI model.",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
