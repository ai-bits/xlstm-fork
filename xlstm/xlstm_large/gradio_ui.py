#- in filename changed to _!!!
#conda activate xlstm
#cd xlstm-fork/xlstm/xlstm_large
#python -m xlstm_large.gradio_ui
import sys
import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the script runs as a module inside xlstm_large
if __name__ == "__main__":
    # Adjust Python path for package resolution
    repo_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(repo_path)

    from xlstm_large.from_pretrained import load_from_pretrained

    # Load model and tokenizer
    MODEL_PATH = "/home/gy/dl/xLSTM-7b"
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
