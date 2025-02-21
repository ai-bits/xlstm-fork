from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time

model_directory = "/home/gy/dl/xLSTM-7b"  # Local model path

# Load model config with proper settings
try:
    xlstm_config = AutoConfig.from_pretrained(model_directory)
    xlstm_config.step_kernel = "native"
    xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
    xlstm_config.sequence_kernel = "native_sequence__native"
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

# Load model with device mapping for GPU support
try:
    xlstm = AutoModelForCausalLM.from_pretrained(
        model_directory,
        config=xlstm_config,
        device_map="auto"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Verify selected kernels
from pprint import pprint
try:
    pprint(xlstm.backbone.blocks[0].mlstm_layer.config)
except Exception as e:
    print(f"Error accessing model configuration: {e}")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# Tokenize input with exception handling
def generate_text(prompt, max_tokens=1000):
    try:
        tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        first_device = next(xlstm.parameters()).device
        tokens = tokens.to(first_device)
        
        start_time = time.time()
        output_tokens = xlstm.generate(tokens, max_new_tokens=max_tokens)
        
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")
        return generated_text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Error generating text."

# Interactive loop for user input
if __name__ == "__main__":
    print("xLSTM Chatbot - Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        max_tokens = input("Max tokens (default 1000): ")
        max_tokens = int(max_tokens) if max_tokens.isdigit() else 1000
        
        print("AI:", end=" ")
        for token in generate_text(user_input, max_tokens).split():
            print(token, end=" ", flush=True)
        print("\n")
