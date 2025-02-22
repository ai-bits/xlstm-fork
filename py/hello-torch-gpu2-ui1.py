# 20250222 Was a bit tiring with OpenAI's 4o trail and error
# to get streaming almost right (CUDA out of memory or so after an estimated 700 tokens),
# but I have no idea how many MONTHS it would have taken me on my own.

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

# Stream output token by token
@torch.no_grad()
def generate_text_stream(prompt, max_tokens=1000):
    try:
        tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        first_device = next(xlstm.parameters()).device
        tokens = tokens.to(first_device)
        
        generated_tokens = []
        for _ in range(max_tokens):
            output = xlstm.generate(tokens, max_new_tokens=1, do_sample=False)
            next_token = output[0, -1].item()
            
            if next_token == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            new_word = tokenizer.decode([next_token], skip_special_tokens=True)
            yield new_word  # Yield only the new token
            
            tokens = torch.cat((tokens, torch.tensor([[next_token]], device=first_device)), dim=1)
            #time.sleep(0.05)  # commented out as it's slow enough on my 2 x 20GB RTX 4000
    except Exception as e:
        print(f"Error during text generation: {e}")
        yield "Error generating text."

# Interactive loop for user input
if __name__ == "__main__":
    print("xLSTM Chatbot - Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        max_tokens = input("Max tokens (default 1000): ")
        max_tokens = int(max_tokens) if max_tokens.isdigit() else 1000
        
        print("AI:", end="", flush=True)
        for word in generate_text_stream(user_input, max_tokens):
            print(word, end="", flush=True)
        print("\n")