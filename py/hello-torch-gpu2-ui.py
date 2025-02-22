# 20250222 2320 WORKING!
# Was tiring with OpenAI's 4o trial and error
# to get streaming right and debug the CUDA out of memory issue,
# but I have no idea how many MONTHS it would have taken me on my own and
# if it had worked at all, if I had had to rely on pre-AI procedures.

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
import gc

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
        
        past_key_values = None  # Keep track of past states for efficiency
        
        new_tokens = []  # Buffer for batched token streaming
        for i in range(max_tokens):
            if not new_tokens:  # Generate in batches of 10 to reduce overhead
                output = xlstm.generate(
                    tokens[:, -512:],  # Keep only the last 512 tokens
                    max_new_tokens=10,  # Generate 10 tokens at a time
                    do_sample=False, 
                    use_cache=True  # Enable cache for efficiency
                )
                new_tokens = output[0, -10:].tolist()  # Store next 10 tokens
            
            next_token = new_tokens.pop(0)  # Stream tokens from buffer
            
            if next_token == tokenizer.eos_token_id:
                break
            
            new_word = tokenizer.decode([next_token], skip_special_tokens=True)
            if new_word:
                yield (' ' + new_word if len(tokenizer.tokenize(new_word)) > 1 else new_word)  # Ensure spacing between words
            
            tokens = torch.cat((tokens, torch.tensor([[next_token]], device=first_device)), dim=1)
            tokens = tokens[:, -512:]  # Limit context length to prevent memory issues
            
            # Free GPU memory every 50 tokens to prevent memory growth
            if i % 50 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
    except torch.cuda.CudaError as e:
        print(f"CUDA error: {e}")
        torch.cuda.empty_cache()
        yield "[Error: CUDA out of memory]"
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
