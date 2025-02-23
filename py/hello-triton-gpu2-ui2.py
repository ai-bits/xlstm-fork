# 20250223 2320
# o3-mini-high was almost as tiring as 4o: much trial and error
# optimizing the stream and then get streaming right again.
# But I have no idea how many MONTHS it would have taken me on my own and
# if it had worked at all, if I had had to rely on pre-AI procedures.

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
import gc

model_directory = "/home/gy/dl/xLSTM-7b"  # Local model path

# Load model config with proper settings
try:
    xlstm_config = AutoConfig.from_pretrained(model_directory)
    xlstm_config.weight_mode = "fused" #new setting
    xlstm_config.step_kernel = "triton" #changed settings...
    xlstm_config.chunkwise_kernel = "chunkwise--triton_limit_chunk"
    xlstm_config.sequence_kernel = "native_sequence__triton"
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
def generate_text_stream(prompt, max_tokens=1000, max_context=512):
    """
    Generate text from a prompt using Triton-based kernels, yielding one complete word at a time.
    
    The function assumes the model (xlstm) is configured to use Triton kernels.
    It accumulates decoded tokens into a buffer. When the buffer contains a space,
    every complete word (i.e. every substring ending in a space) is split and yielded
    (each with a trailing space). The remainder is kept until further tokens complete it.
    
    Args:
        prompt (str): The initial prompt.
        max_tokens (int): Maximum tokens to generate.
        max_context (int): Maximum number of tokens used as context.
        
    Yields:
        str: The next complete word (with a trailing space) once it is fully generated.
    """
    try:
        # Build initial context from the prompt.
        tokenized = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
        context_tokens = tokenized.tolist()
        device = next(xlstm.parameters()).device

        buffer = ""
        for i in range(max_tokens):
            # Prepare input using only the last max_context tokens.
            input_ids = torch.tensor([context_tokens[-max_context:]], device=device)
            output = xlstm.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True
            )
            next_token = output[0, -1].item()
            if next_token == tokenizer.eos_token_id:
                if buffer:
                    yield buffer  # Yield any leftover text.
                break

            # Decode without cleaning up spaces so that token boundaries are preserved.
            token_str = tokenizer.decode(
                [next_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            buffer += token_str
            context_tokens.append(next_token)
            
            # When the buffer contains a space, split out complete words.
            if " " in buffer:
                parts = buffer.split(" ")
                # All parts except the last are complete words.
                for word in parts[:-1]:
                    if word:  # yield non-empty word with a trailing space
                        yield word + " "
                    else:
                        # In case there are multiple spaces.
                        yield " "
                # Keep the incomplete part (after the last space) in the buffer.
                buffer = parts[-1]

            # Periodic cleanup to help with GPU memory.
            if i % 100 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                
        # After finishing generation, yield any remaining text.
        if buffer:
            yield buffer

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
