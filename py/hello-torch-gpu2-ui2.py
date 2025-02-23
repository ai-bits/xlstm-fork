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
def generate_text_stream(prompt, max_tokens=1000, max_context=512):
    """
    Generate text from a prompt, yielding one complete word at a time.
    
    The function accumulates decoded tokens into a buffer. Whenever the buffer
    contains one or more space characters, all complete words (i.e. every substring 
    ending with a space) are split out and yielded (with a trailing space), leaving any
    incomplete word in the buffer for later completion.
    
    Args:
        prompt (str): The initial text prompt.
        max_tokens (int): Maximum number of tokens to generate.
        max_context (int): Maximum number of tokens to use as context.
    
    Yields:
        str: The next complete word (with a trailing space) as soon as it is fully generated.
    """
    try:
        # Build initial context from the prompt.
        tokenized = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
        context_tokens = tokenized.tolist()
        device = next(xlstm.parameters()).device

        # Buffer holds generated text that hasn't yet been output as complete words.
        buffer = ""
        
        for i in range(max_tokens):
            # Prepare input tensor using only the last max_context tokens.
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
                    yield buffer  # yield any leftover text
                break

            # Decode the token; do not clean up spaces to preserve token boundaries.
            token_str = tokenizer.decode(
                [next_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            buffer += token_str
            context_tokens.append(next_token)
            
            # If the buffer contains a space, split out complete words.
            if " " in buffer:
                # Split on space; every element except the last is a complete word.
                parts = buffer.split(" ")
                for word in parts[:-1]:
                    if word:  # yield non-empty words with a trailing space
                        yield word + " "
                    else:
                        # If an empty string appears, it means multiple spaces.
                        yield " "
                # Keep the incomplete part (after the last space) in the buffer.
                buffer = parts[-1]
            
            # Periodically trigger cleanup.
            if i % 100 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
        
        # After finishing generation, yield any text left in the buffer.
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
