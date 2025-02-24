# --- Begin monkey-patching if needed ---
try:
    from utils import round_up_to_next_multiple_of
except ImportError:
    def round_up_to_next_multiple_of(x, multiple_of):
        return ((x + multiple_of - 1) // multiple_of) * multiple_of

try:
    import xlstm.xlstm_large.utils as xlstm_utils
    xlstm_utils.round_up_to_next_multiple_of = round_up_to_next_multiple_of
except Exception as e:
    print(f"Error patching xlstm.xlstm_large.utils: {e}")
    exit(1)
# --- End monkey-patching ---

#tries to convert config.json to config.yaml on the fly, but fails somehow. file not in model dir

import os
import json
import torch
import gc
import importlib
from safetensors.torch import load_file
from pprint import pprint

model_directory = "/home/gy/dl/xLSTM-7b"
config_json_path = os.path.join(model_directory, "config.json")

try:
    with open(config_json_path, "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"Error loading config.json: {e}")
    exit(1)

# Create config.yaml if it doesn't exist (since some functions expect it)
config_yaml_path = os.path.join(model_directory, "config.yaml")
if not os.path.exists(config_yaml_path):
    try:
        import yaml
    except ImportError:
        print("PyYAML not found; please install PyYAML or manually create a config.yaml")
        exit(1)
    with open(config_yaml_path, "w") as f:
        yaml.dump(config, f)

def load_model_and_tokenizer():
    # Delay importing model classes to avoid circular dependency issues.
    try:
        from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
    except Exception as e:
        print(f"Error importing model classes: {e}")
        exit(1)
    
    try:
        xlstm_config = xLSTMLargeConfig(**config)
    except Exception as e:
        print(f"Error instantiating xLSTMLargeConfig: {e}")
        exit(1)
    
    # For Triton kernels, set weight_mode to "fused".
    xlstm_config.weight_mode = "fused"
    
    # Load checkpoint weights.
    try:
        checkpoint_path = os.path.join(model_directory, "model.safetensors")
        if os.path.exists(checkpoint_path):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = {}
            n = 0
            while True:
                shard_path = os.path.join(model_directory, f"model_{n}.safetensors")
                if not os.path.exists(shard_path):
                    break
                state_dict.update(load_file(shard_path))
                n += 1
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    
    try:
        model = xLSTMLarge(xlstm_config)
    except Exception as e:
        print(f"Error instantiating model: {e}")
        exit(1)
    
    try:
        from xlstm.xlstm_large.utils import convert_single_weights_to_fused_weights
        state_dict = convert_single_weights_to_fused_weights(state_dict)
        model.load_state_dict(state_dict)
        model = model.to("cuda")
    except Exception as e:
        print(f"Error converting weights: {e}")
        exit(1)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)
    
    return model, tokenizer

@torch.no_grad()
def generate_text_stream(prompt, model, tokenizer, max_tokens=1000, max_context=512):
    """
    Generate text from a prompt, yielding one complete word at a time.
    
    The function accumulates decoded tokens into a buffer. When the buffer contains a space,
    complete words (each ending with a space) are yielded, leaving any incomplete word in the buffer.
    
    Args:
        prompt (str): The input prompt.
        model: The loaded model.
        tokenizer: The tokenizer.
        max_tokens (int): Maximum number of tokens to generate.
        max_context (int): Maximum number of tokens used as context.
    
    Yields:
        str: The next complete word (with trailing space) when available.
    """
    try:
        tokenized = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
        context_tokens = tokenized.tolist()
        device = next(model.parameters()).device

        buffer = ""
        for i in range(max_tokens):
            input_ids = torch.tensor([context_tokens[-max_context:]], device=device)
            output = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True
            )
            next_token = output[0, -1].item()
            if next_token == tokenizer.eos_token_id:
                if buffer:
                    yield buffer
                break

            token_str = tokenizer.decode(
                [next_token],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            buffer += token_str
            context_tokens.append(next_token)
            
            if " " in buffer:
                parts = buffer.split(" ")
                for word in parts[:-1]:
                    if word:
                        yield word + " "
                    else:
                        yield " "
                buffer = parts[-1]
            
            if i % 100 == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
        
        if buffer:
            yield buffer

    except torch.cuda.CudaError as e:
        print(f"CUDA error: {e}")
        torch.cuda.empty_cache()
        yield "[Error: CUDA out of memory]"
    except Exception as e:
        print(f"Error during text generation: {e}")
        yield "Error generating text."

if __name__ == "__main__":
    print("xLSTM Chatbot - Type 'exit' to quit")
    model, tokenizer = load_model_and_tokenizer()
    try:
        pprint(model.backbone.blocks[0].mlstm_layer.config)
    except Exception as e:
        print(f"Error accessing model configuration: {e}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        max_tokens = input("Max tokens (default 1000): ")
        max_tokens = int(max_tokens) if max_tokens.isdigit() else 1000

        print("AI:", end="", flush=True)
        for word in generate_text_stream(user_input, model, tokenizer, max_tokens):
            print(word, end="", flush=True)
        print("\n")
