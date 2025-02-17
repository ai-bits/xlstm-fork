from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time

model_directory = "/home/gy/dl/xLSTM-7b"

# Load the model configuration.
xlstm_config = AutoConfig.from_pretrained(model_directory)
# Optionally, adjust kernel settings for your Triton build if needed:
# xlstm_config.step_kernel = "triton"
# xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
# xlstm_config.sequence_kernel = "native_sequence__native"

# Load the model with automatic device mapping to distribute it across available GPUs.
xlstm = AutoModelForCausalLM.from_pretrained(
    model_directory,
    config=xlstm_config,
    device_map="auto"
)

# Optionally, verify selected kernels.
from pprint import pprint
pprint(xlstm.backbone.blocks[0].mlstm_layer.config)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Tokenize the input.
tokens = tokenizer("Hello xLSTM, how are you doing?", return_tensors='pt')['input_ids']

# Move tokens to the same device as the model's first parameter (usually cuda:0).
first_device = next(xlstm.parameters()).device
tokens = tokens.to(first_device)

# Generate output using the distributed model.
start_time = time.time()
out = xlstm.generate(tokens, max_new_tokens=20)
generation_time = time.time() - start_time

# Decode and print the output.
print(tokenizer.decode(out[0]))
print(f"Generation time: {generation_time:.2f} seconds")