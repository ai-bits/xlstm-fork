from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time

model_directory = "/home/gy/dl/xLSTM-7b"

xlstm_config = AutoConfig.from_pretrained(model_directory)
xlstm_config.step_kernel = "native"
xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
xlstm_config.sequence_kernel = "native_sequence__native"

# Load the model with automatic device mapping; it will be distributed across available GPUs.
xlstm = AutoModelForCausalLM.from_pretrained(
    model_directory,
    config=xlstm_config,
    device_map="auto"
)

from pprint import pprint
pprint(xlstm.backbone.blocks[0].mlstm_layer.config)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Tokenize the input.
tokens = tokenizer("tell the difference between lstm and transformers.", return_tensors='pt')['input_ids']

# Move tokens to the same device as the model's first parameter (typically cuda:0).
first_device = next(xlstm.parameters()).device
tokens = tokens.to(first_device)

# Generate output using the distributed model.
# Measure time for generation
start_time = time.time()
out = xlstm.generate(tokens, max_new_tokens=2000)

# Decode and print the output.
print(tokenizer.decode(out[0]))

generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")