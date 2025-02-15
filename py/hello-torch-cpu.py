from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time

# Specify the local directory where the model files are stored
model_directory = "/home/gy/dl/xLSTM-7b" #"NX-AI/xLSTM-7b" #HF Hub path

# Set the device to cpu
device = torch.device("cpu")

xlstm_config = AutoConfig.from_pretrained(model_directory)
xlstm_config.step_kernel = "native"
xlstm_config.chunkwise_kernel = "chunkwise--native_autograd"
xlstm_config.sequence_kernel = "native_sequence__native"

# Load the model and tokenizer on the CPU
xlstm = AutoModelForCausalLM.from_pretrained(model_directory, config=xlstm_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# verify selected kernels
from pprint import pprint
pprint(xlstm.backbone.blocks[0].mlstm_layer.config)

# Measure time for tokenization
start_time = time.time()
tokens = tokenizer("tell the difference between lstm and transformers.", return_tensors='pt')['input_ids'].to(device)
tokenization_time = time.time() - start_time
print(f"Tokenization time: {tokenization_time:.2f} seconds")

# Measure time for generation
start_time = time.time()
out = xlstm.generate(tokens, max_new_tokens=2000)

# Decode and print the output
print(tokenizer.decode(out[0]))

generation_time = time.time() - start_time
print(f"Generation time: {generation_time:.2f} seconds")