from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

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

# Tokenize the input and move it to the same device
tokens = tokenizer("Hello xLSTM, how are you doing?", return_tensors='pt')['input_ids'].to(device)

# Generate output
out = xlstm.generate(tokens, max_new_tokens=20)

# Decode and print the output
print(tokenizer.decode(out[0]))