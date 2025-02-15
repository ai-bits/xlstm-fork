#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #??
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the local directory where the model files are stored
model_directory = "/home/gy/dl/xLSTM-7b" #"NX-AI/xLSTM-7b" #HF Hub path

# Set the device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
xlstm = AutoModelForCausalLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Wrap the model with DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    xlstm = torch.nn.DataParallel(xlstm)

xlstm = xlstm.to(device)

# Tokenize the input and move it to the same device
tokens = tokenizer("Hello xLSTM, how are you doing?", return_tensors='pt')['input_ids'].to(device)

# Generate output
out = xlstm.module.generate(tokens, max_new_tokens=20)

# Decode and print the output
print(tokenizer.decode(out[0]))