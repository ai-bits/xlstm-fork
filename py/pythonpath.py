#PYTHONPATH=/home/gy/dl/xlstm-fork/xlstm/xlstm_large python /home/gy/dl/xlstm-fork/py/gradio-ui.py #bash
import sys
import os

# Manually set PYTHONPATH within Python
os.environ["PYTHONPATH"] = "/home/gy/dl/xlstm-fork/xlstm/xlstm_large"
sys.path.append("/home/gy/dl/xlstm-fork/xlstm/xlstm_large")

# Now, try to import and execute the script
import subprocess
subprocess.run(["python", "/home/gy/dl/xlstm-fork/py/gradio-ui.py"])
