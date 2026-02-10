import kagglehub
import hashlib
import base64
import dotenv
import os
import numpy as np
import contextlib
import torch


dotenv.load_dotenv()
# kaggle competitions list
print("hello world")
"""

python3 -c "import kagglehub; path = kagglehub.model_download('google/gemma-3/pyTorch/gemma-3-4b'); print(path)"

"""
path = kagglehub.model_download('google/gemma-3/pyTorch/gemma-3-4b')
print(path)
