# conda create -n qwen-eval-env python=3.10 -y
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate qwen-eval-env
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# conda install tqdm
# pip install pycocotools
# conda install transformers==4.57.0
# pip install qwen_vl_utils
# pip install accelerate
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl


#!/bin/bash
# Exit immediately if a command fails
set -e

api_key=$1

echo "=== Creating conda environment ==="
conda create -n qwen-eval-env python=3.10 -y

echo "=== Activating environment ==="
# Conda activate doesn't work directly in non-interactive shells unless you source it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-eval-env
echo "Active Conda environment: $CONDA_DEFAULT_ENV"

echo "=== Installing PyTorch and dependencies ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

echo "=== Installing tqdm ==="
conda install -y tqdm

echo "=== Installing pycocotools ==="
pip install pycocotools

echo "=== Installing transformers ==="
pip install transformers==4.57.0

echo "=== Installing qwen_vl_utils ==="
pip install qwen_vl_utils==0.0.8

echo "=== Installing accelerate ==="
pip install accelerate

echo "=== Installing FlashAttention ==="
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

echo "=== Installing opencv ==="
conda install -c conda-forge opencv -y

echo "=== Installing supervision ==="
conda install -c conda-forge supervision -y

echo "=== INstalling openai ==="
conda install openai -y

echo "===upgrading numpy==="
conda install -c conda-forge numpy=2.0.1 -y

echo "===creating env for FP8 ===="
conda deactivate
conda create -n qwen-fp8-env python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-fp8-env

echo "===installing vllm in fp8 env===="
pip install vllm

echo "=== Installing tqdm ==="
conda install -y tqdm

echo "=== Installing pycocotools ==="
pip install pycocotools

echo "=== Installing transformers ==="
pip install transformers==4.57.0

echo "=== Installing qwen_vl_utils ==="
pip install qwen_vl_utils==0.0.8

echo "=== Installing accelerate ==="
pip install accelerate

echo "=== Installing opencv ==="
conda install -c conda-forge opencv -y

echo "=== Installing supervision ==="
conda install -c conda-forge supervision -y

echo "===upgrading numpy==="
conda install -c conda-forge numpy=2.0.1 -y

echo "=== installing roboflow ==="
conda deactivate
conda create -n rf100vl-env python=3.9.23 -y
echo "=== Activating environment ==="
# Conda activate doesn't work directly in non-interactive shells unless you source it
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rf100vl-env
echo "Active Conda environment: $CONDA_DEFAULT_ENV"

pip install rf100vl==1.1.0

echo "=== installing pycocotools ==="
pip install pycocotools

echo "===exporting roboflow key==="
export ROBOFLOW_API_KEY=$api_key

echo "=== Downloading the data ==="
mkdir -p data
python download_data_roboflow.py --data_dir data/

echo "===making dir for result==="
mkdir -p results

echo "===script successful!!! ======"