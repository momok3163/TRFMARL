#!/bin/bash
# Install PyTorch and Python Packages

conda create -n trfmarl python=3.8 -y
conda activate trfmarl
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install sacred numpy==1.22.0 scipy gym==0.10.8 matplotlib seaborn \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger
pip install git+https://github.com/oxwhirl/smac.git
pip install setuptools==65.5.0 "wheel<0.40.0"

