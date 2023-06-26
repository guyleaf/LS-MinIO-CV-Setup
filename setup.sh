#!/bin/bash

pip install -U pip

# change url according to your cuda
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
