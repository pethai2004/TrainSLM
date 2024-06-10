#!/bin/bash

VENV_DIR=train_env_0
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

export PYTHONPATH=$(pwd)
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
echo "PYTHONPATH is set to: $PYTHONPATH"

