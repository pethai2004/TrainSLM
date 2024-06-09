#!/bin/bash

VENV_DIR=train_env
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

pip install -r requirements.txt

export PYTHONPATH=$(pwd)

echo "PYTHONPATH is set to: $PYTHONPATH"

