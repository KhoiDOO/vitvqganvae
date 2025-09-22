#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create --prefix ./envs/$CONDA_ENV python=3.10.0 -y
    conda activate ./envs/$CONDA_ENV

    conda install -c nvidia cuda-toolkit=12.9 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# update pip to latest version for pyproject.toml setup.
pip install -U pip wheel

# install Pytorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# install dev dependencies
pip install pytest matplotlib

# install
pip install -e .