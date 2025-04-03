#!/usr/bin/env bash

CWD=$(dirname "$0")
echo "Current working directory: $CWD"
echo "Activating virtual environment..."

pushd $CWD/..
source ve/bin/activate

export PYTHONPATH="${PYTHONPATH}:./"

python trading/strategies/triple_momentum.py


popd