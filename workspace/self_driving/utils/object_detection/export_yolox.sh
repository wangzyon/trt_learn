#!/bin/bash

cd YOLOX-0.2.0
export PYTHONPATH=$PYTHONPATH:.

python tools/export_onnx.py -c yolox_tiny.pth -f exps/default/yolox_tiny.py --output-name=yolox_tiny_v1.onnx --dynamic --no-onnxsim
