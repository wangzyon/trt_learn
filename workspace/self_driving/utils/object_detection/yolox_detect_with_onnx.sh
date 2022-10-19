cd YOLOX-0.2.0
export PYTHONPATH=$PYTHONPATH:.

python ./demo/ONNXRuntime/onnx_inference.py \
-m yolox_nano_v1.onnx -i test_image_06.jpg -o ./output --input_shape "416,416"