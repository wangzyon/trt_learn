import onnx
import onnx.helper

ldrn_onnx = onnx.load("workspace/self_driving/model/ldrn_256x512_ori.onnx")

# 输入输出batch维度设置为动态
ldrn_onnx.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("input.1", 1, ["batch", 3,256,512]))
ldrn_onnx.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("2499", 1, ["batch", 1, 256,512]))
while(len(ldrn_onnx.graph.output)>1):
    ldrn_onnx.graph.output.pop()


# 除了输入输出外，删除onnx中所有的维度信息，用于支持动态batch
while len(ldrn_onnx.graph.value_info) > 0:
    ldrn_onnx.graph.value_info.pop()
    
onnx.save(ldrn_onnx, "workspace/self_driving/model/ldrn_256x512.onnx")