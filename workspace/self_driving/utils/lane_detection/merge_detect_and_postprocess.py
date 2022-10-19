import onnx
import onnx.helper

detect_onnx = onnx.load("workspace/self_driving/model/ultra_fast_lane_detection_culane_288x800.onnx")
postprocess_onnx = onnx.load("workspace/self_driving/model/lane_postprocess.onnx")
# 将postprocess_onnx中所有结点重命名，防止与detect_onnx重名
for n in postprocess_onnx.graph.node:
    n.name = "post_" + n.name

    for i, v in enumerate(n.input):
        if v == "input":
            n.input[i] = "200"
        else:
            n.input[i] = "post_" + v

    for i, v in enumerate(n.output):
        if v != "output":
            n.output[i] = "post_" + v

detect_onnx.graph.node.extend(postprocess_onnx.graph.node)


# 拷贝postprocess的initializer至detect_onnx
for weight in postprocess_onnx.graph.initializer:
    weight.name = f"post_{weight.name}"
detect_onnx.graph.initializer.extend(postprocess_onnx.graph.initializer)

# 删除detect_onnx所有输出结点
while len(detect_onnx.graph.output) > 0:
    detect_onnx.graph.output.pop()

# 添加postprocess_onnx输出结点
detect_onnx.graph.output.extend(postprocess_onnx.graph.output)

# 输入输出batch维度设置为动态
detect_onnx.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("input.1", 1, ["batch", 3, 288, 800]))
detect_onnx.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("output", 1, ["batch", 72, 2]))

# 除了输入输出外，删除onnx中所有的维度信息，用于支持动态batch
while len(detect_onnx.graph.value_info) > 0:
    detect_onnx.graph.value_info.pop()
    
onnx.save(detect_onnx, "workspace/self_driving/model/lane_detection_288_800.onnx")