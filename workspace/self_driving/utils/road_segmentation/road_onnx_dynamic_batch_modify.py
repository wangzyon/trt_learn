import onnx

onnx_file = "workspace/self_driving/model/road_segmentation_512x896_ori.onnx"
road_model = onnx.load(onnx_file)

# 修改batch维度为动态，名称要与原始onnx名称对应上
road_model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("data", 1, ["batch", 3, 512, 896]))
road_model.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("tf.identity", 1, ["batch", 512, 896,4]))



        


# 删除模型中所有node结点的维度信息
while len(road_model.graph.value_info) > 0:
    road_model.graph.value_info.pop()
    
onnx.save(road_model, "workspace/self_driving/model/road_segmentation_512x896.onnx")