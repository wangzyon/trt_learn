import onnxruntime
import cv2
import numpy as np

# 1. 打印onnxruntime provider
providers = onnxruntime.get_available_providers()
print(f"providers: {providers}")

# 2. 从onnx加载模型
onnx_file = "workspace/self_driving/model/road_segmentation_512x896.onnx"
session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

# 3. 配置输入
image = cv2.imread("workspace/self_driving/media/test_image_01.jpg")
image_tensor = cv2.resize(image, (896, 512)).astype(np.float32)
image_tensor = image_tensor.transpose(2,0,1)[None]

# 4. 推理预测
prob = session.run(["tf.identity"], {"data":image_tensor})[0][0] # batch[0]

# 5. 保存输出
output_image = np.zeros((512, 896, 3), dtype=np.float32)
output_image[:,:,0] = prob[:,:,0]*70 + prob[:,:,1]*255
output_image[:,:,1] = prob[:,:,0]*70 + prob[:,:,2]*255
output_image[:,:,2] = prob[:,:,0]*70 + prob[:,:,3]*255

cv2.imwrite("workspace/self_driving/output/road_segmentation.jpg", output_image.astype(np.uint))
# cv2.imwrite("workspace/self_driving/output/no_drive_area.jpg", prob[0,:,:,0]*255)
# cv2.imwrite("workspace/self_driving/output/drive_area.jpg", prob[0,:,:,1]*255)
# cv2.imwrite("workspace/self_driving/output/curb.jpg", prob[0,:,:,2]*255)
# cv2.imwrite("workspace/self_driving/output/lane_line.jpg", prob[0,:,:,3]*255)


