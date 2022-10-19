import onnxruntime
import cv2
import numpy as np
from scipy import special
import torch

image_ori = cv2.imread("workspace/self_driving/media/test_image_01.jpg")
width, height = image_ori.shape[1], image_ori.shape[0]

# 预处理
image = cv2.resize(image_ori, (800, 288))    # 1.resize
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # 2. BGR->RGB
image = (image/255.).astype(np.float32)    # 3. normalize
image = image.transpose(2,0,1)[None]    # 4. transpose


# 推理
onnx_file = "workspace/self_driving/model/lane_detection_288_800.onnx"
session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
predict = session.run(["output"], {"input.1": image})[0] # 获取第0个输出
predict = predict[0] # batch 0

import numpy as np
row_anchors = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])

# 映射到原图
for index, (x, loc) in enumerate(predict):
    if int(loc) ==200: continue
    x_ori = int(x * width/800)
    y_ori = int(row_anchors[index//4] * height/288)
    cv2.circle(image_ori, (x_ori,y_ori) ,5, (0,255,0) ,-1)

cv2.imwrite("workspace/self_driving/output/test_image_01_infer.jpg", image_ori)