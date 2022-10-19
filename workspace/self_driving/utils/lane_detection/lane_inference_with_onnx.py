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
onnx_file = "workspace/self_driving/model/ultra_fast_lane_detection_culane_288x800.onnx"
session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
model_output = session.run(["200"], {"input.1": image})[0] # 获取第0个输出


class PostProcessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size, cell_num, row_anchor_num, lane_num = map(int, x.shape) # [batch_size, 201, 18, 4]
        batch_size = -1
        # cell位置概率
        cell_prob = self.softmax(x[:,:cell_num-1])
        # cell位置
        cell_anchors = torch.arange(1, 201, device=x.device).reshape(1,cell_num-1,1,1)
        # x预测值， (800-1)/ (cell_num-1)为每个cell宽度
        x_pre = torch.sum(cell_prob*cell_anchors, dim=1)* (800-1)/ (cell_num-1)  # [batch_size, 18, 4], cell间隔数量为199
        # 最大位置概率索引
        loc_pre = torch.argmax(x,dim=1) # [batch_size, 18, 4]
        predict = torch.stack([x_pre, loc_pre], dim=3).reshape(batch_size, row_anchor_num*lane_num,2) # [batch_size, 72, 2]
        return predict

model = PostProcessModel()
predict = model(torch.tensor(model_output))
predict = predict.cpu().numpy()[0]

row_anchors = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
# 映射到原图
for index, (x, loc) in enumerate(predict):
    if int(loc) ==200: continue
    x_ori = int(x * width/800)
    y_ori = int(row_anchors[index//4] * height/288)
    cv2.circle(image_ori, (x_ori,y_ori) ,5, (0,255,0) ,-1)

cv2.imwrite("workspace/self_driving/output/test_image_01_infer.jpg", image_ori)