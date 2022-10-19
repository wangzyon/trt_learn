'''
导出mmdetection onnx思路：（或者任意框架）
1. 跑通官方推理代码，等够调试官方推理代码，不能调试则无法分析；
2. 找到模型核心部分，即forward涉及的关键组件，例如backbone， neck, head等，其他部分丢弃；
3. 重新组合Model(nn.Module)封装核心forward组件；
4. 导出onnx观察，若有多个输出，在forward中补充代码，调整模型输出；
mmdetection官方onnx导出不够灵活，调整onnx很容易报错，错误解决需要充分阅读理解框架算法源码实现；
'''
import sys
sys.path.append("/volume/huaru/third_party/mmdetection")  # 配置环境变量

import torch
from mmdet.apis import init_detector, inference_detector

config_file = '/volume/wzy/project/PyInfer/tools/mmdet_export_onnx/balloon/yolox_s_8x8_300e_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
checkpoint_file = '/volume/wzy/project/PyInfer/tools/mmdet_export_onnx/balloon/model.pth'
device = 'cuda:0'

#初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# print(inference_detector(model, 'demo/demo.jpg'))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config_file, checkpoint_file, device=device)

    def forward(self, x):
        ib, ic, ih, iw = map(int, x.shape)
        x = self.model.backbone(x)
        x = self.model.neck(x)
        clas, bbox, objness = self.model.bbox_head(x)

        # 网络输出映射到网络输入，heatmap特征层bbox映射到输入图像bbox
        # 1.映射bbox、2.decode(逆仿射变换+confidence过滤)、3.nms，其中2,3属于后处理，映射放在onnx中是为了简化后处理， 后处理不放在onnx中是为单独编写cuda提高效率
        output_x = []
        for class_item, bbox_item, objness_item in zip(clas, bbox, objness):
            hm_b, hm_c, hm_h, hm_w = map(int, class_item.shape)
            stride_h, stride_w = ih / hm_h, iw / hm_w
            strides = torch.tensor([stride_w, stride_h], device=device).view(-1, 1, 2)

            prior_y, prior_x = torch.meshgrid(torch.arange(hm_h), torch.arange(hm_w))
            prior_x = prior_x.reshape(hm_h * hm_w, 1).to(device)
            prior_y = prior_y.reshape(hm_h * hm_w, 1).to(device)
            prior_xy = torch.cat([prior_x, prior_y], dim=-1)
            class_item = class_item.permute(0, 2, 3, 1).reshape(-1, hm_h * hm_w, hm_c)
            bbox_item  = bbox_item.permute(0, 2, 3, 1).reshape(-1, hm_h * hm_w, 4)
            objness_item = objness_item.reshape(-1, hm_h * hm_w, 1)
            pred_xy = (bbox_item[..., :2] + prior_xy) * strides
            pred_wh = bbox_item[..., 2:4].exp() * strides
            pred_class = torch.cat([objness_item, class_item], dim=-1).sigmoid()
            output_x.append(torch.cat([pred_xy, pred_wh, pred_class], dim=-1))

        return torch.cat(output_x, dim=1)

m = Model().eval()

image = torch.zeros(1, 3, 640, 640, device=device)
torch.onnx.export(
    m, (image,), "yolox.onnx",
    opset_version=11, 
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},
        "output": {0: "batch"}
    }
)
print("Done.!")