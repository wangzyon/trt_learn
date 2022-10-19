import torch

class PostProcessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size, cell_num, row_anchor_num, lane_num = map(int, x.shape) # [batch_size, 201, 18, 4]
        batch_size= -1
        # cell位置概率
        cell_prob = self.softmax(x[:,:cell_num-1])
        # cell位置
        cell_anchors = torch.arange(1, 201, device=x.device).reshape(1,cell_num-1,1,1)
        # x预测值， (800-1)/ (cell_num-1)为每个cell宽度
        cell_width = (800.0-1)/ (cell_num-1)
        x_pre = torch.sum(cell_prob*cell_anchors, dim=1)* cell_width  # [batch_size, 18, 4], cell间隔数量为199
        # 最大位置概率索引
        loc_pre = torch.argmax(x,dim=1).to(x.dtype) # [batch_size, 18, 4]
        predict = torch.stack([x_pre, loc_pre], dim=3).reshape(batch_size, row_anchor_num*lane_num,2) # [batch_size, 72, 2]
        return predict

postprocess_input = torch.zeros((1,201,18,4))
model = PostProcessModel()

torch.onnx.export(model, 
                  (postprocess_input,), 
                  "workspace/self_driving/model/lane_postprocess.onnx", 
                  verbose=True, 
                  input_names=["input"], 
                  output_names=["output"],
                  opset_version=11,
                  dynamic_axes={'input':{0:'batch'},
                                'output':{0:'batch'}})
