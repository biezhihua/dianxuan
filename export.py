import torch
from train import SiameseNetwork

# 获取当前格式化的日期时间

def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

out_onnx = f'model_{get_current_time()}.onnx'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy = (torch.randn(1, 3, 105, 105).to(device), torch.randn(1, 3, 105, 105).to(device))
model = torch.load(r'E:\Projects\dianxuan\model.pth', weights_only=False)
model.eval()

model = model.to(device)
torch_out = torch.onnx.export(model, dummy, out_onnx,input_names=["x1", "x2"])
print("finish!")