import torch
from train import Siamese

out_onnx = 'model.onnx'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy = (torch.randn(1, 3, 105, 105).to(device), torch.randn(1, 3, 105, 105).to(device))
model = torch.load(r'E:\Projects\dianxuan\bj.pth', weights_only=False)
model.eval()

model = model.to(device)
torch_out = torch.onnx.export(model, dummy, out_onnx,input_names=["x1", "x2"])
print("finish!")