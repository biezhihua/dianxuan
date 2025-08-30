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


# # 设置设备
# out_onnx = 'model.onnx'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 创建虚拟输入
# dummy = (torch.randn(1, 3, 105, 105).to(device), torch.randn(1, 3, 105, 105).to(device))

# # 加载模型
# model = torch.load(r'E:\Projects\dianxuan\bj.pth', weights_only=False)
# model.eval()
# model = model.to(device)

# # 导出为 ONNX 格式
# torch.onnx.export(
#     model, 
#     dummy, 
#     out_onnx,
#     input_names=["x1", "x2"],
#     output_names=["output1", "output2"],
#     dynamic_axes={'x1': {0: 'batch_size'}, 'x2': {0: 'batch_size'}, 
#                   'output1': {0: 'batch_size'}, 'output2': {0: 'batch_size'}}
# )

# print("ONNX 导出完成！")