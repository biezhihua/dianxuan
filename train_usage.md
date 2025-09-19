# 训练脚本
 
## 命令行参数

- `--train_dir`: 训练数据集目录路径，直接包含各个类别子文件夹（默认值：`E:\Projects\github\soda_mhxy\py_32\others\chengyu_classify_final`）
- `--val_dir`: 验证数据集目录路径，如果不指定则使用train_dir（默认值：'.val'）
- `--epochs`: 训练轮数（默认值：200）
- `--lr`: 学习率（默认值：0.0001）
- `--batch_size_train`: 训练批次大小（默认值：20）
- `--batch_size_val`: 验证批次大小（默认值：10）
- `--auto_export`: 训练完成后自动导出为ONNX格式（默认值：False）
- `--onnx_path`: ONNX文件输出路径，如果不指定则自动生成时间戳文件名（默认值：None）

## 使用示例

### 1. 使用默认参数
```bash
python train.py
```

### 2. 指定自定义训练目录
```bash
python train.py --train_dir "D:\my_train_dataset"
```

### 3. 指定不同的训练和验证目录
```bash
python train.py --train_dir "D:\my_train_dataset" --val_dir "D:\my_val_dataset"
```

### 4. 自定义多个参数
```bash
python train.py --train_dir "D:\my_train_dataset" --epochs 100 --lr 0.001 --batch_size_train 32 --batch_size_val 16
```

### 5. 训练完成后自动导出ONNX模型
```bash
python train.py --auto_export
```

### 6. 指定ONNX输出路径
```bash
python train.py --auto_export --onnx_path "my_siamese_model.onnx"
```

### 7. 完整的训练和导出命令
```bash
python train.py --train_dir "D:\my_train_dataset" --epochs 50 --auto_export --onnx_path "trained_model.onnx"
```

### 4. 查看帮助信息
```bash
python train.py --help
```

## 数据目录结构
确保你的数据目录结构如下：

### 训练目录
```
your_train_dir/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 验证目录（可选，如果不指定则使用训练目录）
```
your_val_dir/
├── class1/
│   ├── image1.jpg
│   └── ...
├── class2/
│   └── ...
└── ...
```

## 注意事项
- 脚本会自动检查指定的训练和验证目录是否存在
- 如果不指定验证目录，会使用训练目录作为验证目录
- 训练目录应直接包含各个类别的子文件夹，而不需要额外的"train"子目录
- 如果目录不存在，脚本会报错并退出
- 模型会自动保存为 `model.pth`，当验证损失降低时会更新保存