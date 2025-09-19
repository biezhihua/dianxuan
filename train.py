
# 导入必要的库
import torch.nn as nn
import torch
import torch.utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import random
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models


# 设置设备为GPU或CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device_type = "cuda"
    print("正在使用gpu训练")
else:
    device_type = "cpu"
    print("正在使用cpu训练")
device = torch.device(device_type)


# 加载预训练的VGG16模型，并去除最后的池化和分类层，仅保留特征提取部分
mymod = models.vgg16(pretrained=True)
del mymod.avgpool
del mymod.classifier



def get_class_image_paths(path):
    """
    获取每个类别下所有图片的路径。
    参数:
        path: 数据集根目录
    返回:
        dict: {类别名: [图片路径列表]}
    """
    data = {}
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        data[class_name] = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    return data



def get_random_image_path(class_name, class_image_dict, same_class=0):
    """
    随机获取同类或异类图片路径。
    参数:
        class_name: 当前类别名
        class_image_dict: {类别名: [图片路径列表]}
        same_class: 是否同类（1为同类，0为异类）
    返回:
        图片路径
    """
    keys = list(class_image_dict.keys())
    other_keys = keys.copy()
    other_keys.remove(class_name)
    if same_class == 1:
        target_key = class_name
    else:
        target_key = random.choice(other_keys)
    target_list = class_image_dict[target_key]
    if len(target_list) == 0:
        print(f"error: 类别 {target_key} 没有图片")
    return random.choice(target_list)



def generate_siamese_pairs(class_image_dict):
    """
    生成孪生网络训练对，每个样本包含同类和异类图片对。
    参数:
        class_image_dict: {类别名: [图片路径列表]}
    返回:
        list: [[[img1, img2, label], [img3, img4, label]], ...]
    """
    all_pairs = []
    for class_name in class_image_dict:
        for img_path in class_image_dict[class_name]:
            # 异类对 label=0，同类对 label=1
            pair_diff = [img_path, get_random_image_path(class_name, class_image_dict, 0), 0]
            pair_same = [img_path, get_random_image_path(class_name, class_image_dict, 1), 1]
            ku = [pair_diff, pair_same]
            random.shuffle(ku)
            all_pairs.append(ku)
    return all_pairs



class SiameseNetwork(nn.Module):
    """
    孪生神经网络模型，输入两张图片，输出相似度。
    """
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = mymod.features
        self.feature_extractor = self.feature_extractor.eval()
        self.feature_extractor.to(device)
        flat_shape = 512 * 3 * 3
        self.fc1 = torch.nn.Linear(flat_shape, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # 提取特征
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        # 展平
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # 计算特征差异
        x = torch.abs(x1 - x2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x



class SiameseDataset(Dataset):
    """
    孪生网络数据集，返回两组图片对及其标签。
    """
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs
        self.length = len(self.pairs)
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        pair1, pair2 = self.pairs[idx]
        img1 = self.transform(Image.open(pair1[0]))
        img2 = self.transform(Image.open(pair1[1]))
        img3 = self.transform(Image.open(pair2[0]))
        img4 = self.transform(Image.open(pair2[1]))
        img1 = img1.to(device).unsqueeze(0)
        img2 = img2.to(device).unsqueeze(0)
        img3 = img3.to(device).unsqueeze(0)
        img4 = img4.to(device).unsqueeze(0)
        return (
            torch.concat([img1, img3], dim=0),
            torch.concat([img2, img4], dim=0),
            torch.tensor([pair1[2], pair2[2]], dtype=torch.float).to(device),
        )

    def __len__(self):
        return self.length



def train_one_epoch():
    """
    训练一个epoch，返回平均损失。
    """
    mymox.train()
    total_loss = 0
    total_acc = 0
    batch_count = 0
    progress_bar = tqdm(traf)
    for k, x, t in progress_bar:
        # 展平batch维度
        k = k.view(k.shape[0] * k.shape[1], k.shape[2], k.shape[3], k.shape[4])
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        t = t.view(t.shape[0] * t.shape[1], 1)
        Adme.zero_grad()
        out = mymox(k, x)
        loss = myloss(out, t)
        loss.backward()
        Adme.step()
        total_loss += loss.item()
        with torch.no_grad():
            equal = torch.eq(torch.round(out), t)
            acc = torch.mean(equal.float())
        total_acc += acc.item()
        batch_count += 1
        progress_bar.set_description(desc=f"loss [{total_loss / batch_count:.4f}] acc [{total_acc / batch_count:.4f}]")
    return total_loss / batch_count



def get_train_loader():
    """
    获取训练集的DataLoader。
    """
    target_path = "./train"
    target_path = "E:\Projects\github\soda_mhxy\py_32\others\chengyu_classify_final"
    class_image_dict = get_class_image_paths(target_path)
    pairs = generate_siamese_pairs(class_image_dict)
    dataset = SiameseDataset(pairs)
    loader = DataLoader(dataset, shuffle=True, batch_size=20)
    return loader



def get_val_loader():
    """
    获取验证集的DataLoader。
    """
    class_image_dict = get_class_image_paths("./val")
    pairs = generate_siamese_pairs(class_image_dict)
    dataset = SiameseDataset(pairs)
    loader = DataLoader(dataset, shuffle=True, batch_size=10)
    return loader



def validate():
    """
    验证模型在验证集上的表现，输出平均损失和准确率。
    """
    mymox.eval()
    total_loss = 0
    total_acc = 0
    batch_count = 0
    progress_bar = tqdm(texf)
    with torch.no_grad():
        for k, x, t in progress_bar:
            k = k.view(k.shape[0] * k.shape[1], k.shape[2], k.shape[3], k.shape[4])
            x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            t = t.view(t.shape[0] * t.shape[1], 1)
            out = mymox(k, x)
            loss = myloss(out, t)
            total_loss += loss.item()
            batch_count += 1
            equal = torch.eq(torch.round(out), t)
            acc = torch.mean(equal.float())
            total_acc += acc.item()
            progress_bar.set_description(desc=f"loss [{total_loss / batch_count:.4f}] acc [{total_acc / batch_count:.4f}]")



if __name__ == "__main__":
    # 实例化孪生网络模型
    mymox = SiameseNetwork()  # 重新训练
    # mymox = torch.load('./bj.pth') # 迁移学习
    epoch = 200  # 训练轮数
    mymox.to(device)
    Adme = optim.Adam(mymox.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(Adme, step_size=5, gamma=0.1)
    myloss = nn.BCELoss()
    best_loss = float('inf')
    for i in range(epoch):
        print("epoch", i + 1)
        traf = get_train_loader()
        texf = get_val_loader()
        avg_loss = train_one_epoch()
        validate()
        scheduler.step()
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(mymox, "model.pth")
            print("save model ===> model.pth")
