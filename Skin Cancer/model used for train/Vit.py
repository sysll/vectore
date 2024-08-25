from function import model_eval, setup_seed, one_hot_encode
import timm
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import os

Max = 0.0

setup_seed(4)
#得到数据[200,200]的数据
data_dir = 'D:\\Users\\ASUS\\Desktop\\良性癌症等检测'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['train', 'val']}

# 粉笔计算训练与测试的样本数，字典格式
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}   # 训练与测试的样本数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 创建模型实例
model = timm.create_model('vit_base_patch16_224', num_classes=3).to(device)

# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i in range(16):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:
        labels = one_hot_encode(labels, 3)
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(f"第{i + 1}次的loss是：{torch.mean(sum_loss).item()}")

    if i >= 10:
        Max = model_eval(dataloaders, model, 'Vit.pth', Max)
