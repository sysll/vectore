import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn as nn
from function import model_eval, setup_seed, one_hot_encode
import torch.optim as optim
Max = 0.0
setup_seed(4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Inception模型
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(1280, 3)  # 将最后的全连接层设置为3分类
model.aux_logits = False  # 禁用辅助分类器
model = model.to(device)

#得到数据
data_dir = 'D:/Users/ASUS/Desktop/百度下载位置/眼球的训练测试数据'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(300),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'val']}



# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i in range(60):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:
        labels = one_hot_encode(labels, 3)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(f"第{i + 1}次的loss是：{torch.mean(sum_loss).item()}")

    if i >= 10:
        Max = model_eval(dataloaders, model, 'Mobilenet.pth', Max)
