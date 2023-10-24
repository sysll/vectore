import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.nn as nn
from function import model_eval, setup_seed, one_hot_encode
import torch.optim as optim

setup_seed(4)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载预训练的Inception模型
model = models.inception_v3(weights=None)
model.fc = torch.nn.Linear(2048, 3)  # 将最后的全连接层设置为3分类
model.aux_logits = False  # 禁用辅助分类器
model = model.to(device)

#得到数据
data_dir = 'D:/Users/ASUS/Desktop/百度下载位置/眼球的训练测试数据'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val']}




# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

for i in range(16):
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
    print(torch.mean(sum_loss))

    if i>=0:
        model_eval(dataloaders, model)
