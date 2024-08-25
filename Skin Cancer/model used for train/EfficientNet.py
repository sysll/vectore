import timm
from function import model_eval,  one_hot_encode
import torch.optim as optim
import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.transforms as transforms
Max = 0.0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
setup_seed(4)  #


#得到数据[200,200]的数据
data_dir = 'D:\\Users\\ASUS\\Desktop\\良性癌症等检测'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val']}




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# # 创建模型实例
model = timm.create_model('efficientnet_b0', num_classes=2).to(device)

# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in range(16):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:  # inputs[1, 1, 200, 200]   labels[7]
        labels = one_hot_encode(labels, 2)
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
        Max = model_eval(dataloaders, model, 'Efficientnet.pth', Max)
