import timm
from function import model_eval,  one_hot_encode
import torch.optim as optim
import os
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
Max = 0
#得到数据
data_dir = 'D:\\Users\\ASUS\\Desktop\\SetA'   # 样本地址
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



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
setup_seed(0)  #


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 创建模型实例
model = timm.create_model('efficientnet_b0', num_classes=10).to(device)

# # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)
ev = []
for i in range(90):
    model.train()
    p = 0
    sum_loss = torch.zeros((100))
    for inputs, labels in dataloaders['train']:
        labels = one_hot_encode(labels, 10)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss))

    if i>=50:
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        all_test_target = []
        all_test_output = []
        m = 0
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            all_test_target.append(labels)
            output = model(inputs)
            predicted_class = torch.argmax(output, dim=1).to(device)
            all_test_output.append(predicted_class)
            m = m + 1
        all_test_target = torch.cat(all_test_target)
        all_test_output = torch.cat(all_test_output)
        acu = torch.sum(all_test_output == all_test_target).item() / 580.0
        acu_percent = acu * 100
        print(f'Accuracy: {acu_percent:.2f}%')
        print(Max)
        if acu_percent > Max:
            torch.save(model.state_dict(), os.path.join('../model used for get result/best models', 'Efficienntnet.pth'))
            Max = acu_percent

