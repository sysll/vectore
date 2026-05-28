import torch
import os
import torchvision.models as models
def model_eval(dataloaders, model, name, Max = 0.0):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_test_target = []
    all_test_output = []
    m = 0
    for inputs, labels in dataloaders['val']:
          inputs, labels = inputs.to(device), labels.to(device)
          all_test_target.append(labels)
          output = model(inputs)
          predicted_class = torch.argmax(output, dim = 1).to(device)
          all_test_output.append(predicted_class)
          m = m+1
    all_test_target =torch.cat(all_test_target)
    all_test_output = torch.cat(all_test_output)
    acu = torch.sum(all_test_output == all_test_target).item() / 210.0
    acu_percent = acu * 100
    print(f'Accuracy: {acu_percent:.2f}%')
    if acu_percent > Max:
        torch.save(model.state_dict(), os.path.join('python', name))
        Max = acu_percent
        print("保存完毕，准确概是"+str(Max))
    return Max




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    # torch.use_deterministic_algorithms(True)



def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

import timm
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import os
setup_seed(4)

#得到数据

data_dir = 'eye'   # 样本地址

transform1 = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(300),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val']}


Max =0.0
setup_seed(4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 3)  # 将最后的全连接层设置为3分类
model.aux_logits = False  # 禁用辅助分类器
model = model.to(device)
# # 定义损失函数和优化器

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i in range(40):
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

    if i >= 0:
        Max = model_eval(dataloaders, model, 'Vgg16.pth', Max)