import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import os
from function import one_hot_encode


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
setup_seed(4)  #


#得到数据
data_dir = 'D:/Users/ASUS/Desktop/百度下载位置/眼球的训练测试数据'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True)
               for x in ['train', 'val']}
# 粉笔计算训练与测试的样本数，字典格式
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}   # 训练与测试的样本数
# class_names = image_datasets['train'].classes           # 样本的类别，分别对应着蜜蜂和蚂蚁
def get_vector(matrix, n):
    sum = []
    for j in range(matrix.shape[0]):
        real_eigenvectors = []
        U, S, V = torch.svd(matrix[j]+1)
        for i in range(int(n)):
            real_eigenvectors.append(U[:, i] * S[i])
        part = torch.cat(real_eigenvectors)
        sum.append(part)
    sum = torch.cat(sum)
    return sum

def get_bat_v(matrix, n):
    sum = torch.zeros((matrix.shape[0], matrix.shape[2] * n * matrix.shape[1]))
    for j in range(matrix.shape[0]):
        v = get_vector(matrix[j], n)
        sum[j] = v
    return sum


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 定义两层的ResNet模型

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        key = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out, key


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += self.shortcut(residual)
        x = self.relu(x)

        return x


# # # 创建模型实例
# model = ResNet(3).to(device)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# for i in range(14):
#     model.train()
#     p = 0
#     sum_loss = torch.zeros((4))
#     for inputs, labels in dataloaders['train']:
#         labels = one_hot_encode(labels, 3)
#         inputs, labels = inputs.to(device), labels.to(device)
#         output, _ = model.forward(inputs)
#         loss = criterion(output, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         sum_loss[p] = loss
#         p = p + 1
#     print(torch.mean(sum_loss))
#     if i>=7:
#         model.eval()
#         all_test_target = []
#         all_test_output = []
#         m = 0
#         model.eval()
#         for inputs, labels in dataloaders['val']:
#             inputs, labels = inputs.to(device), labels.to(device)
#             all_test_target.append(labels)
#             output, _ = model(inputs)
#             predicted_class = torch.argmax(output, dim=1).to(device)
#             all_test_output.append(predicted_class)
#             m = m + 1
#         all_test_target = torch.cat(all_test_target)
#         all_test_output = torch.cat(all_test_output)
#         acu = torch.sum(all_test_output == all_test_target).item() / 210
#         acu_percent = acu * 100
#         print(f'Accuracy: {acu_percent:.2f}%')
#
#
# torch.save(model, "eye")


#74.29


# model = torch.load("eye")
#
# all_test_target = []
# all_test_output = []
# m = 0
# model.eval()
# for inputs, labels in dataloaders['val']:
#               inputs, labels = inputs.to(device), labels.to(device)
#               all_test_target.append(labels)
#               output,_ = model(inputs)
#               predicted_class = torch.argmax(output, dim = 1).to(device)
#               all_test_output.append(predicted_class)
#               m = m+1
# all_test_target =torch.cat(all_test_target)
# all_test_output = torch.cat(all_test_output)
# acu = torch.sum(all_test_output == all_test_target).item() / 210
# acu_percent = acu * 100
# print(f'Accuracy: {acu_percent:.2f}%')



# #开始训练自己的
"""以下训练是在上述的所有的模型都注释掉的情况下进行的"""
model = torch.load("eye")
for para in model.parameters():
    para.requires_grad = False

def get_vector(matrix, n):
    sum = []
    for j in range(matrix.shape[0]):
        real_eigenvectors = []
        U, S, V = torch.svd(matrix[j]+1)
        for i in range(int(n)):
            real_eigenvectors.append(U[:, i] * S[i])
        part = torch.cat(real_eigenvectors)
        sum.append(part)
    sum = torch.cat(sum)
    return sum




def get_bat_v(matrix, n):
    sum = torch.zeros((matrix.shape[0], matrix.shape[2] * n * matrix.shape[1]))
    for j in range(matrix.shape[0]):
        v = get_vector(matrix[j], n)
        sum[j] = v
    return sum

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, k):     #k是特征向量大小
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            #
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=k, stride=k),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear((int(input_size/k))*64, 60),  # 更新线性层的输入大小
            nn.ReLU(),
            nn.Linear(60, output_size),
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.flatten(1, -1)
        output = self.fc_layer(conv_out)
        return output
cnn = CNNModel(3200, 3, 50).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer1 = optim.Adam(cnn.parameters(), lr=0.001)

model.train()
for i in range(11):
    p = 0
    sum_loss = torch.zeros((30))
    model.train()
    cnn.train()
    for inputs, labels in dataloaders['train']:  # inputs[1, 1, 200, 200]   labels[7]
        labels = one_hot_encode(labels, 3)
        inputs, labels = inputs.to(device), labels.to(device)
        _, key = model.forward(inputs)
        v = get_bat_v(key, 5).to(device)
        output = cnn(v.unsqueeze(1))
        loss = criterion(output, labels)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss))


    if i>=7:    #注意，如果把7改了结果也会改变。因为model.eval会在一开始就对训练产生影响
        with torch.no_grad():
            model.eval()
            cnn.eval()
            #残差的解
            all_test_target = []
            all_test_output = []
            m = 0
            for inputs, labels in dataloaders['val']:
                  inputs, labels = inputs.to(device), labels.to(device)
                  all_test_target.append(labels)

                  _, key = model.forward(inputs)
                  v = get_bat_v(key, 5).to(device)
                  output = cnn(v.unsqueeze(1))

                  predicted_class = torch.argmax(output, dim = 1).to(device)
                  all_test_output.append(predicted_class)
                  m = m+1
            all_test_target =torch.cat(all_test_target)
            all_test_output = torch.cat(all_test_output)
            acu = torch.sum(all_test_output == all_test_target).item() / 210.0
            acu_percent = acu * 100
            print(f'Accuracy: {acu_percent:.2f}%')
