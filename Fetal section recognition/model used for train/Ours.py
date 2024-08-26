import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from function import one_hot_encode, setup_seed
from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
setup_seed(4)
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
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=k, stride=k),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear((int(input_size/k))*36, 100),  # 更新线性层的输入大小
            nn.ReLU(),
            # nn.Linear(100, 60),
            # nn.ReLU(),
            nn.Linear(100, output_size),
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.flatten(1, -1)
        output = self.fc_layer(conv_out)
        return output
        
#得到数据
data_dir = 'D:\\Users\\ASUS\\Desktop\\SetA'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0], [1])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True)
               for x in ['train', 'val']}
# 粉笔计算训练与测试的样本数，字典格式
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}   # 训练与测试的样本数
# class_names = image_datasets['train'].classes           # 样本的类别，分别对应着蜜蜂和蚂蚁

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 定义两层的ResNet模型

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2, stride=5)
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


# # 创建模型实例
# model = ResNet(10).to(device)

# # # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# Max = 0
# for i in range(30):
#     model.train()
#     p = 0
#     sum_loss = torch.zeros((13))
#     for inputs, labels in dataloaders['train']:
#         labels = one_hot_encode(labels, 10)
#         inputs, labels = inputs.to(device), labels.to(device)
#         output, _ = model.forward(inputs)
#         loss = criterion(output, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         sum_loss[p] = loss
#         p = p + 1
#     print(torch.mean(sum_loss))
#     if i>=10:
#         model.eval()
#         all_test_target = []
#         all_test_output = []
#         m = 0
#         for inputs, labels in dataloaders['val']:
#             inputs, labels = inputs.to(device), labels.to(device)
#             all_test_target.append(labels)
#             output, _ = model(inputs)
#             predicted_class = torch.argmax(output, dim=1).to(device)
#             all_test_output.append(predicted_class)
#             m = m + 1
#         all_test_target = torch.cat(all_test_target)
#         all_test_output = torch.cat(all_test_output)
#         acu = torch.sum(all_test_output == all_test_target).item() / 580.0
#         acu_percent = acu * 100
#         print(f'Accuracy: {acu_percent:.2f}%')
#         if acu_percent >= Max:
#             torch.save(model, "the best")


# model = torch.load("the best")
# # #残差的解
# model.eval()
# all_test_target = []
# all_test_output = []
# m = 0
# for inputs, labels in dataloaders['val']:
#       inputs, labels = inputs.to(device), labels.to(device)
#       all_test_target.append(labels)
#       output,_ = model(inputs)
#       predicted_class = torch.argmax(output, dim = 1).to(device)
#       all_test_output.append(predicted_class)
#       m = m+1
# all_test_target =torch.cat(all_test_target)
# all_test_output = torch.cat(all_test_output)
# acu = torch.sum(all_test_output == all_test_target).item() / 580.0
# acu_percent = acu * 100
# print(f'Accuracy: {acu_percent:.2f}%')







model = torch.load("the best")
for para in model.parameters():
    para.requires_grad = False

setup_seed(0)

cnn = CNNModel(3200, 10, 50).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer1 = optim.Adam(cnn.parameters(), lr=0.001)


for i in range(10):
    p = 0
    sum_loss = torch.zeros((13))
    for inputs, labels in dataloaders['train']:  # inputs[1, 1, 200, 200]   labels[7]
        labels = one_hot_encode(labels, 10)
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

    if i>=1:
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
        acu = torch.sum(all_test_output == all_test_target).item() / 580.0
        acu_percent = acu * 100
        print(f'Accuracy: {acu_percent:.2f}%')



