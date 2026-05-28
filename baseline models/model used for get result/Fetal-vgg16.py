#Accuracy: 81.90%
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
import os
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
setup_seed(4)


import torchvision.transforms as transforms
import torchvision.datasets as datasets
#得到数据
data_dir = 'E:\特征向量三数据集\胎儿切片判断'   # 样本地址
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 因为只有一个通道，所以只需要一个均值和一个标准差
])

# 构建训练和验证的样本数据集，应用transform
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform1)
                  for x in ['train', 'val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['train', 'val']}



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载模型
# 加载预训练的Vgg16模型
model = models.vgg16().to(device)
model.classifier[6] = torch.nn.Linear(4096, 10)  # 将最后的全连接层设置为3分类
model.aux_logits = False  # 禁用辅助分类器
model = model.to(device)

model.load_state_dict(torch.load('./best models/Vgg16.pth'))

model.eval()

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
acu = torch.sum(all_test_output == all_test_target).item() / 580.0
acu_percent = acu * 100
print(f'Accuracy: {acu_percent:.2f}%')