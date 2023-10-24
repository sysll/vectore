import timm
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    torch.use_deterministic_algorithms(True)
setup_seed(4)


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
                  for x in ['val']}

# 分别对训练和验证样本集构建样本加载器，使用适当的batch_size
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['val']}



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载模型
model = timm.create_model('efficientnet_b0', num_classes=3).to(device)
model.load_state_dict(torch.load('./best models/Efficientnet.pth'))

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
acu = torch.sum(all_test_output == all_test_target).item() / 210.0
acu_percent = acu * 100
print(f'Accuracy: {acu_percent:.2f}%')