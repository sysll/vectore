from function import model_eval, dataloaders, setup_seed, one_hot_encode
import timm
import torch.optim as optim
import torch
import torch.nn as nn
setup_seed(4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# # 创建模型实例
model = timm.create_model('swin_base_patch4_window7_224', num_classes=3).to(device)

# # 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i in range(30):
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
    print(torch.mean(sum_loss))

    if i>=0:
        model_eval(dataloaders, model)