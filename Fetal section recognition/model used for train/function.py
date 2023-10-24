from torchvision import datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

#把图片变成one-hot编码
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot



class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv_layer1 = nn.Sequential(
            #
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=100, stride=50),  # 将 kernel_size 改为 100
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1600, 400),  # 更新线性层的输入大小
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(400, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, channels, sequence_length)
        conv_out1 = self.conv_layer1(x)
        conv_out1 = conv_out1+x
        conv_out = self.conv_layer2(conv_out1)
        conv_out = conv_out.view(1, -1)
        output = self.fc_layer(conv_out)
        return output







def model_eval(dataloaders, model,name,  Max = 0):
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
    acu = torch.sum(all_test_output == all_test_target).item() / 580.0
    acu_percent = acu * 100
    print(f'Accuracy: {acu_percent:.2f}%')
    print(Max)
    if acu_percent > Max:
        torch.save(model.state_dict(), os.path.join('../model used for get result/best models', name))
        Max = acu_percent
    return Max



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #所有GPU
    torch.cuda.manual_seed(seed)     # 当前GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
    # torch.use_deterministic_algorithms(True)


