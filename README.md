> ⚠️ **Note:**  
> The code currently available on GitHub is from an **older version**.  
> The **newer version has been completely rewritten**, but it's currently **disorganized** and **not ready for release**.  
>  
> I haven’t had the time to clean it up yet, but I plan to **revise and upload it after my graduation**.  
>  
> In the meantime, I’ve included the **detailed model code below** for reference.
<pre>
import torch
import torch.nn as nn

def get_vector(matrix, n):
    sum = []
    for j in range(matrix.shape[0]):
        real_eigenvectors = []
        U, S, V = torch.svd(matrix[j])
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



class fCNNModel(nn.Module):
    def __init__(self, input_size, output_size, k):     #k是特征向量大小
        super(fCNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            #
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=k, stride=k),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear((int(input_size / k)) * 36, output_size),  # 更新线性层的输入大小
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.flatten(1, -1)
        output = self.fc_layer(conv_out)
        return output



class kCNNModel(nn.Module):
    def __init__(self, input_size, output_size, k):     #k是特征向量大小
        super(kCNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            #
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=k, stride=k),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear((int(input_size/k))*36, output_size),  # 更新线性层的输入大小
        )

    def forward(self, x):
        conv_out = self.conv_layer(x)
        conv_out = conv_out.flatten(1, -1)
        output = self.fc_layer(conv_out)
        return output



#hyperparameter (4） stride=FZ, kernel_size=FZ
feature_map = torch.randn((1, 64, 10, 10))   #This feature_map is the feature map extracted by the convolutional model.
cnn=fCNNModel(3200, 10, 10)
v = get_bat_v(feature_map, 5).unsqueeze(0)
print(v.shape)   # torch.Size([1, 3200])
out = cnn(v)
print(out.shape)  # torch.Size([1, 10])



#hyperparameter (5） stride=k*FZ, kernel_size=k*FZ
feature_map = torch.randn((1, 64, 10, 10))   #This feature_map is the feature map extracted by the convolutional model.
cnn = kCNNModel(3200, 10, 50)
v = get_bat_v(feature_map, 5).unsqueeze(0)
print(v.shape)   # torch.Size([1, 3200])
out = cnn(v)
print(out.shape)  # torch.Size([1, 10])
</pre>
