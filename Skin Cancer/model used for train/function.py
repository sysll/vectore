import torch
import os

def model_eval(dataloaders, model, name, Max):
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
    acu = torch.sum(all_test_output == all_test_target).item() / 660.0
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
    torch.use_deterministic_algorithms(True)



def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

