import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import json
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


with open("all_predict_score1.json", 'r') as file:
    data1 = json.load(file)
    data1 = np.array(data1)
    data1 = softmax(data1)

with open("all_predict_score2.json", 'r') as file:
    data2 = json.load(file)
    data2 = np.array(data2)
    data2 = softmax(data2)

with open("True_label_1", 'r') as file:
    label1 = json.load(file)
    label1 = np.array(label1)

with open("True_label2.json", 'r') as file:
    label2 = json.load(file)
    label2 = np.array(label2)
#里面的data1和data2都是shape为[580, 10]的数据，data1是原始ResNet模型的，data2是我们的模型的。label是[580]的标签


# 假设predict和true是已经给定的列表，这里用随机数据代替

true = label2
scores = data2
predict = np.argmax(scores, axis=1)  # 预测类别

# 初始化图像和子图
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(19, 3.7))


# 遍历每个类别
for i in range(5):

    # 获取当前类别的真实标签和预测得分
    y_true = (true == i).astype(int)
    y_scores = scores[:, i]

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[i].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color="#3EC3AE")
    axes[ i].plot([0, 1], [0, 1], 'k--', color="#134857")
    axes[ i].set_xlim([0.0, 1.0])
    axes[ i].set_ylim([0.0, 1.05])
    axes[ i].set_xlabel('False Positive Rate', fontsize=12)
    axes[ i].set_ylabel('True Positive Rate', fontsize=12)
    axes[ i].set_title(f'({i+1}) ROC curve')
    axes[ i].legend(loc="lower right")

plt.tight_layout()
plt.show()

# 初始化图像和子图
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(19, 3.7))
i=0
for j in range(5,10):

    # 获取当前类别的真实标签和预测得分
    y_true = (true == j).astype(int)
    y_scores = scores[:, j]

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    axes[ i].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color="#3EC3AE")
    axes[ i].plot([0, 1], [0, 1], 'k--', color="#134857")
    axes[ i].set_xlim([0.0, 1.0])
    axes[ i].set_ylim([0.0, 1.05])
    axes[ i].set_xlabel('False Positive Rate', fontsize=12)
    axes[ i].set_ylabel('True Positive Rate', fontsize=12)
    axes[ i].set_title(f'({j+1}) ROC curve')
    axes[ i].legend(loc="lower right")
    i=i+1

plt.tight_layout()
plt.show()






fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(19, 3.7))
# 遍历每个类别

for i in range(5):
    y_true = (true == i).astype(int)
    y_scores = scores[:, i]

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    axes[ i].plot(recall, precision, label=f'PR (AP = {pr_auc:.4f})', color="#3EC3AE")
    axes[i].plot([0, 1], [1, 0], 'k--', color="#134857")
    axes[ i].set_xlim([0.0, 1.0])
    axes[ i].set_ylim([0.0, 1.05])
    axes[ i].set_xlabel('Recall', fontsize=12)
    axes[ i].set_ylabel('Precision', fontsize=12)
    axes[ i].set_title(f'({i+11}) PR curve')
    axes[ i].legend(loc="lower right")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(19, 3.7))
# 遍历每个类别
i=0
for j in range(5, 10):
    y_true = (true == j).astype(int)
    y_scores = scores[:, j]
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    axes[ i].plot(recall, precision, label=f'PR (AP = {pr_auc:.4f})', color="#3EC3AE")
    axes[i].plot([0, 1], [1, 0], 'k--', color="#134857")
    axes[ i].set_xlim([0.0, 1.0])
    axes[ i].set_ylim([0.0, 1.05])
    axes[ i].set_xlabel('Recall', fontsize=12)
    axes[ i].set_ylabel('Precision', fontsize=12)
    axes[ i].set_title(f'({j+11}) PR curve')
    axes[ i].legend(loc="lower right")
    i=i+1
plt.tight_layout()
plt.show()