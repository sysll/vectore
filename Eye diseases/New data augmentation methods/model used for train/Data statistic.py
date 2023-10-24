#358是类别0，
import os
import cv2
import random
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil
data=pd.read_csv('D:/Users/ASUS/Desktop/百度下载位置/眼球数据/dataset/labels.csv')
Type_of_category = pd.unique(data["category"])
dic_category = {}
for j in Type_of_category:
    dic_category[j]=data['category'].value_counts()[j]

print("类别以及对应的个数是")
print(dic_category)

sns.barplot(x = list(dic_category.keys()), y = list(dic_category.values()))
plt.suptitle("Type_of_category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()


