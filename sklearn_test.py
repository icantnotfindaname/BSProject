from sklearn.neighbors import KNeighborsClassifier
from config import read_config
import numpy as np
import pandas as pd

train = pd.read_csv(read_config("train_label_dir") + '/train.csv')
print(train.head)
train_x = np.loadtxt('./result/train_glcm.csv', skiprows=1, delimiter=',')
train_y = np.loadtxt('./result/train_label.csv', skiprows=1)
test_x = np.loadtxt('./result/test_glcm.csv', skiprows=1, delimiter=',')
test_y = np.loadtxt('./result/test_label.csv', skiprows=1)
print(train_x.shape)

n = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', p = 2)
n.fit(train_x, train_y)

count = 0
c = 0;
for i in test_x:
    if n.predict(i) == test_y[c]:
        count += 1
    c += 1

print(count)

