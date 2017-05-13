#!/Users/wangwang/anaconda/bin/python
# -*- coding: UTF-8 -*-

# ==========
# 1. 数据导入
import pandas as pd
import numpy as np

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size ', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']
# print column_names

# 导入本地数据
data = pd.read_csv('../data/breast-cancer-wisconsin.data', names=column_names)
# print type(data)

# 异常数据处理
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
# print data.shape

# ==========
# 2. 准备训练数据、测试数据
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                    test_size=0.25, random_state=33)
# print y_train.value_counts()
# print y_test.value_counts()

# ==========
# 3. 模型前数据预处理
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# 标准化处理，方差为1，均值为0
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# ==========
# 4. 建模
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
# 训练
lr.fit(x_train, y_train)
# 预测
lr_y_predict = lr.predict(x_test)

from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)

# ==========
# 5. 模型结果检测
from sklearn.metrics import classification_report

print "lr acc = ", lr.score(x_test, y_test)
print classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant'])
print "sgdc acc = ", sgdc.score(x_test, y_test)
print classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant'])

# 训练数据在10W量级以上，则推荐SGDClassifier，计算耗时小