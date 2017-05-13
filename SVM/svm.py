#!/Users/wangwang/anaconda/bin/python
# -*- coding: UTF-8 -*-

# 1. 数据导入
from sklearn.datasets import load_digits

digits = load_digits()
# print digits.data.shape
# print type(digits.data)

# 2. 训练数据，测试数据准备
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

# 3. 模型前数据预处理
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 4. 建模
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

# 5. 模型检测
from sklearn.metrics import classification_report

print "svm acc = ", lsvc.score(x_test, y_test)
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))