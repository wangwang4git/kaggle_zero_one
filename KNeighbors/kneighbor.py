#!/Users/wangwang/anaconda/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import load_iris

iris = load_iris()
# print iris.data.shape
# print iris.DESCR

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)

from sklearn.metrics import classification_report

print "kneighbors acc = ", knc.score(x_test, y_test)
print classification_report(y_test, y_predict, target_names=iris.target_names)
