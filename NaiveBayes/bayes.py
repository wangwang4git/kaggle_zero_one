#!/Users/wangwang/anaconda/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print len(news.data)
print news.data[0]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)

from sklearn.metrics import classification_report
print "bayes acc = ", mnb.score(x_test, y_test)
print classification_report(y_test, y_predict, target_names=news.target_names)