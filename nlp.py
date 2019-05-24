import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")
news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
news_df.head()
X_train, X_test, y_train, y_test = train_test_split(
    news_df['TITLE'],  
    news_df['CATEGORY'], 
    random_state = 1
)
count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
############naive bayes###########
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
print("Accuracy score: ", accuracy_score(y_test, predictions))
############
nb = BernoulliNB()
nb.fit(training_data, y_train)
pred = nb.predict(testing_data)
print("Accuracy score:",accuracy_score(y_test,pred))
############decision tree###########
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, y_train)
clf.feature_importances_
pred = clf.predict(testing_data)
print("Accuracy score:",accuracy_score(y_test, pred))
#########svm######################
