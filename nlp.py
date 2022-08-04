import numpy as np
import pandas as pd

yelp = pd.read_csv('yelp.csv')
print(yelp.head())
print(yelp.info())
yelp.describe()

yelp['text length'] = yelp['text'].apply(len)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
plt.show()

sns.countplot(x='stars',data=yelp,palette='rainbow')
plt.show()

stars = yelp.groupby('stars').mean()
print(stars)
print(stars.corr())

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
x = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x = cv.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(x_train,y_train)

predictions = nb.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

x = yelp_class['text']
y = yelp_class['stars']
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=101)

pipeline.fit(x_train,y_train)
predictions = pipeline.predict(x_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))