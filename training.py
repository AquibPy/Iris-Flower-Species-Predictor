import pickle
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X,Y)

file = open('model.pkl','wb')

pickle.dump(clf,file)
file.close()