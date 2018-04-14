import random

class  ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# consider f(x)=y, so x is data and y is target
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()
my_classifier = ScrappyKNN()

# training classifier with test data
my_classifier.fit(X_train, y_train)

# make predictions on testing data
predictions = my_classifier.predict(X_test)
# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
