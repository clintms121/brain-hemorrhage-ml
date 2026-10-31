from sklearn.dummy import DummyClassifier
from ml.baseline.load import X_train, y_train

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)