from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.baseline.load import X_train, y_train, X_test, y_test

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data.cleaning.process import OUTCOME_CODES

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, predictions))
print("Macro F1: ", f1_score(y_test, predictions, average="macro"))
print(classification_report(y_test, predictions, target_names=list(OUTCOME_CODES)))
print(confusion_matrix(y_test, predictions))