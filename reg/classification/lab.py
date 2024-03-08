import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from reg.classification.LogisticReg import LogisticReg

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

n_samples = 100

X = np.linspace(-10, 10, n_samples)

Y = np.where(X > 0, 1, 0)

noise = np.random.normal(0, 0.1, n_samples)
Y_noisy = np.where(X + noise > 0, 1, 0)

clazz_reg_model = LogisticReg(learning_rate=0.01, iterations=1000)
clazz_reg_model.fit(X, Y_noisy)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_noisy, test_size=0.2, random_state=42)
predictions = clazz_reg_model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, predictions))
print("Precision:", precision_score(Y_test, predictions))
print("Recall:", recall_score(Y_test, predictions))
print("F1 Score:", f1_score(Y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_test, predictions))

