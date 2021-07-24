'''In this lesson you were shown how to build a model using both Linear and Polynomial Regression. Using this knowledge, find a dataset or use one of Scikit-learn's built-in sets to build a fresh model. Explain in your notebook why you chose the technique you did, and demonstrate your model's accuracy. If it is not accurate, explain why.'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

"""
Multivariate polynomial linear regression
Explore the relation between Weight and exercises (Chins, Situps and Jumps)
"""

# load data
X, y = load_linnerud(return_X_y=True)
X = X[:, [0]]

# explore correlations
print("Correlation Coefficient")
print('Weight - Chins: ', np.corrcoef(X[:, 0], y[:, 0])[0, 1])
print('Weight - Situps: ', np.corrcoef(X[:, 0], y[:, 1])[0, 1])
print('Weight - Jumps: ', np.corrcoef(X[:, 0], y[:, 2])[0, 1])

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# train model and predict
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# evaluate accuracy
accuracy_score = pipeline.score(X_train, y_train)
print('\nModel Accuracy: ', accuracy_score)

# visualisation
results = [(X_test[i], y_pred[i]) for i in range(X_test.shape[0])]
results.sort(key=lambda x: x[0])
X_result = [m[0] for m in results]
y_result = np.array([m[1] for m in results])
plt.plot(X_result, y_result[:, 0], color='blue', linewidth=2, label='Chins')
plt.plot(X_result, y_result[:, 1], color='orange', linewidth=2, label='Situps')
plt.plot(X_result, y_result[:, 2], color='green', linewidth=2, label='Jumps')
plt.scatter(X, y[:, 0], color='blue')
plt.scatter(X, y[:, 1], color='orange')
plt.scatter(X, y[:, 2], color='green')
plt.xlabel('Weight')
plt.legend()
plt.show()
