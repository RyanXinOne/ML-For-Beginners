'''Take a look at the Linnerud dataset in Scikit-learn. This dataset has multiple targets: 'It consists of three exercise (data) and three physiological (target) variables collected from twenty middle-aged men in a fitness club'.

In your own words, describe how to create a Regression model that would plot the relationship between the waistline and how many situps are accomplished. Do the same for the other datapoints in this dataset.'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# get data
X, y = datasets.load_linnerud(return_X_y=True)

# filter exercise data on Situps and Waist
X = X[:, np.newaxis, 1]
y = y[:, 1]

# split dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

# create model
model = linear_model.LinearRegression()
# train model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# plot results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel("Situps")
plt.ylabel("Waist")
plt.show()
