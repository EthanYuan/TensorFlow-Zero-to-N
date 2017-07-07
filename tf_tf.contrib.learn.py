import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are
# many other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])

x = np.array([7, 8, 9, 10])
y = np.array([33, 32, 31, 30])

x = np.array([1, 2, 3, 4])
y = np.array([39, 38, 37, 36])

x = np.array([22, 25, 28, 30])
y = np.array([18, 15, 12, 10])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
                                              num_epochs=5000)

# We can invoke 1000 training steps by invoking the `fit` method and passing
# the training data set.
estimator.fit(input_fn=input_fn, steps=5000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
# print(estimator.predict(x=[x, features]))
