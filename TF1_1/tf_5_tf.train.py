import tensorflow as tf

# model parameters
a = tf.Variable([-1.], tf.float32)
b = tf.Variable([50.], tf.float32)

# model input and output
x = tf.placeholder(tf.float32)
linear_model = a * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) / 8   # sum of the squares

# training data
x_train = [22, 25, 28, 30]
y_train = [18, 15, 12, 10]

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0028)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(70000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_a, curr_b, curr_loss = sess.run([a, b, loss], {x: x_train, y: y_train})
print("a: %s b: %s loss: %s" % (curr_a, curr_b, curr_loss))
