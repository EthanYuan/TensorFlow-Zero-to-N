import tensorflow as tf

# model parameters
a = tf.Variable([-1.], tf.float32)
b = tf.Variable([50.], tf.float32)

# model input and output
x = tf.placeholder(tf.float32)
linear_model = a * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) / 8

# training data
x_train = [22, 25, 28, 30]
y_train = [18, 15, 12, 10]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 1st

print("loss: %s" % (sess.run(loss, {x: x_train, y: y_train})))

# 2nd
fixa = tf.assign(a, [-1.])
fixb = tf.assign(b, [40.])
sess.run([fixa, fixb])

print("loss: %s" % (sess.run(loss, {x: x_train, y: y_train})))
