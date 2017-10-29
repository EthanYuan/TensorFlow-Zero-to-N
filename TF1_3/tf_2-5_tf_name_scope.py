import tensorflow as tf

with tf.name_scope('V1'):
    a1 = tf.Variable(tf.random_normal(
        shape=[2, 3], mean=0, stddev=1))
    a2 = tf.Variable([(100, 100, 100), (50, 50, 50.0)])
    a3 = tf.add(a1, a2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    assert a1.name == 'V1/Variable:0'
    assert a2.name == 'V1/Variable_1:0'
    assert a3.name == 'V1/Add:0'

with tf.name_scope("V2"):
    a1 = tf.Variable([1], name='a1')
    a2 = tf.Variable(tf.random_normal(
        shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    assert a1.name == 'V2/a1:0'
    assert a2.name == 'V2/a2:0'
