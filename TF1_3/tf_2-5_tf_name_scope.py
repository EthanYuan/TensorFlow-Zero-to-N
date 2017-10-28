import tensorflow as tf

with tf.name_scope('V1'):
    a1 = tf.get_variable(name='a1', shape=[1],
                         initializer=tf.constant_initializer(1))

    a2 = tf.Variable(tf.random_normal(
        shape=[2, 3], mean=0, stddev=1))
    a3 = tf.Variable([100])

with tf.variable_scope("V2"):
    b1 = tf.get_variable(
        name='a1', shape=[1], initializer=tf.constant_initializer(1))
    b2 = tf.Variable(tf.random_normal(
        shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.variable_scope("V2", reuse=True):
    c3 = tf.get_variable('a1')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    assert a1.name == 'a1:0'
    assert sess.run(a1) == [1.]
    assert a2.name == 'V1/Variable:0'
    assert a3.name == 'V1/Variable_1:0'

    assert b1.name == 'V2/a1:0'
    assert b2.name == 'V2/a2:0'

    assert c3.name == 'V2/a1:0'

assert a1.name == 'a1:0'
assert a2.name == 'V1/Variable:0'