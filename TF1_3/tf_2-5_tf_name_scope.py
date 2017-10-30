import tensorflow as tf

with tf.name_scope('V1'):
    a1 = tf.Variable([50])
    a2 = tf.Variable([100], name='a1')
    d5 = a2

assert a1.name == 'V1/Variable:0'
assert a2.name == 'V1/a1:0'

with tf.name_scope("V2"):
    a1 = tf.add(a1, a2, name="Add_Variable_a1")
    a2 = tf.multiply(a1, a2, name="Add_Variable_a1")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    assert a1.name == 'V2/Add_Variable_a1:0'
    assert sess.run(a1) == 150
    assert a2.name == 'V2/Add_Variable_a1_1:0'
    assert sess.run(a2) == 15000

a2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='V1/a1:0')[0]
assert a2.name == 'V1/a1:0'
