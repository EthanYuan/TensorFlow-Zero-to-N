import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,
                                      validation_size=10000)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    with tf.name_scope('fc1'):
        W_2 = tf.Variable(tf.random_normal([784, 100]) / tf.sqrt(784.0 / 2))
        b_2 = tf.Variable(tf.random_normal([100]))
        z_2 = tf.matmul(x, W_2) + b_2
        a_2 = tf.nn.relu(z_2)

    with tf.name_scope('fc2'):
        W_3 = tf.Variable(tf.random_normal([100, 100]) / tf.sqrt(100.0 / 2))
        b_3 = tf.Variable(tf.random_normal([100]))
        z_3 = tf.matmul(a_2, W_3) + b_3
        a_3 = tf.nn.relu(z_3)

    with tf.name_scope('fc3'):
        W_4 = tf.Variable(tf.random_normal([100, 10]) / tf.sqrt(100.0))
        b_4 = tf.Variable(tf.random_normal([10]))
        z_4 = tf.matmul(a_3, W_4) + b_4
        a_4 = tf.sigmoid(z_4)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_2)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_3)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_4)
    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    with tf.name_scope('loss'):
        loss = (tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_4)) +
            reg_term)

    with tf.name_scope('optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(a_4, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name='accuracy')

    saver = tf.train.Saver()

    # Train
    best = 0
    for epoch in range(30):
        for _ in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Test trained model
        accuracy_currut_train = sess.run(
            accuracy,
            feed_dict={x: mnist.train.images,
                       y_: mnist.train.labels})

        accuracy_currut_validation = sess.run(
            accuracy,
            feed_dict={x: mnist.validation.images,
                       y_: mnist.validation.labels})

        print("Epoch %s: train: %s validation: %s"
              % (epoch, accuracy_currut_train, accuracy_currut_validation))
        best = (best, accuracy_currut_validation)[
            best <= accuracy_currut_validation]
        
        saver.save(sess, 'MNIST/logs/tf2-6/checkpoint_2/model.ckpt', epoch+1)

    # Test trained model
    print("best: %s" % best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
