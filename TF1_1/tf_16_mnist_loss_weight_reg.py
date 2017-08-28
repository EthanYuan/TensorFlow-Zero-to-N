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
    x = tf.placeholder(tf.float32, [None, 784])
    W_2 = tf.Variable(tf.random_normal([784, 100]) / tf.sqrt(784.0))
    '''W_2 = tf.get_variable(
        name="W_2",
        regularizer=regularizer,
        initializer=tf.random_normal([784, 30], stddev=1 / tf.sqrt(784.0)))'''
    b_2 = tf.Variable(tf.random_normal([100]))
    z_2 = tf.matmul(x, W_2) + b_2
    a_2 = tf.sigmoid(z_2)

    W_3 = tf.Variable(tf.random_normal([100, 10]) / tf.sqrt(100.0))
    '''W_3 = tf.get_variable(
        name="W_3",
        regularizer=regularizer,
        initializer=tf.random_normal([30, 10], stddev=1 / tf.sqrt(30.0)))'''
    b_3 = tf.Variable(tf.random_normal([10]))
    z_3 = tf.matmul(a_2, W_3) + b_3
    a_3 = tf.sigmoid(z_3)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_2)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_3)
    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    loss = (tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3)) +
        reg_term)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(a_3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    scalar_accuracy = tf.summary.scalar('accuracy', accuracy)
    train_writer = tf.summary.FileWriter(
        'MNIST/logs/tf16_reg/train', sess.graph)
    validation_writer = tf.summary.FileWriter(
        'MNIST/logs/tf16_reg/validation')

    # Train
    best = 0
    for epoch in range(30):
        for _ in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Test trained model
        accuracy_currut_train = sess.run(accuracy,
                                         feed_dict={x: mnist.train.images,
                                                    y_: mnist.train.labels})

        accuracy_currut_validation = sess.run(
            accuracy,
            feed_dict={x: mnist.validation.images,
                       y_: mnist.validation.labels})
       
        sum_accuracy_train = sess.run(
            scalar_accuracy,
            feed_dict={x: mnist.train.images,
                       y_: mnist.train.labels})

        sum_accuracy_validation = sess.run(
            scalar_accuracy,
            feed_dict={x: mnist.validation.images,
                       y_: mnist.validation.labels})

        train_writer.add_summary(sum_accuracy_train, epoch)
        validation_writer.add_summary(sum_accuracy_validation, epoch)

        print("Epoch %s: train: %s validation: %s"
              % (epoch, accuracy_currut_train, accuracy_currut_validation))
        best = (best, accuracy_currut_validation)[
            best <= accuracy_currut_validation]

    # Test trained model
    print("best: %s" % best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
