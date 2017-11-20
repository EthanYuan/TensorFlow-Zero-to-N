import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from importlib import import_module
import_module('tf_2-6_checkpoint')

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,
                                      validation_size=10000)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    cnn = sys.modules['tf_2-6_checkpoint']
    y_conv, keep_prob = cnn.deepnn(x)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('MNIST/logs/tf2-6/checkpoint')
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        print(ckpt)
        accuracy_test = accuracy.eval(
                    feed_dict={
                        x: mnist.test.images,
                        y_: mnist.test.labels,
                        keep_prob: 1.0})
        print('test accuracy %s' % accuracy_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
