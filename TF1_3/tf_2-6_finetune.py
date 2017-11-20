import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from importlib import import_module
import_module('tf_2-6_checkpoint')

FLAGS = None


def cnn():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    cnn = sys.modules['tf_2-6_checkpoint']
    y_conv, keep_prob = cnn.deepnn(x)

    return y_conv, keep_prob, x, y_


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,
                                      validation_size=10000)

    y_conv, keep_prob, x, y_ = cnn()

    fine_tune_var_list = [
        i for i in tf.trainable_variables() if 'fc2' in i.name]

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        gradients = tf.gradients(cross_entropy, fine_tune_var_list)
        gradients = list(zip(gradients, fine_tune_var_list))
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_step = optimizer.apply_gradients(grads_and_vars=gradients)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    best = 0
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('MNIST/logs/tf2-6/checkpoint')
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

        for epoch in range(20):
            for _ in range(1000):
                batch = mnist.train.next_batch(50)
                train_step.run(
                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            accuracy_validation = accuracy.eval(
                feed_dict={x: mnist.validation.images,
                           y_: mnist.validation.labels,
                           keep_prob: 1.0})
            print('epoch %d, validation accuracy %s' %
                  (epoch + 1, accuracy_validation))
            best = (best, accuracy_validation)[best <= accuracy_validation]

    # Test trained model
    print("best: %s" % best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
