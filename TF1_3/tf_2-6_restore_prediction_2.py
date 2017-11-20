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

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'MNIST/logs/tf2-6/checkpoint_2/model.ckpt-10.meta')
        saver.restore(sess, tf.train.latest_checkpoint(
            'MNIST/logs/tf2-6/checkpoint_2'))
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        print(x)
        print(sess.run('accuracy/accuracy:0', feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
