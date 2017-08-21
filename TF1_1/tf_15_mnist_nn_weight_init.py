import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W_2 = tf.Variable(tf.random_normal([784, 30]) / tf.sqrt(784.0))
    b_2 = tf.Variable(tf.random_normal([30]))
    z_2 = tf.matmul(x, W_2) + b_2
    a_2 = tf.sigmoid(z_2)

    W_3 = tf.Variable(tf.random_normal([30, 10]) / tf.sqrt(30.0))
    b_3 = tf.Variable(tf.random_normal([10]))
    z_3 = tf.matmul(a_2, W_3) + b_3
    a_3 = tf.sigmoid(z_3)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    loss = tf.reduce_mean(tf.norm(y_ - a_3, axis=1)**2) / 2
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_3))
    train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    best = 0
    for epoch in range(30):
        for _ in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(a_3, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
        accuracy_currut = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels})
        print("Epoch %s: %s / 10000" % (epoch, accuracy_currut))
        best = (best, accuracy_currut)[best <= accuracy_currut]

    # Test trained model
    print("best: %s / 10000" % best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
