from __future__ import print_function

import tensorflow as tf
import numpy

#import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.000001
training_epochs = 10000
display_step = 200
beta = 0.01
# Training Data
train_X1 = numpy.asarray([2104.0,1600.0,2400.0,1416.0,3000.0,1985.0,1534.0,1427.0,1380.0,1494.0,1940.0,2000.0,1890.0,4478.0,1268.0,2300.0,1320.0,1236.0,2609.0,3031.0,1767.0])
train_X2 = numpy.asarray([3.0,3.0,3.0,2.0,4.0,4.0,3.0,3.0,3.0,3.0,4.0,3.0,3.0,5.0,3.0,4.0,2.0,3.0,4.0,4.0,3.0])
train_Y = numpy.asarray([399900.0,329900.0,369000.0,232000.0,539900.0,299900.0,314900.0,198999.0,212000.0,242500.0,239999.0,347000.0,329999.0,699900.0,259900.0,449900.0,299900.0,199900.0,499998.0,599000.0,252900.0])
n_samples = train_Y.shape[0]

# tf Graph Input
X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W1 = tf.Variable(rng.randn(), name="weight1")
W2 = tf.Variable(rng.randn(), name="weight2")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
p = tf.add(tf.multiply(X1, W1), tf.multiply(X2, W2))
pred = tf.add(p, b)
reg = tf.add(tf.pow(W1,2),tf.pow(W2,2))
rp  = tf.multiply(beta,reg)
# Mean squared error
cost = tf.reduce_sum(tf.add((tf.pow(pred-Y, 2)),rp))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x1, x2, y) in zip(train_X1, train_X2, train_Y):
            sess.run(optimizer, feed_dict={X1: x1, X2: x2, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X1: train_X1, X2: train_X2, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X1: train_X1, X2: train_X2, Y: train_Y})
    print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "b=", sess.run(b), '\n')

    # Testing
    test_X1 = numpy.asarray([1888.0,1604.0,1962.0,3890.0,1100.0,1458.0,2526.0,2200.0,2637.0,1839.0])
    test_X2 = numpy.asarray([2.0,3.0,4.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0])
    test_Y  = numpy.asarray([ 255000.0,242900.0,259900.0,573900.0,249900.0,464500.0,469000.0,475000.0,299900.0,349900.0])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X1.shape[0]),
        feed_dict={X1: test_X1, X2: test_X2, Y: test_Y})  # same function as cost above
    mse=tf.sqrt(abs(training_cost - testing_cost))
    with tf.Session() as sess:
        m_val=mse.eval()
    print("Testing cost=", testing_cost)
    print("Mean square error:",m_val)

    #Regularisation
    
