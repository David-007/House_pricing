from __future__ import print_function
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")

X = data[::,0:2]
Y = data[::,-1:] #
m,n = X.shape
X_bias = np.ones((m,n+1))
X_bias[::,1:] = X
XT = tf.transpose(X_bias)
W = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X_bias)),XT),Y)
with tf.Session() as sess:
    W_value=W.eval()
#print("X_bias = \n",X_bias[0:5,:])
#print("Y = \n",Y[0:5,::])
print("\nBefore Regularisation:\n W = \n",W_value[0:3,::])
#data1 = np.loadtxt("ex1data3.txt",dtype=np.float64,delimiter=",")
#x = tf.placeholder(tf.float32)
X_predict = np.array([1.0,3000.0,4])
#wt=tf.transpose(W_value)
#xt=tf.transpose(x)
hypothesis = X_predict.dot(W_value)
print( "Cost of house with 3000 sq ft and 4 bedroom is ",hypothesis)
#Regularisation
#beta=0.01 m1 = tf.Variable(tf.eye(n+1)) m1[0][0]=0
WR = tf.matrix_solve_ls(X_bias, Y, l2_regularizer=0.01, fast=True, name=None)
with tf.Session() as sess:
    WR_value=WR.eval()
print("\nAfter Regularisation:\nLambda = 0.01\n WR = \n",WR_value[0:3,::])
hypothesis = X_predict.dot(WR_value)
print( "Cost of house with 3000 sq ft and 4 bedroom is ",hypothesis)
#Mean square error
y_p = tf.matmul(X_bias,WR_value)
ydiff=tf.subtract(y_p,Y)
ydt=tf.transpose(ydiff)
ms =tf.matmul(ydt,ydiff)
mse=tf.sqrt(tf.scalar_mul(1/m,ms))
with tf.Session() as sess:
    m_val=mse.eval()
print("\n Mean Square error:\n",m_val)