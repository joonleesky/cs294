#!/usr/bin/env python


import tensorflow as tf
import numpy as np
import tf_util

class CloneNetwork:
    def __init__(self, input_size, output_size, config):
        self.input_size    = input_size
        self.output_size   = output_size
        self.hidden_sizes   = config.hidden_sizes
        self.learning_rate = config.learning_rate

        self.build_network()


    def build_network(self):
        self.add_placeholder()
        self.add_embedding()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_train_op(self.loss)

        self.add_summary()

    def train(self, sess, x, y):
        _ , summary = sess.run([self.train_op, self.merged],
                        feed_dict = {self.X:x, self.Y:y})
        return summary

    def predict(self, sess, x):
        return sess.run(self.pred, feed_dict = {self.X:x})


    def add_placeholder(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.output_size])

    def add_embedding(self):
        self.global_step = tf.Variable(0, trainable = False, name = 'global_step')
        
        self.W1 = tf.get_variable("W1", shape=[self.input_size, self.hidden_sizes[0]],
           initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.zeros([self.hidden_sizes[0]])
        
        self.W2 = tf.get_variable("W2", shape=[self.hidden_sizes[0], self.hidden_sizes[1]],
           initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.zeros([self.hidden_sizes[1]])
        
        self.W3 = tf.get_variable("W3", shape=[self.hidden_sizes[1], self.output_size],
           initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.zeros([self.output_size])

    def add_prediction_op(self):
        h1 = tf.nn.relu(tf.matmul(self.X,self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1,self.W2) + self.b2)
        pred   = tf.matmul(h2, self.W3) + self.b3
        
        return pred

    def add_loss_op(self, pred):
        return tf.reduce_sum(tf.square(self.Y - pred))

    def add_train_op(self, loss):
        return tf.train.AdamOptimizer(learning_rate =self.learning_rate).minimize(loss, global_step = self.global_step)

    def add_summary(self):
        tf.summary.scalar('loss',self.loss)
        self.merged = tf.summary.merge_all()