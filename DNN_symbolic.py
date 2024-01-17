import tensorflow as tf
import sympy
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from keras import backend as K
from keras import initializers
import time


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, activation_functions, mask_alpha=None, weight_initializer=None, kernel_regularizer=None, activity_regularizer=None, **kwargs):
        
        super(MyLayer, self).__init__(**kwargs)
        self.activation_functions =[]
        for z in activation_functions:
            if str(z) == '1' or str(z) == '*':
                self.activation_functions.append('*')
            elif str(z) == 'c':
                self.activation_functions.append('c')
            elif str(z) == 'power':
                self.activation_functions.append('power')
            else:
                self.activation_functions.append(sympy.parsing.sympy_parser.parse_expr(str(z)))
        self.num_outputs = len(activation_functions)
        self.mask_alpha = mask_alpha
        self.weight_initializer = weight_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):

        if self.weight_initializer is not None:
            if self.weight_initializer.shape[0] != input_shape[-1]:
                self.weight_initializer = tf.tile(self.weight_initializer,[input_shape[-1],1]) # repeat characteristics in initializer, for hidden layers
            self.kernel = self.add_weight(name='kernel',
                                          shape=self.weight_initializer.shape,
                                          initializer=tf.keras.initializers.Constant(self.weight_initializer),
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)
            if 'power' in self.activation_functions:
                total=self.activation_functions.count('power')
                self.bias = self.add_weight(shape=(total,),
                                        initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=10),
                                        trainable=True,
                                        name='bias')
        else:
            self.kernel = self.add_weight(name='kernel', 
                                        shape=(input_shape[-1], self.num_outputs),
                                        initializer='glorot_uniform',
                                        regularizer=self.kernel_regularizer,
                                        trainable=True)
        super(MyLayer, self).build(input_shape)


    def call(self, inputs):

        output_list = []
        if self.mask_alpha is not None:
            weights_prod=self.filter_mat(threshold=self.mask_alpha)
        else:
            weights_prod=self.filter_mat() 
        output = tf.matmul(inputs, self.kernel)
        
        for i in range(self.num_outputs):
            if self.activation_functions[i] == '*':
                output_i = inputs*weights_prod[:,i]
                output_i=tf.boolean_mask(output_i,~tf.reduce_all(tf.equal(output_i, 0.0), axis=0),axis=1)
                output_i=tf.reduce_prod(output_i,axis=-1)
            elif self.activation_functions[i] == 'c':
                output_shape = tf.shape(output)
                tensor = tf.ones(shape=(output_shape[0],weights_prod.shape[0]),dtype=tf.float32)
                value =  tf.expand_dims(weights_prod[:,i], axis=1)
                output_i = tf.squeeze(tf.matmul(tensor,value))
            elif self.activation_functions[i] == 'power':
                output_i = tf.matmul(tf.pow(inputs, self.bias),self.kernel)
            else:
                output_i = self.eval_expr(self.activation_functions[i], output[:, i])


            output_list.append(output_i)
        return tf.stack(output_list, axis=1)

    def get_config(self):
        config = super(MyLayer, self).get_config()
        config.update({'output_dim': self.num_outputs,
                       'activation_functions': [str(x) for x in self.activation_functions],
                       'mask_neuron': self.mask_neuron,
                       'mask_alpha': self.mask_alpha,
                       'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
                       'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer)})
        return config