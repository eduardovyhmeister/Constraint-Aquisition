import tensorflow as tf
import tensorflow_probability as tfp

def custom_loss(y_true, y_pred): # this is for inequalities x1+x2 >= 7
    """ This is a loss function that helps to find inequalities since is not based on MSE """
    error = y_pred - y_true #this is negative predictions
    abs_error = tf.abs(error)
    threshold = tfp.stats.percentile(abs_error, 5.0)
    mask = tf.math.less(error, threshold)
    y_true_with_low_error = tf.boolean_mask(y_true, mask)
    y_pred_with_low_error = tf.boolean_mask(y_pred, mask)
    MSE_low_error = tf.reduce_mean(tf.square(y_true_with_low_error - y_pred_with_low_error))
    negatives = tf.cast(tf.math.count_nonzero(tf.math.less(error,0)),tf.float32)
    equal = tf.cast(tf.math.count_nonzero(tf.math.equal(error,0)),tf.float32)
    return tf.reduce_mean(error) + 0.5*tf.abs(tf.reduce_max(error))+0.1*MSE_low_error
    #return tf.reduce_mean(error) + 0.5*tf.reduce_max(tf.abs(error))+0.1*MSE_low_error

def custom_loss_2(y_true, y_pred): # this is for inequalities x1+x2 >= 7
    """ This is a loss function that helps to find inequalities since is not based on MSE """
    error = y_true - y_pred #this is negative predictions
    abs_error = tf.abs(error)
    threshold = tfp.stats.percentile(abs_error, 50.0)
    mask = tf.math.less(error, threshold)
    y_true_with_low_error = tf.boolean_mask(y_true, mask)
    y_pred_with_low_error = tf.boolean_mask(y_pred, mask)
    MSE_low_error = tf.reduce_mean(tf.square(y_true_with_low_error - y_pred_with_low_error))
    negatives = tf.cast(tf.math.count_nonzero(tf.math.less(error,0)),tf.float32)
    equal = tf.cast(tf.math.count_nonzero(tf.math.equal(error,0)),tf.float32)
    return tf.reduce_mean(error) + 0.5*tf.abs(tf.reduce_max(error))+0.5*MSE_low_error