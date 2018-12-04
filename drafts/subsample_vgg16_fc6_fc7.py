import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import numpy as np
import os
import tools

base_dir = tools.get_base_dir()
original_weights_path = os.path.join(base_dir, 'weights', 'vgg_16.ckpt')
outdir = os.path.join(base_dir, 'weights', 'vgg_16_for_ssd')
if not os.path.exists(outdir):
    os.makedirs(outdir)

vgg = tf.contrib.slim.nets.vgg
inputs = tf.placeholder(shape=(None, 224, 224, 3), dtype=tf.float32)
# inputs = tf.Variable(shape=(None, 300, 300, 3), dtype=tf.float32)
logits, _ = vgg.vgg_16(inputs, num_classes=10, is_training=True)

varnames_to_restore = []
for var in tf.trainable_variables():
    if 'fc8' not in var.name:
        varnames_to_restore.append(var.name)

vars_to_restore = tf.contrib.framework.get_variables_to_restore(include=varnames_to_restore)

restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(original_weights_path, vars_to_restore)


with tf.Session() as sess:
    restore_fn(sess)
    for var in vars_to_restore:
        print('')
        print(var.name)
        x = sess.run(var)
        print(x.shape)
        print(str(np.min(x)) + ' < ' + str(np.mean(x)) + ' < ' + str(np.max(x)) + '  (' + str(np.std(x)) + ')')
    for var in vars_to_restore:
        if 'fc6' in var.name:
            if 'weights' in var.name:
                print('fc6 weights')
                weights = sess.run(var)
                np_fc6_weights_sub = weights[0:7:3, 0:7:3, :, 0:4096:4]
                print('weights.shape: ' + str(weights.shape))
                print('weights_sub.shape: ' + str(np_fc6_weights_sub.shape))
            elif 'biases' in var.name:
                print('fc6 bias')
                bias = sess.run(var)
                np_fc6_bias_sub = bias[0:4096:4]
                print('bias.shape: ' + str(bias.shape))
                print('bias_sub.shape: ' + str(np_fc6_bias_sub.shape))
    for var in vars_to_restore:
        if 'fc7' in var.name:
            if 'weights' in var.name:
                print('fc7 weights')
                weights = sess.run(var)
                np_fc7_weights_sub = weights[:, :, 0:4096:4, 0:4096:4]
                print('weights.shape: ' + str(weights.shape))
                print('weights_sub.shape: ' + str(np_fc7_weights_sub.shape))
            elif 'biases' in var.name:
                print('fc7 bias')
                bias = sess.run(var)
                np_fc7_bias_sub = bias[0:4096:4]
                print('bias.shape: ' + str(bias.shape))
                print('bias_sub.shape: ' + str(np_fc7_bias_sub.shape))


fc6_weights_sub = tf.Variable(initial_value=np_fc6_weights_sub, dtype=tf.float32, name='vgg_16/fc6_sub/weights')
fc6_bias_sub = tf.Variable(initial_value=np_fc6_bias_sub, dtype=tf.float32, name='vgg_16/fc6_sub/biases')
fc7_weights_sub = tf.Variable(initial_value=np_fc7_weights_sub, dtype=tf.float32, name='vgg_16/fc7_sub/weights')
fc7_bias_sub = tf.Variable(initial_value=np_fc7_bias_sub, dtype=tf.float32, name='vgg_16/fc7_sub/biases')

vars_sub = [fc6_weights_sub, fc6_bias_sub, fc7_weights_sub, fc7_bias_sub]
vars_new = []
for var in tf.trainable_variables():
    if var not in vars_to_restore:
        vars_new.append(var)
init_op = tf.variables_initializer(vars_new)

saver = tf.train.Saver()

with tf.Session() as sess:
    restore_fn(sess)
    sess.run(init_op)
    for var in vars_to_restore:
        print('')
        print(var.name)
        x = sess.run(var)
        print(x.shape)
        print(str(np.min(x)) + ' < ' + str(np.mean(x)) + ' < ' + str(np.max(x)) + '  (' + str(np.std(x)) + ')')
    for var in vars_sub:
        print('')
        print(var.name)
        x = sess.run(var)
        print(x.shape)
        print(str(np.min(x)) + ' < ' + str(np.mean(x)) + ' < ' + str(np.max(x)) + '  (' + str(np.std(x)) + ')')
    save_path = saver.save(sess, os.path.join(outdir, 'vgg_16_for_ssd.ckpt'))
    print("Model saved in path: %s" % save_path)
