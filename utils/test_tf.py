import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np

"""
order = 3
dummy_action = tf.convert_to_tensor(np.arange(21))
act_prob_input = tf.convert_to_tensor(np.ones((1,21))/21.0,dtype=tf.float32)
act_emb_layer = tf.keras.layers.Embedding(21, 8)
act_emb = act_emb_layer(dummy_action)

act_emb_moment_raw = tf.matmul(act_prob_input, tf.pow(act_emb,order))
act_emb_moment_abs = tf.where(act_emb_moment_raw<0, act_emb_moment_raw*(-1), act_emb_moment_raw)
act_emb_moment = tf.pow(act_emb_moment_abs, 1/order)
act_emb_moment = tf.where(act_emb_moment_raw<0, act_emb_moment*(-1), act_emb_moment)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(dummy_action))
    print(sess.run(act_emb_moment_raw))
    print(sess.run(act_emb_moment))
"""

state_input = tf.placeholder(tf.float32, [None, 32], 'state')
l1 = tf.layers.dense(state_input, 256, tf.nn.relu)

s = np.zeros((10,25,32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(l1, {state_input:s})