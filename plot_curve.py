import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def gen_curve():
    

    tf_dir = './data/tmp/ac/ac/events.out.tfevents.1683525194.ssrws3'
    for e in tf.train.summary_iterator(tf_dir):
        print(e)

gen_curve()
