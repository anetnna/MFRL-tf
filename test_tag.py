"""
Multi-agent particle environment, spread task. Test.
"""
import argparse
import os
import random
import time
import argparse

import numpy as np
# import wandb
from examples.tag_model.algo import spawn_ai
from examples.tag_model.algo import tools
from examples.tag_model.senario_tag import test

from env.mpe.make_env import make_env

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, choices={'attention_mfq', 'causal_mfq','ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--prey', type=str, choices={'attention_mfq', 'causal_mfq', 'ac', 'mfac', 'mfq', 'il'}, help='indicate the opponent model')
    parser.add_argument('--pred_dir', type=str, help='the path of the algorithm')
    parser.add_argument('--prey_dir', type=str, help='the path of the opponent model')
    parser.add_argument('--n_round', type=int, default=10, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=100, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # Initialize the environment
    env = make_env('exp_tag')

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    predator_model_dir = os.path.join(BASE_DIR, 'data/models/{}-predator'.format(args.pred))
    prey_model_dir = os.path.join(BASE_DIR, 'data/models/{}-prey'.format(args.prey))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.pred, sess, env, None, args.pred + 'predator', args.max_steps),
              spawn_ai(args.prey, sess, env, None, args.prey + 'prey', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(predator_model_dir, step=args.idx[0])
    models[1].load(prey_model_dir, step=args.idx[1])

    runner = tools.Runner(sess, env, None, args.map_size, args.max_steps, models, test, render_every=0)
    reward_ls = {'predator': [], 'prey': []}

    for k in range(0, args.n_round):
        runner.run(0.0, k, mean_reward=reward_ls)

    print('\n[*] >>> Reward: Predator[{0}] {1} | Prey[{2} {3}]'.format(args.pred, sum(reward_ls['predator']) / args.n_round,
                                                                       args.prey, sum(reward_ls['prey']) / args.n_round))
