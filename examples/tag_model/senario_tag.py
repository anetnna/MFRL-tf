"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import logging
import time
import logging as log
import numpy as np


from env.mpe.make_env import make_env

logging.basicConfig(level=logging.ERROR)



def _calc_moment(a):
    moment = np.array([1,a[0],a[1],a[0]*a[1],a[0]**2,a[1]**2])
    return moment

def play(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    max_nums = [20, 40]  # 20 predators, 40 prey


    action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]

    all_obs = env.reset()
    obs = [all_obs[:20], all_obs[20:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],action_dim[i]), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_meanaction = [np.zeros((1, action_dim[0])), np.zeros((1, action_dim[1]))]

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            acts[i], values[i], logprobs[i] = models[i].act(state=obs[i])

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:20], all_obs[20:]]
        rewards = [all_rewards[:20], all_rewards[20:]]
        done = all(all_done)

        predator_buffer = {
            'state': old_obs[0], 
            'acts': acts[0], 
            'rewards': rewards[0], 
            'dones': all_done[:20],
            'values': values[0], 
            'logps': logprobs[0],
            'ids': range(max_nums[0]), 
        }

        prey_buffer = {
            'state': old_obs[1], 
            'acts': acts[1], 
            'rewards': rewards[1], 
            'dones': all_done[20:],
            'values': values[1], 
            'logps': logprobs[1],
            'ids': range(max_nums[1]), 
        }

        #############################
        # Calculate former_act_prob #
        #############################
        # obs, idx, te_arr may change its len every step, while acts, pi, former_act_prob, onehot_former_acts always keep its shape.
        # for i in range(n_group):
            # former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0,keepdims=True)

        if train:
            models[0].flush_buffer(**predator_buffer)
            models[1].flush_buffer(**prey_buffer)
        
        for i in range(n_group):
            former_meanaction[i] = _calc_moment(np.mean(acts[i], axis=0))

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            mean_rewards[i] = sum_reward / max_nums[i]
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        info = {"Mean-Reward": np.round(mean_rewards, decimals=6), "NUM": max_nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    predator_buffer = {
        'state': obs[0], 
        'acts': [None for i in range(max_nums[0])], 
        'rewards': [None for i in range(max_nums[0])], 
        'dones': [None for i in range(max_nums[0])],
        'values': [None for i in range(max_nums[0])], 
        'logps': [None for i in range(max_nums[0])],
        'ids': range(max_nums[0]), 
    }

    prey_buffer = {
        'state': obs[1], 
        'acts': [None for i in range(max_nums[1])], 
        'rewards': [None for i in range(max_nums[1])], 
        'dones': [None for i in range(max_nums[1])],
        'values': [None for i in range(max_nums[1])], 
        'logps': [None for i in range(max_nums[1])],
        'ids': range(max_nums[1]), 
    }

    if train:
        models[0].flush_buffer(**predator_buffer)
        models[1].flush_buffer(**prey_buffer)

        models[0].train()
        models[1].train()

    for i in range(n_group):
        total_rewards[i] = sum(total_rewards[i])

    return total_rewards

def test_play(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()
    env.discrete_action_input = True

    step_ct = 0
    done = False
    n_group = 2

    pos_reward_ct = set()
    rewards = [None for _ in range(n_group)]
    max_nums = [20, 40]

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.action_space[0].n for _ in range(n_group)]

    all_obs = env.reset()
    obs = [all_obs[:20], all_obs[20:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums[0]))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.action_space[0].n)), np.zeros((1, env.action_space[-1].n))]

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (max_nums[i], 1))
            acts[i], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=True)

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:20], all_obs[20:]]
        rewards = [all_rewards[:20], all_rewards[20:]]
        done = all(all_done)

        #############################
        # Calculate former_act_prob #
        #############################
        # obs, idx, te_arr may change its len every step, while acts, pi, former_act_prob, onehot_former_acts always keep its shape.
        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0,keepdims=True)

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            mean_rewards[i] = sum_reward / max_nums[i]
            total_rewards[i].append(sum_reward)


        info = {"Mean-Reward": np.round(mean_rewards, decimals=6), "NUM": max_nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        total_rewards[i] = sum(total_rewards[i])

    return total_rewards
