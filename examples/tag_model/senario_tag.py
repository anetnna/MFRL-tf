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



def _calc_moment(a, order=3):
    moments = []
    for i in range(order):
        for j in range(order):
            moments.append(np.power(a[0],i)*np.power(a[1],j))
    moment = np.array(moments)
    return moment.reshape(1,-1)

# def _calc_moment(a):
    # moment = np.array([1,a[0],a[1],a[0]*a[1],a[0]**2,a[1]**2])
    # return moment

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

    former_meanaction = [np.zeros((1, 9)), np.zeros((1, 9))]

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
            acts[i], values[i], logprobs[i] = models[i].act(state=obs[i], meanaction=former_meanaction[i])

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
        if 'mf' in models[0].name:
            predator_buffer['meanaction'] = former_meanaction[0]

        prey_buffer = {
            'state': old_obs[1], 
            'acts': acts[1], 
            'rewards': rewards[1], 
            'dones': all_done[20:],
            'values': values[1], 
            'logps': logprobs[1],
            'ids': range(max_nums[1]), 
        }
        if 'mf' in models[1].name:
            prey_buffer['meanaction'] = former_meanaction[1]

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
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        info = {"kill": sum(total_rewards[0])/10}

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
    if 'mf' in models[0].name:
        predator_buffer['meanaction'] = np.tile(former_meanaction[0], (max_nums[0], 1))

    prey_buffer = {
        'state': obs[1], 
        'acts': [None for i in range(max_nums[1])], 
        'rewards': [None for i in range(max_nums[1])], 
        'dones': [None for i in range(max_nums[1])],
        'values': [None for i in range(max_nums[1])], 
        'logps': [None for i in range(max_nums[1])],
        'ids': range(max_nums[1]), 
    }
    if 'mf' in models[1].name:
        prey_buffer['meanaction'] = np.tile(former_meanaction[1], (max_nums[1], 1))

    if train:
        models[0].flush_buffer(**predator_buffer)
        models[1].flush_buffer(**prey_buffer)

        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]/max_steps

    return mean_rewards


def test(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    max_nums = [20, 40]

    action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]

    all_obs = env.reset()
    obs = [all_obs[:20], all_obs[20:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],action_dim[i]), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums[0]))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_meanaction = [np.zeros((1, 9)), np.zeros((1, 9))]

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
            acts[i], values[i], logprobs[i] = models[i].act(state=obs[i], meanaction=former_meanaction[i])
        ## random predator
        # acts[0] = np.random.rand(20,2)*2-1  

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:20], all_obs[20:]]
        rewards = [all_rewards[:20], all_rewards[20:]]
        done = all(all_done)

        #############################
        # Calculate mean action #
        #############################
        for i in range(n_group):
            former_meanaction[i] = _calc_moment(np.mean(acts[i], axis=0))

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            total_rewards[i].append(sum_reward)
        
        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]/max_steps

    return mean_rewards
